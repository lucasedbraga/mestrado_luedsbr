#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBUC (Profit-Based Unit Commitment) multi-período para Genco em mercado zonal.
VERSÃO AC (corrente alternada) — usa AC_BalanceConstraints, AC_LineConstraints
e ThermalGeneratorConstraints para o OPF; mantém as variáveis binárias de UC
e os contratos com multas.

Funciona em dois modos:
  • COM seção MERCADO no input  → PBUC completo (receita zonal + contratos + UC).
  • SEM seção MERCADO no input  → UC + OPF puro (mínimo custo quadrático).

Solvers recomendados (MINLP):
  - BONMIN  (open source, Coin-OR)
  - SCIP    (open source)
  - Couenne (open source)
  - BARON / Gurobi 9+ (comerciais, para NLP+MINLP)

Uso típico:
    db = PBUC_DBHandler('DATA/output/pbuc.db')
    db.create_tables()
    modelo = PBUC_TimeCoupled(sistema, mercado, db_handler=db, cen_id='c1',
                              use_dcopf=False)   # AC
    modelo.solve_timecoupled(solver_name='ipopt',
                             lambda_zona_previsto=precos_hgb)
    resultado = modelo.get_pbuc_result()
"""

import os
import sys
import json
import numpy as np
import pyomo.environ as pyo
from typing import List, Optional, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Handler (opcional) ---
try:
    from PBUC_ResultHandler import PBUC_DBHandler, PBUC_ResultExtractor
    _HANDLER_AVAILABLE = True
except ImportError:
    try:
        from SOLVER.PBUC_ResultHandler import PBUC_DBHandler, PBUC_ResultExtractor
        _HANDLER_AVAILABLE = True
    except ImportError:
        PBUC_DBHandler = None
        PBUC_ResultExtractor = None
        _HANDLER_AVAILABLE = False

# --- Classes de restrições (com fallbacks de path) ---
try:
    from RES.BalanceConstraints import AC_BalanceConstraints
    from RES.LineConstraints import AC_LineConstraints
    from RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
    from RES.MarketConstraints import MarketConstraints
except ImportError:
    AC_BalanceConstraints = None
    AC_LineConstraints = None
    ThermalGeneratorConstraints = None
    print("⚠ Classes de restrições AC não encontradas. use_dcopf=False falhará.")


class PBUC_TimeCoupled:
    """
    Profit-Based Unit Commitment com OPF (AC completo ou DC linearizado).
    Objetivo: Maximizar Lucro = Receita(zonal + contratos) - Custos
    (ou minimizar custo, se não houver seção MERCADO no input).
    """

    # ==================================================================
    # Construtor
    # ==================================================================
    def __init__(self,
                 sistema,
                 json_market: dict,
                 n_horas: int = 24,
                 n_dias: int = 1,
                 db_handler=None,
                 cen_id: Optional[str] = None,
                 dia_inicial: int = 0,
                 use_dcopf: bool = False,
                 auto_save: bool = True):

        self.sistema = sistema
        self.market = json_market
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.horizon_time = n_horas * n_dias
        self.db_handler = db_handler
        self.cen_id = cen_id
        self.dia_inicial = dia_inicial
        self.use_dcopf = use_dcopf
        self.auto_save = auto_save

        self.model = None
        self._solved = False
        self._result_dataclass = None

        # --- Detecta se o input traz a seção MERCADO ---
        self.has_market = (
            isinstance(self.market, dict)
            and isinstance(self.market.get('MERCADO'), dict)
            and len(self.market['MERCADO']) > 0
        )
        if not self.has_market:
            print("  ℹ Input sem seção 'MERCADO' → PBUC sem variáveis/restrições de mercado.")

        # --- Leitura dos arrays do sistema (parâmetros) ---
        self._build_SistemaEletrico_arrays()

        # --- Mapas de mercado (zonas, ofertas, contratos) ---
        self._parse_market_data()

        # --- Matriz de admitância (sempre necessária para AC) ---
        self._monta_matriz_admitancia()

        # --- Dicionários de variáveis ---
        self.var_lists: Dict[str, List] = {}
        self.PGER_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.QGER_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.PGWIND_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.CURTAILMENT_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.DEFICIT_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.V_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.ANG_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.FLOW_dict: Dict[Tuple[int, int], pyo.Var] = {}

        # Unit Commitment
        self.U_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.V_start_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.W_stop_dict: Dict[Tuple[int, int], pyo.Var] = {}

        # Contratos
        self.CONTRATO_ENTREGA_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.CONTRATO_DEFICIT_dict: Dict[Tuple[int, int], pyo.Var] = {}

        # Parâmetros exógenos
        self.PLOAD: Optional[np.ndarray] = None
        self.QLOAD: Optional[np.ndarray] = None
        self.PGWIND_AVAIL: Optional[np.ndarray] = None
        self.LAMBDA_ZONA: Optional[np.ndarray] = None

    # ==================================================================
    # 1. LEITURA DOS PARÂMETROS DO SISTEMA
    # ==================================================================
    def _build_SistemaEletrico_arrays(self):
        """Lê os atributos do sistema exatamente como as classes AC esperam."""
        s = self.sistema
        self.NBAR = int(s.NBAR)
        self.NUTE = int(s.NGER_UTE)
        self.NGWD = int(s.NGER_GWD)
        self.NLIN = int(s.NLIN)

        # Barramentos dos geradores térmicos e eólicos (0-based)
        self.thermal_bus = np.array(s.BAR_PGER_UTE, dtype=int)
        self.wind_bus = np.array(
            getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [])),
            dtype=int)

        # Linhas e matriz de admitância
        self.line_from = np.array(s.line_fr, dtype=int)
        self.line_to = np.array(s.line_to, dtype=int)
        self.line_r = np.array(s.r_line, dtype=float)
        self.line_x = np.array(s.x_line, dtype=float)
        self.line_flow_max = np.array(s.FLIM, dtype=float)
        self.line_b = np.array(getattr(s, 'Bsh', np.zeros(self.NLIN)), dtype=float)

        # Limites térmicos de ativa
        self.thermal_pmin = np.array(s.PGER_MIN_UTE, dtype=float)
        self.thermal_pmax = np.array(s.PGER_MAX_UTE, dtype=float)

        if hasattr(s, 'QGER_MIN_UTE') and hasattr(s, 'QGER_MAX_UTE'):
            self.thermal_qmin = np.array(s.QGER_MIN_UTE, dtype=float)
            self.thermal_qmax = np.array(s.QGER_MAX_UTE, dtype=float)
        else:
            self.thermal_qmin = -self.thermal_pmax.copy()
            self.thermal_qmax = self.thermal_pmax.copy()

        # ======================================================
        # Custos — quadráticos por gerador
        #   C_g(P, u) = a_g·P² + b_g·P + c_g·u  (+ startup em V_start)
        # ======================================================
        self.thermal_cost_quad = np.array(
            getattr(s, 'custo_GER_quad',
                    getattr(s, 'custo_quadratico_a', [0.0] * self.NUTE)),
            dtype=float)

        self.thermal_cost_lin = np.array(
            getattr(s, 'custo_GER_linear',
                    getattr(s, 'custo_GER', [50.0] * self.NUTE)),
            dtype=float)

        self.thermal_cost_base = np.array(
            getattr(s, 'custo_GER_base',
                    getattr(s, 'custo_base', [0.0] * self.NUTE)),
            dtype=float)

        self.startup_cost = np.array(
            getattr(s, 'custo_GER_startup',
                    getattr(s, 'custo_startup', [10.0] * self.NUTE)),
            dtype=float)

        # Alias retrocompatível
        self.thermal_cost = self.thermal_cost_lin.copy()

        # Rampas
        self.ramp_up = np.array(
            getattr(s, 'RAMP_UP', [0.5] * self.NUTE), dtype=float)
        self.ramp_down = np.array(
            getattr(s, 'RAMP_DOWN', [0.5] * self.NUTE), dtype=float)

        # Geração inicial
        if hasattr(s, 'PGER_INICIAL_UTE'):
            self.pger_inicial = np.array(s.PGER_INICIAL_UTE, dtype=float)
        else:
            self.pger_inicial = self.thermal_pmin.copy()

        # Tempos mínimos
        self.min_up = np.array(
            getattr(s, 'MIN_UP', [1] * self.NUTE), dtype=int)
        self.min_down = np.array(
            getattr(s, 'MIN_DOWN', [1] * self.NUTE), dtype=int)
        self.u_inicial = np.array(
            getattr(s, 'U_INICIAL', [1] * self.NUTE), dtype=int)

        # Slack e cargas-base
        self.slack_bus = int(getattr(s, 'slack_idx', 0))
        self.base_Pload = np.array(s.PLOAD, dtype=float)
        self.base_Qload = (np.array(s.QLOAD, dtype=float)
                           if hasattr(s, 'QLOAD')
                           else np.zeros(self.NBAR))

    # ==================================================================
    # 2. PARSE DO MERCADO
    # ==================================================================
    def _parse_market_data(self):
        # ---- Zonas (opcional) ----
        self.zonas = {z['ID_Zona']: z for z in self.market.get('ZONAS', [])}
        if self.zonas:
            self.zona_list = list(self.zonas.keys())
        else:
            self.zona_list = ['zona_unica']
        self.NZONAS = len(self.zona_list)
        self.zona_idx = {z: i for i, z in enumerate(self.zona_list)}

        self.barra_zona = {}
        if self.zonas:
            for zid, zdata in self.zonas.items():
                for b in zdata.get('Barras_Associadas', []):
                    self.barra_zona[int(b) - 1] = self.zona_idx[zid]
        else:
            for b in range(self.NBAR):
                self.barra_zona[b] = 0

        # ---- Sem MERCADO: atalho e saída ----
        if not self.has_market:
            self.ofertas_geracao  = []
            self.mapa_geracao     = {}
            self.bid_price_gen    = np.zeros(self.NUTE)
            self.ofertas_demanda  = []
            self.mapa_demanda     = {}
            self.contratos        = []
            self.NCONTRATOS       = 0
            self.custo_deficit_global = 50.0
            return

        # ---- Ofertas de geração (bids) ----
        self.ofertas_geracao = self.market['MERCADO']['ofertas_geracao']
        self.mapa_geracao = {
            m['ID_Oferta']: m['Geradores_Fisicos']
            for m in self.market['MERCADO']['mapa_geracao']
        }
        self.bid_price_gen = np.zeros(self.NUTE)
        for bid in self.ofertas_geracao:
            for ger_fis in self.mapa_geracao.get(bid['ID_Oferta'], []):
                try:
                    num = int(ger_fis[1:])
                    for idx, bus in enumerate(self.thermal_bus):
                        if bus + 1 == num:
                            self.bid_price_gen[idx] = bid['Preco_EUR_MWh']
                except (ValueError, IndexError):
                    pass

        # ---- Ofertas de demanda (contratos) ----
        self.ofertas_demanda = self.market['MERCADO']['ofertas_demanda']
        self.mapa_demanda = {
            m['ID_Oferta']: m['Demandas_Fisicas']
            for m in self.market['MERCADO']['mapa_demanda']
        }
        self.contratos = []
        for d in self.ofertas_demanda:
            self.contratos.append({
                'ID': d['ID_Oferta'],
                'Zona': d['ID_Zona'],
                'Amount': d['Amount_MWh'] / self.sistema.SB,
                'Preco': d['Preco_EUR_MWh'],
                'Multa': d['Preco_EUR_MWh'] * 2.0,
            })
        self.NCONTRATOS = len(self.contratos)

        self.custo_deficit_global = 50.0

    # ==================================================================
    # 3. MATRIZ DE ADMITÂNCIA (Ybus)
    # ==================================================================
    def _monta_matriz_admitancia(self):
        """Constrói G e B (partes real e imaginária da Ybus) para AC."""
        self.G = np.zeros((self.NBAR, self.NBAR))
        self.B = np.zeros((self.NBAR, self.NBAR))
        for LIN in range(self.NLIN):
            i, j = self.line_from[LIN], self.line_to[LIN]
            r, x = self.line_r[LIN], self.line_x[LIN]
            z2 = r * r + x * x
            if z2 == 0:
                continue
            g, b = r / z2, -x / z2
            self.G[i, i] += g; self.G[j, j] += g
            self.B[i, i] += b; self.B[j, j] += b
            self.G[i, j] -= g; self.G[j, i] -= g
            self.B[i, j] -= b; self.B[j, i] -= b
            b_sh = self.line_b[LIN] / 2.0
            if b_sh != 0:
                self.B[i, i] += b_sh
                self.B[j, j] += b_sh

    # ==================================================================
    # 4. PREVISÕES EXÓGENAS
    # ==================================================================
    def _processa_previsoes(self,
                            fator_carga=None,
                            fator_vento=None,
                            lambda_zona_previsto=None):
        T = self.horizon_time

        # --- Carga ---
        if fator_carga is None:
            fc = np.ones((T, self.NBAR))
        else:
            fc = np.asarray(fator_carga, dtype=float)
            if fc.ndim == 1:
                fc = fc[:, np.newaxis] * np.ones((1, self.NBAR))
            elif fc.ndim == 2 and fc.shape == (T, self.NBAR):
                pass
            elif fc.ndim == 2 and fc.shape[0] == T:
                fc = np.repeat(fc[:, :1], self.NBAR, axis=1)
            else:
                raise ValueError(f"fator_carga shape inválido: {fc.shape}")

        self.PLOAD = self.base_Pload[np.newaxis, :] * fc
        self.QLOAD = self.base_Qload[np.newaxis, :] * fc

        # --- Vento ---
        if self.NGWD == 0:
            self.PGWIND_AVAIL = np.zeros((T, 0))
        else:
            if fator_vento is None:
                fv = np.ones((T, self.NGWD))
            else:
                fv = np.asarray(fator_vento, dtype=float)
                if fv.ndim == 1:
                    fv = fv[:, np.newaxis] * np.ones((1, self.NGWD))
                elif fv.ndim == 2 and fv.shape == (T, self.NGWD):
                    pass
                elif fv.ndim == 2 and fv.shape[0] == T:
                    fv = np.repeat(fv[:, :1], self.NGWD, axis=1)
                else:
                    raise ValueError(f"fator_vento shape inválido: {fv.shape}")
            base_wind = np.array(
                getattr(self.sistema, 'PGWD_MAX_ORIGINAL',
                        [1.0] * self.NGWD))
            self.PGWIND_AVAIL = base_wind[np.newaxis, :] * fv

        # --- Preços zonais (só se houver mercado) ---
        if not self.has_market:
            self.LAMBDA_ZONA = None
            return

        if lambda_zona_previsto is None:
            self.LAMBDA_ZONA = np.full((T, self.NZONAS), 3.0)
        else:
            self.LAMBDA_ZONA = np.asarray(lambda_zona_previsto, dtype=float)
            if self.LAMBDA_ZONA.shape != (T, self.NZONAS):
                raise ValueError(
                    f"lambda_zona_previsto shape inválido: "
                    f"{self.LAMBDA_ZONA.shape} (esperado ({T},{self.NZONAS}))")

    # ==================================================================
    # 5. VARIÁVEIS
    # ==================================================================
    def _add_VARS(self):
        T = self.horizon_time
        m = self.model

        # --- 5.1 Tensão e ângulo (AC) ---
        for t in range(T):
            for b in range(self.NBAR):
                v = pyo.Var(bounds=(0.9, 1.1), initialize=1.0)
                setattr(m, f"V_T{t}_B{b+1}", v)
                self.V_dict[(t, b)] = v

                a = pyo.Var(bounds=(-np.pi, np.pi), initialize=0.0)
                setattr(m, f"ANG_T{t}_B{b+1}", a)
                self.ANG_dict[(t, b)] = a

            setattr(m, f"fix_slack_ang_T{t}",
                    pyo.Constraint(expr=self.ANG_dict[(t, self.slack_bus)] == 0.0))
            setattr(m, f"fix_slack_V_T{t}",
                    pyo.Constraint(expr=self.V_dict[(t, self.slack_bus)] == 1.0))

        # --- 5.2 Potência ativa e reativa dos geradores térmicos ---
        for t in range(T):
            for g in range(self.NUTE):
                p = pyo.Var(bounds=(0.0, self.thermal_pmax[g]), initialize=0.0)
                setattr(m, f"PGER_T{t}_G{g+1}", p)
                self.PGER_dict[(t, g)] = p

                q = pyo.Var(
                    bounds=(self.thermal_qmin[g], self.thermal_qmax[g]),
                    initialize=0.0)
                setattr(m, f"QGER_T{t}_G{g+1}", q)
                self.QGER_dict[(t, g)] = q

        # --- 5.3 Unit Commitment (binárias) ---
        for t in range(T):
            for g in range(self.NUTE):
                u = pyo.Var(domain=pyo.Binary, initialize=int(self.u_inicial[g]))
                v = pyo.Var(domain=pyo.Binary, initialize=0)
                w = pyo.Var(domain=pyo.Binary, initialize=0)
                setattr(m, f"U_T{t}_G{g+1}", u)
                setattr(m, f"V_T{t}_G{g+1}", v)
                setattr(m, f"W_T{t}_G{g+1}", w)
                self.U_dict[(t, g)] = u
                self.V_start_dict[(t, g)] = v
                self.W_stop_dict[(t, g)] = w

        # --- 5.4 Déficit por barra ---
        for t in range(T):
            for b in range(self.NBAR):
                d = pyo.Var(bounds=(0, 1e3), initialize=0.0)
                setattr(m, f"DEF_T{t}_B{b+1}", d)
                self.DEFICIT_dict[(t, b)] = d

        # --- 5.5 Eólico + curtailment ---
        if self.NGWD > 0:
            for t in range(T):
                for w in range(self.NGWD):
                    avail = self.PGWIND_AVAIL[t, w]
                    p = pyo.Var(bounds=(0, avail), initialize=0.0)
                    c = pyo.Var(bounds=(0, avail), initialize=0.0)
                    setattr(m, f"PW_T{t}_W{w+1}", p)
                    setattr(m, f"CURT_T{t}_W{w+1}", c)
                    self.PGWIND_dict[(t, w)] = p
                    self.CURTAILMENT_dict[(t, w)] = c

        # --- 5.6 Contratos (somente se houver mercado) ---
        if self.has_market:
            for t in range(T):
                for c_idx, c in enumerate(self.contratos):
                    e = pyo.Var(bounds=(0, c['Amount']), initialize=c['Amount'])
                    d = pyo.Var(bounds=(0, c['Amount']), initialize=0.0)
                    setattr(m, f"CONT_ENT_T{t}_C{c_idx+1}", e)
                    setattr(m, f"CONT_DEF_T{t}_C{c_idx+1}", d)
                    self.CONTRATO_ENTREGA_dict[(t, c_idx)] = e
                    self.CONTRATO_DEFICIT_dict[(t, c_idx)] = d

    # ==================================================================
    # 6. RESTRIÇÕES
    # ==================================================================
    def _add_CONS(self):
        T = self.horizon_time
        m = self.model

        if self.NUTE > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_UTE=self.NUTE,
                PGER=self.PGER_dict,
                QGER=self.QGER_dict,
                PGER_MIN_UTE=self.thermal_pmin,
                PGER_MAX_UTE=self.thermal_pmax,
                QGER_MIN_UTE=self.thermal_qmin,
                QGER_MAX_UTE=self.thermal_qmax,
                PGER_INICIAL_UTE=self.sistema.PGER_INICIAL_UTE,
                RAMP_UP=self.sistema.RAMP_UP,
                RAMP_DOWN=self.sistema.RAMP_DOWN,
                SB=self.sistema.SB
            )

        AC_BalanceConstraints.add_constraints(
            model=m,
            sistema=self.sistema,
            HORA=T,
            G=self.G,
            B=self.B,
            V=self.V_dict,
            ANG=self.ANG_dict,
            PGER=self.PGER_dict,
            QGER=self.QGER_dict,
            PLOAD=self.PLOAD,
            QLOAD=self.QLOAD,
            DEFICIT=self.DEFICIT_dict,
            PGWIND=self.PGWIND_dict if self.NGWD > 0 else None,
            BESS_SOC_op=None,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            wind_gen_to_bar=(self.wind_bus.tolist()
                             if self.NGWD > 0 else None),
            battery_list=None,
        )

        AC_LineConstraints.add_constraints(
            model=m,
            sistema=self.sistema,
            HORA=T,
            V=self.V_dict,
            ANG=self.ANG_dict,
            line_from=self.line_from,
            line_to=self.line_to,
            line_r=self.line_r,
            line_x=self.line_x,
            line_flow_max=self.line_flow_max,
        )

        # ---- Contratos: só se houver mercado ----
        if self.has_market:
            MarketConstraints.add_constraints(
                model=m,
                T=T,
                contratos=self.contratos,
                CONTRATO_ENTREGA=self.CONTRATO_ENTREGA_dict,
                CONTRATO_DEFICIT=self.CONTRATO_DEFICIT_dict,
            )

    # ==================================================================
    # 7. FUNÇÃO OBJETIVO
    # ==================================================================
    def _add_FOB(self):
        """
        Maximizar Lucro = Receita(zonal + contratos) - Custos.
        Se não houver seção MERCADO, o objetivo é puro -Custo (UC + OPF).
        """
        T = self.horizon_time
        m = self.model
        expr = 0.0

        # ----- Receita zonal + contratos (só se houver mercado) -----
        if self.has_market:
            for t in range(T):
                for g in range(self.NUTE):
                    b = self.thermal_bus[g]
                    zona = self.barra_zona.get(b, 0)
                    expr += self.LAMBDA_ZONA[t, zona] * self.PGER_dict[(t, g)]
                for w in range(self.NGWD):
                    if w < len(self.wind_bus):
                        b = self.wind_bus[w]
                        zona = self.barra_zona.get(b, 0)
                        expr += self.LAMBDA_ZONA[t, zona] * self.PGWIND_dict[(t, w)]

            for t in range(T):
                for c_idx, c in enumerate(self.contratos):
                    expr += c['Preco'] * self.CONTRATO_ENTREGA_dict[(t, c_idx)]

        # ----- Custo de combustível: a·P² + b·P + c·u (sempre) -----
        for t in range(T):
            for g in range(self.NUTE):
                P = self.PGER_dict[(t, g)]
                u = self.U_dict[(t, g)]
                expr -= (self.thermal_cost_quad[g] * P * P
                         + self.thermal_cost_lin[g] * P
                         + self.thermal_cost_base[g] * u)

        # ----- Custo de startup (sempre) -----
        for t in range(T):
            for g in range(self.NUTE):
                expr -= self.startup_cost[g] * self.V_start_dict[(t, g)]

        # ----- Multa por não atendimento (só se houver mercado) -----
        if self.has_market:
            for t in range(T):
                for c_idx, c in enumerate(self.contratos):
                    expr -= c['Multa'] * self.CONTRATO_DEFICIT_dict[(t, c_idx)]

        # ----- Déficit global (sempre) -----
        for t in range(T):
            for b in range(self.NBAR):
                expr -= self.custo_deficit_global * self.DEFICIT_dict[(t, b)]

        # ----- Custo de oportunidade do curtailment (só se houver mercado) -----
        if self.has_market:
            for t in range(T):
                for w in range(self.NGWD):
                    if w < len(self.wind_bus):
                        b = self.wind_bus[w]
                        zona = self.barra_zona.get(b, 0)
                        expr -= self.LAMBDA_ZONA[t, zona] * self.CURTAILMENT_dict[(t, w)]

        m.FOB = pyo.Objective(expr=expr, sense=pyo.maximize)

    # ==================================================================
    # 8. BUILD + SOLVE
    # ==================================================================
    def _build_Cenario(self, fator_carga=None, fator_vento=None,
                       lambda_zona_previsto=None):
        self.model = pyo.ConcreteModel(name="PBUC_TimeCoupled")
        self._processa_previsoes(fator_carga, fator_vento, lambda_zona_previsto)
        self._add_VARS()
        self._add_CONS()
        self._add_FOB()
        self._solved = False
        self._result_dataclass = None

    def solve(self, solver_name: Optional[str] = None, tee: bool = True, **kwargs):
        """Resolve o modelo."""
        if self.model is None:
            raise RuntimeError("Modelo não construído.")

        opt = pyo.SolverFactory(solver_name)
        if not opt.available():
            raise RuntimeError(
                f"Solver '{solver_name}' não está disponível. "
                f"Instale-o ou passe outro via solver_name=.")

        results = opt.solve(self.model, tee=tee, **kwargs)
        self._solved = (
            results.solver.status == pyo.SolverStatus.ok and
            results.solver.termination_condition in (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.feasible))

        if not self._solved:
            print(f"⚠ Solver retornou: {results.solver.termination_condition}")
        return results

    def solve_timecoupled(self,
                          solver_name: Optional[str] = None,
                          fator_carga=None,
                          fator_vento=None,
                          lambda_zona_previsto=None,
                          tee: bool = True,
                          save_to_db: Optional[bool] = None):
        """Pipeline: build → solve → (opcional) save_to_db."""
        self._build_Cenario(fator_carga, fator_vento, lambda_zona_previsto)
        #self.model.pprint()
        results = self.solve(solver_name=solver_name, tee=tee)

        if save_to_db is None:
            save_to_db = self.auto_save
        if save_to_db:
            self._try_save_to_db()

        return results

    def _try_save_to_db(self) -> bool:
        if not self._solved or self.db_handler is None or self.cen_id is None:
            return False
        try:
            self.db_handler.save_pbuc_result(
                modelo_pbuc=self, sistema=self.sistema,
                mercado=self.market, cen_id=self.cen_id)
            return True
        except Exception as e:
            print(f"⚠ Falha ao salvar no banco: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ==================================================================
    # 9. EXTRAÇÃO
    # ==================================================================
    def extract_profit_report(self) -> dict:
        """Dict com lucro, receitas, custos e arrays (método legado)."""
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")
        T = self.horizon_time

        receita_mercado = receita_contratos = 0.0
        multa_contratos = 0.0
        custo_combustivel = 0.0   # a·P² + b·P
        custo_base        = 0.0   # c·u
        custo_startup     = 0.0
        custo_deficit     = 0.0
        custo_curtailment = 0.0

        schedule = np.zeros((T, self.NUTE))
        pg = np.zeros((T, self.NUTE))
        qg = np.zeros((T, self.NUTE))
        curtail = np.zeros((T, self.NGWD)) if self.NGWD > 0 else None

        for t in range(T):
            # --- Térmicos ---
            for g in range(self.NUTE):
                pg[t, g] = pyo.value(self.PGER_dict[(t, g)])
                qg[t, g] = pyo.value(self.QGER_dict[(t, g)])
                schedule[t, g] = pyo.value(self.U_dict[(t, g)])

                # Receita zonal (só se houver mercado)
                if self.has_market:
                    b_bus = self.thermal_bus[g]
                    zona = self.barra_zona.get(b_bus, 0)
                    receita_mercado += self.LAMBDA_ZONA[t, zona] * pg[t, g]

                # Custo quadrático + linear
                custo_combustivel += (
                    self.thermal_cost_quad[g] * pg[t, g] ** 2
                    + self.thermal_cost_lin[g] * pg[t, g]
                )
                # Termo constante (só se ligada)
                custo_base += self.thermal_cost_base[g] * schedule[t, g]
                # Startup
                custo_startup += self.startup_cost[g] * \
                                 pyo.value(self.V_start_dict[(t, g)])

            # --- Eólico ---
            if self.NGWD > 0:
                for w in range(self.NGWD):
                    curtail[t, w] = pyo.value(self.CURTAILMENT_dict[(t, w)])
                    if self.has_market and w < len(self.wind_bus):
                        b = self.wind_bus[w]
                        zona = self.barra_zona.get(b, 0)
                        receita_mercado += self.LAMBDA_ZONA[t, zona] * \
                                           pyo.value(self.PGWIND_dict[(t, w)])
                        custo_curtailment += self.LAMBDA_ZONA[t, zona] * \
                                             curtail[t, w]

            # --- Contratos (só se houver mercado) ---
            if self.has_market:
                for c_idx, c in enumerate(self.contratos):
                    entrega = pyo.value(self.CONTRATO_ENTREGA_dict[(t, c_idx)])
                    deficit = pyo.value(self.CONTRATO_DEFICIT_dict[(t, c_idx)])
                    receita_contratos += c['Preco'] * entrega
                    multa_contratos   += c['Multa'] * deficit

            # --- Déficit (sempre) ---
            for b in range(self.NBAR):
                custo_deficit += self.custo_deficit_global * \
                                 pyo.value(self.DEFICIT_dict[(t, b)])

        lucro = (receita_mercado + receita_contratos
                 - custo_combustivel - custo_base - custo_startup
                 - multa_contratos - custo_deficit - custo_curtailment)

        return {
            'lucro_total': lucro,
            'receita_mercado': receita_mercado,
            'receita_contratos': receita_contratos,
            'custo_combustivel': custo_combustivel,
            'custo_base': custo_base,
            'custo_startup': custo_startup,
            'multa_contratos': multa_contratos,
            'custo_deficit': custo_deficit,
            'custo_curtailment': custo_curtailment,
            'schedule_uc': schedule,
            'potencia_gerada': pg,
            'potencia_reativa': qg,
            'curtailment': curtail,
        }

    def get_pbuc_result(self):
        if self._result_dataclass is not None:
            return self._result_dataclass
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")
        if not _HANDLER_AVAILABLE or PBUC_ResultExtractor is None:
            raise ImportError("PBUC_ResultHandler não disponível.")
        extractor = PBUC_ResultExtractor(self, self.sistema, self.market)
        self._result_dataclass = extractor.extract()
        return self._result_dataclass

    def save_to_db(self, cen_id: Optional[str] = None) -> bool:
        if cen_id is not None:
            self.cen_id = cen_id
        return self._try_save_to_db()


# =============================================================================
# Exemplo de uso
# =============================================================================
if __name__ == "__main__":
    from UTILS.SystemLoader import SistemaLoader

    sistema = SistemaLoader("DATA/input/3barras_QUAD.json")
    with open("DATA/input/3barras_QUAD.json", "r") as f:
        mercado = json.load(f)

    db_handler = None
    cen_id = None
    if _HANDLER_AVAILABLE:
        db_handler = PBUC_DBHandler('DATA/output/pbuc_resultados.db')
        db_handler.create_tables()
        cen_id = 'cen_ac_' + str(np.random.randint(0, 9999))
        print(f"✓ Handler configurado, cen_id = {cen_id}")

    # ── Modo AC (MINLP) ──
    modelo = PBUC_TimeCoupled(
        sistema=sistema, json_market=mercado,
        n_horas=24, n_dias=1,
        use_dcopf=False,
        db_handler=db_handler,
        cen_id=cen_id,
        auto_save=True)

    T = 24
    fator_carga = np.ones(T)
    fator_vento = np.ones(T) * 0.8 if sistema.NGER_GWD > 0 else None

    # Preço zonal só é passado se houver mercado
    if modelo.has_market:
        NZ = len(mercado.get('ZONAS', [])) or 1
        base_lambda = np.array([3.5, 3.2, 2.8, 3.0][:NZ])
        lambda_zona = np.tile(base_lambda, (T, 1))
    else:
        lambda_zona = None

    print(f"Total demanda base : {sum(sistema.PLOAD):.2f} pu")
    print(f"Total Pmax geração : {sum(sistema.PGER_MAX_UTE):.2f} pu")
    print(f"Total Pmin geração : {sum(sistema.PGER_MIN_UTE):.2f} pu")
    print(f"Capacidade linhas  : {sum(sistema.FLIM):.2f} pu")
    print(f"Modo mercado       : "
          f"{'COM MERCADO' if modelo.has_market else 'SEM MERCADO (UC+OPF puro)'}")

    for b in range(sistema.NBAR):
        if sistema.PLOAD[b] > 0:
            cap = sum(sistema.FLIM[l] for l in range(sistema.NLIN)
                      if sistema.line_fr[l] == b or sistema.line_to[l] == b)
            if cap < sistema.PLOAD[b]:
                print(f"⚠ Barra {b+1}: carga={sistema.PLOAD[b]:.2f} > "
                      f"cap_local={cap:.2f}")

    results = modelo.solve_timecoupled(
        solver_name='ipopt',       # MINLP: 'ipopt', 'bonmin', 'couenne'
        fator_carga=fator_carga,
        fator_vento=fator_vento,
        lambda_zona_previsto=lambda_zona,
        tee=True)

    relatorio = modelo.extract_profit_report()
    print("\n" + "=" * 60)
    print("RELATÓRIO PBUC (AC)")
    print("=" * 60)
    print(f"Lucro Total         : € {relatorio['lucro_total']:,.2f}")
    print(f"Receita Mercado     : € {relatorio['receita_mercado']:,.2f}")
    print(f"Receita Contratos   : € {relatorio['receita_contratos']:,.2f}")
    print(f"Custo Combustível   : € {relatorio['custo_combustivel']:,.2f}")
    print(f"Custo Base (c·u)    : € {relatorio['custo_base']:,.2f}")
    print(f"Custo Startup       : € {relatorio['custo_startup']:,.2f}")
    print(f"Multa Contratos     : € {relatorio['multa_contratos']:,.2f}")
    print(f"Custo Déficit       : € {relatorio['custo_deficit']:,.2f}")
    print(f"Custo Curtailment   : € {relatorio['custo_curtailment']:,.2f}")
    print("\nSchedule UC (linhas=horas, colunas=geradores):")
    print(relatorio['schedule_uc'])

    if _HANDLER_AVAILABLE:
        try:
            resultado = modelo.get_pbuc_result()
            from PBUC_ResultHandler import PBUC_Analyzer, PBUC_Plotter
            analisador = PBUC_Analyzer(resultado)
            print("\n" + analisador.resumo_executivo())
            plotter = PBUC_Plotter(
                db_handler, cen_id,
                output_dir=f"DATA/output/figuras/{cen_id}")
            plotter.plot_all()
        except Exception as e:
            print(f"⚠ Falha na análise/plotagem: {e}")
            import traceback
            traceback.print_exc()