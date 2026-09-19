#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de otimização multi-período acoplado (UC + AC OPF) com Pyomo.
Suporta custo quadrático (a·P² + b·P + c·u) + startup + rampas + min up/down.
Todas as grandezas em pu, variáveis indexadas por (t, idx).
"""

import os
import sys
import traceback
import numpy as np
import pyomo.environ as pyo
from typing import List, Union, Optional, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_AC.RES.BatteryConstraints import BatteryConstraints
from SOLVER.OPF_AC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_AC.RES.BalanceConstraints import AC_BalanceConstraints
from SOLVER.OPF_AC.RES.LineConstraints import AC_LineConstraints
from SOLVER.OPF_AC.RES.ControlBusConstraints import VoltageControlConstraints
from DB.DBmodel_OPF import TimeCoupled_OPF_Result, OPF_SnapshotResult


class ACOPF_TimeCoupled:
    """
    Modelo UC + AC-OPF multi-período.

    Variáveis por (t, g):
        PGER, QGER, U, V_start, W_stop
    Custo do gerador g no período t:
        a_g · P² + b_g · P + c_g · U + startup_g · V_start
    """

    # ------------------------------------------------------------------
    # Construtor
    # ------------------------------------------------------------------
    def __init__(self,
                 sistema,
                 n_horas: int = 24,
                 n_dias: int = 1,
                 db_handler=None,
                 dia_inicial: int = 0):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.horizon_time = n_horas * n_dias
        self.db_handler = db_handler
        self.dia_inicial = dia_inicial

        self.model = None
        self._solved = False

        self._build_SistemaEletrico_arrays()
        self._monta_matiz_admitancia()

        # Dicionários de variáveis
        self.var_lists: Dict[str, List] = {}
        self.var_indices: Dict[str, Dict] = {}

        self.PGER_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.QGER_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.U_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.V_start_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.W_stop_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.PGWIND_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.CURTAILMENT_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.DEFICIT_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.V_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.ANG_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.CHARGE_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.DISCHARGE_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.SOC_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.BatteryOperation_dict: Dict[Tuple[int, int], pyo.Var] = {}

        self.PLOAD: Optional[np.ndarray] = None
        self.QLOAD: Optional[np.ndarray] = None
        self.PGWIND_AVAIL: Optional[np.ndarray] = None
        self.soc_inicial_list: List[float] = []
        self.soc_final_list: List[float] = []

    # ------------------------------------------------------------------
    # 1. Arrays do sistema — inclui custos quadráticos/startup e UC
    # ------------------------------------------------------------------
    def _build_SistemaEletrico_arrays(self):
        s = self.sistema
        self.NBAR = s.NBAR
        self.NUTE = s.NGER_UTE
        self.NGWD = s.NGER_GWD
        self.NLIN = s.NLIN
        self.NBESS = len(getattr(s, 'BARRAS_COM_BATERIA', []))

        self.thermal_bus = np.array(s.BAR_PGER_UTE, dtype=int)
        self.wind_bus = np.array(
            getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [])), dtype=int)

        self.line_from = np.array(s.line_fr, dtype=int)
        self.line_to = np.array(s.line_to, dtype=int)
        self.line_r = np.array(s.r_line, dtype=float)
        self.line_x = np.array(s.x_line, dtype=float)
        self.line_flow_max = np.array(s.FLIM, dtype=float)

        self.thermal_pmin = np.array(s.PGER_MIN_UTE, dtype=float)
        self.thermal_pmax = np.array(s.PGER_MAX_UTE, dtype=float)

        # ---- Custos quadráticos ----
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
        # alias retrocompatível
        self.thermal_cost = self.thermal_cost_lin.copy()

        # ---- Reativos ----
        if hasattr(s, 'QGER_MIN_UTE') and hasattr(s, 'QGER_MAX_UTE'):
            self.thermal_qmin = np.array(s.QGER_MIN_UTE, dtype=float)
            self.thermal_qmax = np.array(s.QGER_MAX_UTE, dtype=float)
        else:
            self.thermal_qmin = 0 * self.thermal_pmax
            self.thermal_qmax = 0 * self.thermal_pmax

        if hasattr(s, 'V_ESP_BUS'):
            self.V_ESP = s.V_ESP_BUS
        else:
            self.V_ESP = 0

        # ---- Rampas ----
        self.ramp_up = np.array(
            getattr(s, 'RAMP_UP', [0.5] * self.NUTE), dtype=float)
        self.ramp_down = np.array(
            getattr(s, 'RAMP_DOWN', [0.5] * self.NUTE), dtype=float)

        # ---- Geração inicial ----
        if hasattr(s, 'PGER_INICIAL_UTE'):
            self.pger_inicial = np.array(s.PGER_INICIAL_UTE, dtype=float)
        else:
            self.pger_inicial = self.thermal_pmin.copy()

        # ---- Tempos mínimos / estados iniciais ----
        self.min_up = np.array(
            getattr(s, 'MIN_UP', [1] * self.NUTE), dtype=int)
        self.min_down = np.array(
            getattr(s, 'MIN_DOWN', [1] * self.NUTE), dtype=int)
        self.u_inicial = np.array(
            getattr(s, 'U_INICIAL', [1] * self.NUTE), dtype=int)

        # ---- Baterias ----
        self.battery_buses = np.array(
            getattr(s, 'BARRAS_COM_BATERIA', []), dtype=int)
        if self.NBESS > 0:
            self.battery_capacity = np.array(
                [s.BATTERY_CAPACITY[b] for b in self.battery_buses], dtype=float)
            self.battery_power_limit = np.array(
                [s.BATTERY_POWER_LIMIT[b] for b in self.battery_buses], dtype=float)
            self.battery_min_soc_frac = np.array(
                [s.BATTERY_MIN_SOC[b] for b in self.battery_buses], dtype=float)
            self.battery_charge_eff = getattr(s, 'BATTERY_CHARGE_EFF', 1.0)
            self.battery_discharge_eff = getattr(s, 'BATTERY_DISCHARGE_EFF', 1.0)
        else:
            self.battery_capacity = np.array([])
            self.battery_power_limit = np.array([])
            self.battery_min_soc_frac = np.array([])

        self.slack_bus = getattr(s, 'slack_idx', 0)
        self.base_Pload = np.array(s.PLOAD, dtype=float)
        self.base_Qload = (np.array(s.QLOAD, dtype=float)
                           if hasattr(s, 'QLOAD')
                           else np.zeros(self.NBAR))

        # ---- Diagnóstico rápido de factibilidade de rampa ----
        for g in range(self.NUTE):
            if self.u_inicial[g] == 1 and self.pger_inicial[g] < self.thermal_pmin[g] - 1e-6:
                print(f"  ⚠ G{g+1}: PGER_INICIAL={self.pger_inicial[g]:.3f} "
                      f"< Pmin={self.thermal_pmin[g]:.3f} "
                      f"→ ajuste o input ou o modelo pode ficar infactível.")

    # ------------------------------------------------------------------
    # 2. Ybus
    # ------------------------------------------------------------------
    def _monta_matiz_admitancia(self):
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

    # ------------------------------------------------------------------
    # 3. Fatores e previsões
    # ------------------------------------------------------------------
    def _processa_multiplicadores(self,
                                  fator_carga=None,
                                  fator_vento=None,
                                  soc_inicial=0.5,
                                  soc_final=None):
        T = self.horizon_time

        # =========================================================
        # CARGA — aceita shapes:
        #   (n_dias, n_horas, n_barras) → (T, n_barras)
        #   (n_dias, n_horas)           → (T, n_barras) broadcast
        #   (T,)                        → (T, n_barras) broadcast
        #   (T, n_barras)               → como está
        #   escalar                     → (T, n_barras) constante
        # =========================================================
        if fator_carga is None:
            fc = np.ones((T, self.NBAR))
        else:
            fc = np.asarray(fator_carga, dtype=float)

            if fc.ndim == 3:
                # (n_dias, n_horas, n_barras) ou (n_dias, n_horas, 1)
                if fc.shape[0] == self.n_dias and fc.shape[1] == self.n_horas:
                    fc = fc.reshape((T, fc.shape[2]))
                    if fc.shape[1] != self.NBAR:
                        if fc.shape[1] == 1:
                            fc = np.repeat(fc, self.NBAR, axis=1)
                        else:
                            raise ValueError(
                                f"fator_carga 3D com {fc.shape[1]} barras; "
                                f"esperado 1 ou {self.NBAR}")
                else:
                    raise ValueError(
                        f"fator_carga 3D com shape {fc.shape}; "
                        f"esperado ({self.n_dias}, {self.n_horas}, nb)")

            elif fc.ndim == 2:
                if fc.shape == (T, self.NBAR):
                    pass                                  # já OK
                elif fc.shape[0] == T and fc.shape[1] == 1:
                    fc = np.repeat(fc, self.NBAR, axis=1)
                elif fc.shape[0] == self.n_dias and fc.shape[1] == self.n_horas:
                    fc = np.repeat(fc.reshape((T, 1)), self.NBAR, axis=1)
                elif fc.shape[0] == self.n_horas and self.n_dias == 1:
                    fc = np.repeat(fc.reshape((T, 1)), self.NBAR, axis=1)
                else:
                    raise ValueError(f"fator_carga 2D shape inválido: {fc.shape}")

            elif fc.ndim == 1:
                if fc.size == T:
                    fc = fc[:, np.newaxis] * np.ones((1, self.NBAR))
                elif fc.size == self.NBAR:
                    fc = np.ones((T, 1)) * fc[np.newaxis, :]
                elif fc.size == self.n_horas and self.n_dias == 1:
                    fc = fc[:, np.newaxis] * np.ones((1, self.NBAR))
                else:
                    raise ValueError(f"fator_carga 1D shape inválido: {fc.shape}")

            else:
                raise ValueError(f"fator_carga ndim inválido: {fc.ndim}")

        self.PLOAD = self.base_Pload[np.newaxis, :] * fc
        self.QLOAD = self.base_Qload[np.newaxis, :] * fc

        # =========================================================
        # VENTO — mesma lógica para PGWIND (n_dias, n_horas, n_wind)
        # =========================================================
        if self.NGWD == 0:
            self.PGWIND_AVAIL = np.zeros((T, 0))
        else:
            if fator_vento is None:
                fv = np.ones((T, self.NGWD))
            else:
                fv = np.asarray(fator_vento, dtype=float)

                if fv.ndim == 3:
                    if fv.shape[0] == self.n_dias and fv.shape[1] == self.n_horas:
                        fv = fv.reshape((T, fv.shape[2]))
                        if fv.shape[1] != self.NGWD:
                            if fv.shape[1] == 1:
                                fv = np.repeat(fv, self.NGWD, axis=1)
                            else:
                                raise ValueError(
                                    f"fator_vento 3D com {fv.shape[1]} eólicos; "
                                    f"esperado 1 ou {self.NGWD}")
                    else:
                        raise ValueError(
                            f"fator_vento 3D shape inválido: {fv.shape}")

                elif fv.ndim == 2:
                    if fv.shape == (T, self.NGWD):
                        pass
                    elif fv.shape[0] == T and fv.shape[1] == 1:
                        fv = np.repeat(fv, self.NGWD, axis=1)
                    elif fv.shape[0] == self.n_dias and fv.shape[1] == self.n_horas:
                        fv = np.repeat(fv.reshape((T, 1)), self.NGWD, axis=1)
                    else:
                        raise ValueError(f"fator_vento 2D shape inválido: {fv.shape}")

                elif fv.ndim == 1:
                    if fv.size == T:
                        fv = fv[:, np.newaxis] * np.ones((1, self.NGWD))
                    elif fv.size == self.NGWD:
                        fv = np.ones((T, 1)) * fv[np.newaxis, :]
                    else:
                        raise ValueError(f"fator_vento 1D shape inválido: {fv.shape}")

                else:
                    raise ValueError(f"fator_vento ndim inválido: {fv.ndim}")

            base_wind = np.array(
                getattr(self.sistema, 'PGWD_MAX_ORIGINAL',
                        getattr(self.sistema, 'PGWIND_disponivel',
                                [1.0] * self.NGWD)))
            self.PGWIND_AVAIL = base_wind[np.newaxis, :] * fv

        # =========================================================
        # SOC das baterias
        # =========================================================
        self.soc_inicial_list = []
        self.soc_final_list = []
        if self.NBESS > 0:
            soc_ini_frac = ([soc_inicial] * self.NBESS
                            if isinstance(soc_inicial, (float, int))
                            else list(soc_inicial))
            self.soc_inicial_list = [
                soc_ini_frac[i] * self.battery_capacity[i]
                for i in range(self.NBESS)]
            if soc_final is not None:
                soc_fin_frac = ([soc_final] * self.NBESS
                                if isinstance(soc_final, (float, int))
                                else list(soc_final))
                self.soc_final_list = [
                    soc_fin_frac[i] * self.battery_capacity[i]
                    for i in range(self.NBESS)]
    # ------------------------------------------------------------------
    # 4. Variáveis — incluindo UC
    # ------------------------------------------------------------------
    def _add_VARS(self):
        T = self.horizon_time

        # --- V, ANG ---
        for t in range(T):
            for b in range(self.NBAR):
                v = pyo.Var(bounds=(0.95, 1.05), initialize=1.0)
                setattr(self.model, f"V_pu_T{t}_BAR{b+1}", v)
                self.V_dict[(t, b)] = v

                a = pyo.Var(bounds=(-np.pi, np.pi), initialize=0.0)
                setattr(self.model, f"ANG_pu_T{t}_BAR{b+1}", a)
                self.ANG_dict[(t, b)] = a
            setattr(self.model, f"fix_slack_angle_T{t}",
                    pyo.Constraint(expr=self.ANG_dict[(t, self.slack_bus)] == 0.0))

        # --- PGER, QGER, U, V_start, W_stop ---
        for t in range(T):
            for g in range(self.NUTE):
                # Potência ativa: apenas limite superior global; o limite
                # inferior efetivo é imposto via restrição com U.
                p = pyo.Var(bounds=(0.0, self.thermal_pmax[g]), initialize=0.0)
                setattr(self.model, f"PGER_UTE_T{t}_G{g+1}", p)
                self.PGER_dict[(t, g)] = p

                q = pyo.Var(bounds=(self.thermal_qmin[g], self.thermal_qmax[g]),
                            initialize=0.0)
                setattr(self.model, f"QGER_UTE_T{t}_G{g+1}", q)
                self.QGER_dict[(t, g)] = q

                u = pyo.Var(domain=pyo.Binary,
                            initialize=int(self.u_inicial[g]))
                vst = pyo.Var(domain=pyo.Binary, initialize=0)
                wst = pyo.Var(domain=pyo.Binary, initialize=0)
                setattr(self.model, f"U_T{t}_G{g+1}", u)
                setattr(self.model, f"Vstart_T{t}_G{g+1}", vst)
                setattr(self.model, f"Wstop_T{t}_G{g+1}", wst)
                self.U_dict[(t, g)] = u
                self.V_start_dict[(t, g)] = vst
                self.W_stop_dict[(t, g)] = wst

        # --- Déficit ---
        for t in range(T):
            for b in range(self.NBAR):
                d = pyo.Var(bounds=(0, 1e6), initialize=0.0)
                setattr(self.model, f"DEFICIT_T{t}_BAR{b+1}", d)
                self.DEFICIT_dict[(t, b)] = d

        # --- Eólico ---
        if self.NGWD > 0:
            for t in range(T):
                for w in range(self.NGWD):
                    avail = self.PGWIND_AVAIL[t, w]
                    pw = pyo.Var(bounds=(0, avail), initialize=0.0)
                    cu = pyo.Var(bounds=(0, avail), initialize=0.0)
                    setattr(self.model, f"PGWD_T{t}_W{w}", pw)
                    setattr(self.model, f"CURTAIL_T{t}_W{w}", cu)
                    self.PGWIND_dict[(t, w)] = pw
                    self.CURTAILMENT_dict[(t, w)] = cu

        # --- Baterias ---
        if self.NBESS > 0:
            for t in range(T):
                for i, bus in enumerate(self.battery_buses):
                    power_limit = self.battery_power_limit[i]
                    cap = self.battery_capacity[i]
                    min_soc = self.battery_min_soc_frac[i] * cap
                    ch = pyo.Var(bounds=(0, power_limit), initialize=0.0)
                    dch = pyo.Var(bounds=(0, power_limit), initialize=0.0)
                    soc = pyo.Var(bounds=(min_soc, cap), initialize=min_soc)
                    op = pyo.Var(bounds=(-power_limit, power_limit),
                                 initialize=0.0)
                    setattr(self.model, f"charge_T{t}_bus{bus+1}", ch)
                    setattr(self.model, f"discharge_T{t}_bus{bus+1}", dch)
                    setattr(self.model, f"soc_T{t}_bus{bus+1}", soc)
                    setattr(self.model, f"battery_op_T{t}_bus{bus+1}", op)
                    self.CHARGE_dict[(t, bus)] = ch
                    self.DISCHARGE_dict[(t, bus)] = dch
                    self.SOC_dict[(t, bus)] = soc
                    self.BatteryOperation_dict[(t, bus)] = op
                    setattr(self.model,
                            f"battery_link_T{t}_bus{bus}",
                            pyo.Constraint(expr=op == dch - ch))

    # ------------------------------------------------------------------
    # 5. Restrições — inclui bloco UC (limites, rampas, lógica, min up/down)
    # ------------------------------------------------------------------
    def _add_CONS(self):
        T = self.horizon_time
        m = self.model

        # ---- Controle de tensão ----
        if self.NUTE > 0:
            VoltageControlConstraints.add_constraints(
                model=m, T=T, V_PU=self.V_dict, V_ESP=self.V_ESP)

        # ---- Eólico ----
        wind_gen_to_bar = None
        if self.NGWD > 0:
            wind_gen_to_bar = self.wind_bus.tolist()
            WindGeneratorConstraints.add_constraints(
                model=m, T=T, NGER_GWD=self.NGWD,
                PGWIND=self.PGWIND_dict,
                CURTAILMENT=self.CURTAILMENT_dict,
                PGWIND_AVAIL=self.PGWIND_AVAIL)

        # ---- Baterias ----
        battery_list = None
        if self.NBESS > 0:
            battery_list = self.battery_buses.tolist()
            BatteryConstraints.add_constraints(
                model=m, sistema=self.sistema, T=T,
                battery_list=self.battery_buses.tolist(),
                CHARGE=self.CHARGE_dict,
                DISCHARGE=self.DISCHARGE_dict,
                SOC=self.SOC_dict,
                BatteryOperation=self.BatteryOperation_dict,
                soc_inicial_list=self.soc_inicial_list,
                soc_final_list=None,
                daily_reset_to_initial=False)

        # ---- Balanço AC ----
        AC_BalanceConstraints.add_constraints(
            model=m, sistema=self.sistema, HORA=T,
            G=self.G, B=self.B,
            V=self.V_dict, ANG=self.ANG_dict,
            PGER=self.PGER_dict, QGER=self.QGER_dict,
            PLOAD=self.PLOAD, QLOAD=self.QLOAD,
            DEFICIT=self.DEFICIT_dict,
            PGWIND=self.PGWIND_dict if self.NGWD > 0 else None,
            BESS_SOC_op=self.BatteryOperation_dict if self.NBESS > 0 else None,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            wind_gen_to_bar=wind_gen_to_bar,
            battery_list=battery_list)

        # ---- Linhas AC ----
        AC_LineConstraints.add_constraints(
            model=m, sistema=self.sistema, HORA=T,
            V=self.V_dict, ANG=self.ANG_dict,
            line_from=self.line_from, line_to=self.line_to,
            line_r=self.line_r, line_x=self.line_x,
            line_flow_max=self.line_flow_max)

    # ------------------------------------------------------------------
    # 6. Objetivo — custo quadrático + startup
    # ------------------------------------------------------------------
    def _add_FOB(self):
        s = self.sistema
        T = self.horizon_time
        expr = 0.0

        # ---- Térmicos: a·P² + b·P + c·u + startup·V ----
        for t in range(T):
            for g in range(self.NUTE):
                P = self.PGER_dict[(t, g)]
                U = self.U_dict[(t, g)]
                expr += (self.thermal_cost_quad[g] * P * P
                         + self.thermal_cost_lin[g] * P
                         + self.thermal_cost_base[g] * U)
                expr += self.startup_cost[g] * self.V_start_dict[(t, g)]

        # ---- Curtailment ----
        if hasattr(s, 'custo_CURTAILMENT') and self.NGWD > 0:
            custo_curt = s.custo_CURTAILMENT
            for t in range(T):
                for w in range(self.NGWD):
                    c = (float(custo_curt[w])
                         if hasattr(custo_curt, '__getitem__')
                         else float(custo_curt))
                    expr += c * self.CURTAILMENT_dict[(t, w)]

        # ---- Déficit ----
        if hasattr(s, 'custo_DEFICIT'):
            for t in range(T):
                for b in range(self.NBAR):
                    expr += float(s.custo_DEFICIT) * self.DEFICIT_dict[(t, b)]

        # ---- Baterias ----
        if hasattr(s, 'BATTERY_COST_CHARGE') and self.NBESS > 0:
            arr = s.BATTERY_COST_CHARGE
            for t in range(T):
                for i, b in enumerate(self.battery_buses):
                    custo = (float(arr[i]) if hasattr(arr, '__getitem__')
                             and len(arr) > 1 else float(arr))
                    expr += custo * self.CHARGE_dict[(t, b)]
        if hasattr(s, 'BATTERY_COST_DISCHARGE') and self.NBESS > 0:
            arr = s.BATTERY_COST_DISCHARGE
            for t in range(T):
                for i, b in enumerate(self.battery_buses):
                    custo = (float(arr[i]) if hasattr(arr, '__getitem__')
                             and len(arr) > 1 else float(arr))
                    expr += custo * self.DISCHARGE_dict[(t, b)]

        self.model.FOB = pyo.Objective(expr=expr, sense=pyo.minimize)

    # ------------------------------------------------------------------
    # 7. Build
    # ------------------------------------------------------------------
    def _build_AC_OPF_TIME(self, fator_carga=None, fator_vento=None,
                           soc_inicial=0.5, soc_final=None):
        self.model = pyo.ConcreteModel(name="ACOPF_TimeCoupled")
        self._processa_multiplicadores(fator_carga, fator_vento,
                                       soc_inicial, soc_final)
        self._add_VARS()
        self._add_CONS()
        self._add_FOB()
        self._solved = False

    def _build_Cenario(self, fator_carga=None, fator_vento=None,
                       soc_inicial=0.5, soc_final=None):
        self._build_AC_OPF_TIME(fator_carga, fator_vento,
                                soc_inicial, soc_final)

    # ------------------------------------------------------------------
    # 8. Solve
    # ------------------------------------------------------------------
    def solve(self, solver_name='ipopt', tee=True, **kwargs):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")

        # Detecta solver MINLP
        minlp_solvers = {'ipopt', 'bonmin', 'couenne', 'baron'}
        has_binaries = self.NUTE > 0
        if has_binaries and solver_name.lower() not in minlp_solvers:
            print(f"  ⚠ '{solver_name}' não trata MINLP com binárias de UC. "
                  f"Use um de {minlp_solvers} ou relaxe U/V/W para contínuo.")

        opt = pyo.SolverFactory(solver_name)
        try:
            opt.set_executable(
                f'/home/lucasedbraga/anaconda3/envs/otm_venv/bin/{solver_name}')
        except Exception:
            pass

        results = opt.solve(self.model, tee=tee, **kwargs)
        self._solved = (
            results.solver.status == pyo.SolverStatus.ok and
            results.solver.termination_condition in (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.feasible))
        return results

    def solve_timecoupled(self, solver_name='ipopt',
                          fator_carga=None, fator_vento=None,
                          soc_inicial=0.5, soc_final=None,
                          cen_id=None, tee=True):
        self._build_Cenario(fator_carga, fator_vento, soc_inicial, soc_final)
        results = self.solve(solver_name, tee=tee)

        if self.db_handler is not None and cen_id is not None and self._solved:
            resultado_global = self.extract_results()
            for snap in resultado_global.snapshots:
                self.db_handler.save_hourly_result(
                    resultado=snap, sistema=self.sistema,
                    hora=snap.hora, solver_name=solver_name,
                    dia=str(snap.dia + 1), cen_id=cen_id)
        return results

    # ------------------------------------------------------------------
    # 9. Extração 
    # ------------------------------------------------------------------
    def extract_results(self) -> TimeCoupled_OPF_Result:
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")

        s = self.sistema
        T = self.horizon_time
        snapshots = []
        dias_nomes = ["domingo", "segunda", "terça", "quarta",
                      "quinta", "sexta", "sábado"]

        for t in range(T):
            dia = t // self.n_horas
            hora = t % self.n_horas
            dia_semana = ((self.dia_inicial + dia) % 7) + 1
            dia_semana_nome = dias_nomes[dia_semana - 1]

            try:
                PLOAD_vals = self.PLOAD[t, :].tolist()
                QLOAD_vals = (self.QLOAD[t, :].tolist()
                              if self.QLOAD is not None else [])
                PGER_vals = [pyo.value(self.PGER_dict[(t, g)])
                             for g in range(self.NUTE)]
                QGER_vals = [pyo.value(self.QGER_dict[(t, g)])
                             for g in range(self.NUTE)]

                if self.NGWD > 0:
                    PGWIND_disponivel = self.PGWIND_AVAIL[t, :].tolist()
                    PGWIND_vals = [pyo.value(self.PGWIND_dict[(t, w)])
                                   for w in range(self.NGWD)]
                    CURTAILMENT_vals = [pyo.value(self.CURTAILMENT_dict[(t, w)])
                                        for w in range(self.NGWD)]
                else:
                    PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

                DEFICIT_vals = [pyo.value(self.DEFICIT_dict[(t, b)])
                                for b in range(self.NBAR)]

                SOC_init = [0.0] * self.NBAR
                SOC_atual = [0.0] * self.NBAR
                BESS_operation = [0.0] * self.NBAR
                if self.NBESS > 0:
                    for i, bus in enumerate(self.battery_buses):
                        if t == 0:
                            soc_init_val = self.soc_inicial_list[i]
                        else:
                            soc_init_val = pyo.value(self.SOC_dict[(t - 1, bus)])
                        SOC_init[bus] = soc_init_val
                        SOC_atual[bus] = pyo.value(self.SOC_dict[(t, bus)])
                        ch = pyo.value(self.CHARGE_dict[(t, bus)])
                        dch = pyo.value(self.DISCHARGE_dict[(t, bus)])
                        BESS_operation[bus] = dch - ch

                V_vals = [pyo.value(self.V_dict[(t, b)])
                          for b in range(self.NBAR)]
                ANG_vals = [pyo.value(self.ANG_dict[(t, b)])
                            for b in range(self.NBAR)]

                # Fluxos
                P_flow = np.zeros(self.NLIN)
                Q_flow = np.zeros(self.NLIN)
                for e in range(self.NLIN):
                    i, j = self.line_from[e], self.line_to[e]
                    vi, vj = V_vals[i], V_vals[j]
                    th_i, th_j = ANG_vals[i], ANG_vals[j]
                    r, x = self.line_r[e], self.line_x[e]
                    z2 = r * r + x * x
                    if z2 == 0:
                        continue
                    g = r / z2
                    b = -x / z2
                    P_flow[e] = (vi ** 2 * g
                                 - vi * vj * (g * np.cos(th_i - th_j)
                                              + b * np.sin(th_i - th_j)))
                    Q_flow[e] = (-vi ** 2 * b
                                 - vi * vj * (g * np.sin(th_i - th_j)
                                              - b * np.cos(th_i - th_j)))

                total_gen = (np.sum(PGER_vals) + np.sum(PGWIND_vals)
                             + np.sum(BESS_operation) + np.sum(DEFICIT_vals))
                total_load = np.sum(PLOAD_vals)
                PERDAS_TOTAIS = total_gen - total_load

                custo_deficit = getattr(s, 'custo_DEFICIT', 1000.0)
                CUSTO = [d * custo_deficit for d in DEFICIT_vals]

                snap = OPF_SnapshotResult(
                    dia=dia, dia_semana=dia_semana, hora=hora, sucesso=True,
                    PLOAD=PLOAD_vals, QLOAD=QLOAD_vals,
                    PGER=PGER_vals, QGER=QGER_vals,
                    PGWIND_disponivel=PGWIND_disponivel,
                    PGWIND=PGWIND_vals, CURTAILMENT=CURTAILMENT_vals,
                    SOC_init=SOC_init, BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual, DEFICIT=DEFICIT_vals,
                    V=V_vals, ANG=ANG_vals,
                    FLUXO_LIN=P_flow.tolist(), REATIVO_LIN=Q_flow.tolist(),
                    CUSTO=CUSTO, CMO=[0.0],
                    PERDAS_TOTAIS=PERDAS_TOTAIS,
                    dia_semana_nome=dia_semana_nome)

                snapshots.append(snap)
            except Exception as e:
                traceback.print_exc()
                snapshots.append(OPF_SnapshotResult(
                    dia=dia, hora=hora, sucesso=False, mensagem=str(e),
                    dia_semana=dia_semana, dia_semana_nome=dia_semana_nome))

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupled_OPF_Result(
            snapshots=snapshots,
            sucesso_global=sucesso_global,
            mensagem_global="OK" if sucesso_global else "Falhas na extração")


# =============================================================================
# Exemplo de uso
# =============================================================================
if __name__ == "__main__":
    import secrets
    from datetime import datetime

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from UTILS.SystemLoader import SistemaLoader
    from DB.DBhandler_OPF import OPF_DBHandler
    from UTILS.EvaluateFactors import EvaluateFactors

    print("=" * 70)
    print("SIMULAÇÃO ACOPLADA - UC + AC OPF (Pyomo)")
    print("=" * 70)

    json_path = "DATA/input/3barras_QUAD.json"
    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")

    n_dias, n_horas = 1, 24
    T = n_dias * n_horas
    SOC_inicial, SOC_final = 0.5, 0.5

    db_handler = OPF_DBHandler('DATA/output/3barras_QUAD.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')

    modelo = ACOPF_TimeCoupled(
        sistema=sistema, n_horas=n_horas, n_dias=n_dias,
        db_handler=db_handler, dia_inicial=0)

    seed = secrets.randbits(32)
    avaliador = EvaluateFactors(
        sistema=sistema, n_dias=n_dias, n_horas=n_horas,
        carga_incerteza=0.05, vento_variacao=0.9, seed=seed)
    fatores_carga, fatores_vento = avaliador.gerar_tudo()

    print(f"\n  Modo: UC + AC OPF  (custo quadrático + startup + rampas)")
    print(f"  Solver: ipopt (MINLP)\n")

    results = modelo.solve_timecoupled(
        solver_name='ipopt',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cen_id=cen_id,
        tee=True)

    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    for t in range(min(T, 6)):
        snap = resultados.snapshots[t]
        u_vals = getattr(snap, 'U', [None] * sistema.NGER_UTE)
        print(f"\nHora {snap.hora} (dia {snap.dia+1}):")
        print(f"  Carga         : {sum(snap.PLOAD):.3f} pu")
        print(f"  PGER (UTE)    : {sum(snap.PGER):.3f} pu")
        print(f"  U (UC status) : "
              f"{[int(u) if u is not None else '?' for u in u_vals]}")
        print(f"  Déficit       : {sum(snap.DEFICIT):.3f} pu")
        print(f"  Perdas        : {snap.PERDAS_TOTAIS:.3f} pu")
        print(f"  V range       : [{min(snap.V):.3f}, {max(snap.V):.3f}] pu")