#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DC-OPF TIME-COUPLED (multi-período) com custo quadrático e perdas iterativas.

  Objetivo:
      min  Σ_t Σ_g [ a_g·P² + b_g·P + c_g ]
           + Σ_t Σ_b custo_déficit·DEFICIT[t,b]
           + Σ_t Σ_w custo_curtailment·CURTAILMENT[t,w]

  Coeficientes lidos do SistemaLoader:
      a_g = custo_GER_quad
      b_g = custo_GER_linear
      c_g = custo_GER_base

  Nomenclatura padronizada conforme ACOPF_Snapshot_Model / DC_OPF_Snapshot.
"""

import os
import sys
import traceback
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import List, Union, Optional, Tuple, Dict, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_DC.RES.BatteryConstraints import BatteryConstraints
from SOLVER.OPF_DC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_DC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_DC.RES.EletricConstraints import ElectricConstraints
from DB.DBmodel_OPF import TimeCoupled_OPF_Result, OPF_SnapshotResult


class TimeCoupledOPFModel:
    """
    Modelo DC-OPF multi-período acoplado no tempo.
    Todas as grandezas em pu. Custo quadrático por gerador.
    """

    def __init__(self,
                 sistema,
                 n_horas: int = 24,
                 n_dias: int = 1,
                 db_handler=None,
                 considerar_perdas: bool = True,
                 dia_inicial: int = 0):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.horizon_time = n_horas * n_dias
        self.db_handler = db_handler
        self.considerar_perdas = considerar_perdas
        self.dia_inicial = dia_inicial

        self.model = None
        self._solved = False

        # ---- Dicionários de variáveis (t, idx) ----
        self.PGER: Dict[Tuple[int, int], poi.Variable] = {}
        self.PGWIND: Dict[Tuple[int, int], poi.Variable] = {}
        self.CURTAILMENT: Dict[Tuple[int, int], poi.Variable] = {}
        self.DEFICIT: Dict[Tuple[int, int], poi.Variable] = {}
        self.V: Dict[Tuple[int, int], poi.Variable] = {}
        self.ANG: Dict[Tuple[int, int], poi.Variable] = {}
        self.FLUXO_LIN: Dict[Tuple[int, int], poi.Variable] = {}
        self.CHARGE: Dict[Tuple[int, int], poi.Variable] = {}
        self.DISCHARGE: Dict[Tuple[int, int], poi.Variable] = {}
        self.SOC: Dict[Tuple[int, int], poi.Variable] = {}
        self.BatteryOperation: Dict[Tuple[int, int], poi.Variable] = {}

        # ---- Parâmetros (T, ...) em pu ----
        self.PLOAD: Optional[np.ndarray] = None
        self.PGWIND_AVAIL: Optional[np.ndarray] = None

        # ---- Custos (carregados no build) ----
        self.thermal_cost_quad    = np.array([])
        self.thermal_cost_lin     = np.array([])
        self.thermal_cost_base    = np.array([])
        self.startup_cost         = np.array([])

        # ---- Restrições de balanço (recriadas a cada iteração de perdas) ----
        self.balance_constraints: List[Tuple[int, int, poi.Constraint]] = []

        # ---- Baterias ----
        self._battery_list: List[int] = []
        self._battery_index: Dict[int, int] = {}
        self._soc_inicial_list: List[float] = []
        self._soc_final_list: List[float] = []

        # ---- Perdas (T, NBAR) em pu ----
        self._perdas_calculadas: Optional[np.ndarray] = None

    # ==================================================================
    # BUILD
    # ==================================================================
    def build(self,
              fator_carga: Optional[np.ndarray] = None,
              fator_vento: Optional[np.ndarray] = None,
              soc_inicial: Union[float, List[float]] = 0.5,
              soc_final: Optional[Union[float, List[float]]] = None) -> None:

        self._read_costs_from_sistema()   # <<< ESSENCIAL
        self._process_fatores(fator_carga, fator_vento)
        self._process_soc(soc_inicial, soc_final)

        self.model = highs.Model()
        self._add_VARS()
        self._add_CONS()
        self._solved = False

    # ------------------------------------------------------------------
    # Fatores
    # ------------------------------------------------------------------
    def _process_fatores(self, fator_carga, fator_vento):
        s = self.sistema
        T = self.horizon_time

        # ----- Carga -----
        if fator_carga is None:
            fc = np.ones((T, s.NBAR))
        else:
            fc = np.asarray(fator_carga, dtype=float)
            if fc.ndim == 3:                      # (n_dias, n_horas, NBAR)
                fc = fc.reshape((T, s.NBAR))
            elif fc.ndim == 2:
                if fc.shape == (T, s.NBAR):
                    pass
                elif fc.shape == (self.n_dias, self.n_horas):
                    fc = np.repeat(fc.reshape((T, 1)), s.NBAR, axis=1)
                else:
                    raise ValueError(f"fator_carga shape inválido: {fc.shape}")
            elif fc.ndim == 1:
                if fc.size == T:
                    fc = fc[:, np.newaxis] * np.ones((1, s.NBAR))
                else:
                    raise ValueError(f"fator_carga shape inválido: {fc.shape}")
        self.PLOAD = s.PLOAD[np.newaxis, :] * fc     # (T, NBAR)

        # ----- Vento -----
        if s.NGER_GWD == 0:
            self.PGWIND_AVAIL = np.zeros((T, 0))
        else:
            if fator_vento is None:
                fv = np.ones((T, s.NGER_GWD))
            else:
                fv = np.asarray(fator_vento, dtype=float)
                if fv.ndim == 3:
                    fv = fv.reshape((T, s.NGER_GWD))
                elif fv.ndim == 2:
                    if fv.shape == (T, s.NGER_GWD):
                        pass
                    elif fv.shape == (self.n_dias, self.n_horas):
                        fv = np.repeat(fv.reshape((T, 1)), s.NGER_GWD, axis=1)
                    else:
                        raise ValueError(f"fator_vento shape inválido: {fv.shape}")
                elif fv.ndim == 1:
                    if fv.size == T:
                        fv = fv[:, np.newaxis] * np.ones((1, s.NGER_GWD))
                    else:
                        raise ValueError(f"fator_vento shape inválido: {fv.shape}")
            base_wind = np.array(
                getattr(s, 'PGWD_MAX_ORIGINAL',
                        getattr(s, 'PGWIND_disponivel',
                                [1.0] * s.NGER_GWD)), dtype=float)
            self.PGWIND_AVAIL = base_wind[np.newaxis, :] * fv

    # ------------------------------------------------------------------
    # SOC
    # ------------------------------------------------------------------
    def _process_soc(self, soc_inicial, soc_final):
        s = self.sistema
        self._battery_list = list(getattr(s, 'BARRAS_COM_BATERIA', []))
        self._battery_index = {b: i for i, b in enumerate(self._battery_list)}
        nb = len(self._battery_list)
        if nb == 0:
            return

        # SOC inicial
        if isinstance(soc_inicial, (float, int)):
            soc_ini_frac = [soc_inicial] * nb
        else:
            soc_ini_frac = list(soc_inicial)
            if len(soc_ini_frac) != nb:
                raise ValueError(f"soc_inicial deve ter {nb} elementos")
        self._soc_inicial_list = [
            soc_ini_frac[i] * s.BATTERY_CAPACITY[b]
            for i, b in enumerate(self._battery_list)
        ]

        # SOC final (opcional)
        if soc_final is not None:
            if isinstance(soc_final, (float, int)):
                soc_fin_frac = [soc_final] * nb
            else:
                soc_fin_frac = list(soc_final)
                if len(soc_fin_frac) != nb:
                    raise ValueError(f"soc_final deve ter {nb} elementos")
            self._soc_final_list = [
                soc_fin_frac[i] * s.BATTERY_CAPACITY[b]
                for i, b in enumerate(self._battery_list)
            ]
        else:
            self._soc_final_list = []

    def _read_costs_from_sistema(self):
        """Lê coeficientes de custo do SistemaLoader com fallbacks."""
        s = self.sistema
        n = s.NGER_UTE
        self.thermal_cost_quad = np.array(
            getattr(s, 'custo_GER_quad',
                    getattr(s, 'custo_quadratico_a', [0.0] * n)),
            dtype=float)
        self.thermal_cost_lin = np.array(
            getattr(s, 'custo_GER_linear',
                    getattr(s, 'custo_GER', [50.0] * n)),
            dtype=float)
        self.thermal_cost_base = np.array(
            getattr(s, 'custo_GER_base',
                    getattr(s, 'custo_base', [0.0] * n)),
            dtype=float)
        self.startup_cost = np.array(
            getattr(s, 'custo_GER_startup',
                    getattr(s, 'custo_startup', [0.0] * n)),
            dtype=float)
        
    # ==================================================================
    # VARIÁVEIS
    # ==================================================================
    def _add_VARS(self):

        def _create_VARx_V():
            for t in range(self.horizon_time):
                for b in range(self.sistema.NBAR):
                    self.V[t, b] = self.model.add_variable(
                        lb=0.95, ub=1.05, name=f"V_{t}_{b+1}")

        def _create_VARx_ANG():
            s = self.sistema
            for t in range(self.horizon_time):
                for b in range(s.NBAR):
                    self.ANG[t, b] = self.model.add_variable(
                        lb=-np.pi, ub=np.pi, name=f"ANG_{t}_{b+1}")
                self.model.add_linear_constraint(
                    self.ANG[t, s.slack_idx] == 0.0,
                    name=f"fix_ANG_slack_{t}")

        def _create_VARx_PGER():
            s = self.sistema
            for t in range(self.horizon_time):
                for g in range(s.NGER_UTE):
                    self.PGER[t, g] = self.model.add_variable(
                        lb=float(s.PGER_MIN_UTE[g]),
                        ub=float(s.PGER_MAX_UTE[g]),
                        name=f"PGER_{t}_{g+1}")

        def _create_VARx_GWD():
            s = self.sistema
            if s.NGER_GWD == 0:
                return
            for t in range(self.horizon_time):
                for w in range(s.NGER_GWD):
                    avail = float(self.PGWIND_AVAIL[t, w])
                    self.PGWIND[t, w] = self.model.add_variable(
                        lb=0.0, ub=avail, name=f"PGWD_{t}_{w+1}")
                    self.CURTAILMENT[t, w] = self.model.add_variable(
                        lb=0.0, ub=avail, name=f"CURTAIL_{t}_{w+1}")

        def _create_VARx_DEF():
            s = self.sistema
            for t in range(self.horizon_time):
                for b in range(s.NBAR):
                    self.DEFICIT[t, b] = self.model.add_variable(
                        lb=0.0, ub=1e6, name=f"DEFICIT_{t}_{b+1}")

        def _create_VARx_FLUXO():
            s = self.sistema
            for t in range(self.horizon_time):
                for e in range(s.NLIN):
                    self.FLUXO_LIN[t, e] = self.model.add_variable(
                        lb=-float(s.FLIM[e]),
                        ub=float(s.FLIM[e]),
                        name=f"FLOW_{t}_{e+1}")

        def _create_VARx_BESS():
            if not self._battery_list:
                return
            s = self.sistema
            for t in range(self.horizon_time):
                for i, b in enumerate(self._battery_list):
                    cap       = float(s.BATTERY_CAPACITY[b])
                    min_soc   = float(s.BATTERY_MIN_SOC[b]) * cap
                    p_in      = float(s.BATTERY_POWER_LIMIT[b])
                    p_out_raw = getattr(s, 'BATTERY_POWER_OUT', None)
                    if p_out_raw is None:
                        p_out = p_in
                    elif isinstance(p_out_raw, (list, tuple, np.ndarray)):
                        p_out = float(p_out_raw[i])
                    elif isinstance(p_out_raw, dict):
                        p_out = float(p_out_raw.get(b, p_in))
                    else:
                        p_out = float(p_out_raw)

                    self.CHARGE[t, b] = self.model.add_variable(
                        lb=0.0, ub=p_in, name=f"CHARGE_{t}_{b+1}")
                    self.DISCHARGE[t, b] = self.model.add_variable(
                        lb=0.0, ub=p_out, name=f"DISCHARGE_{t}_{b+1}")
                    self.SOC[t, b] = self.model.add_variable(
                        lb=min_soc, ub=cap, name=f"SOC_{t}_{b+1}")
                    self.BatteryOperation[t, b] = self.model.add_variable(
                        lb=-p_out, ub=p_out, name=f"BatteryOp_{t}_{b+1}")

                    self.model.add_linear_constraint(
                        self.BatteryOperation[t, b]
                        == self.DISCHARGE[t, b] - self.CHARGE[t, b],
                        name=f"battery_link_{t}_{b+1}")

        _create_VARx_V()
        _create_VARx_ANG()
        _create_VARx_PGER()
        _create_VARx_GWD()
        _create_VARx_DEF()
        _create_VARx_FLUXO()
        _create_VARx_BESS()

    # ==================================================================
    # RESTRIÇÕES
    # ==================================================================
    def _add_CONS(self):
        s = self.sistema
        T = self.horizon_time

        # ---- Térmicos ----
        if s.NGER_UTE > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_CONV=s.NGER_UTE,
                PGER=self.PGER,
                PGER_MIN_UTE=s.PGER_MIN_UTE,
                PGER_MAX_UTE=s.PGER_MAX_UTE,
                PGER_INICIAL_UTE=s.PGER_INICIAL_UTE,
                RAMP_UP=s.RAMP_UP,
                RAMP_DOWN=s.RAMP_DOWN,
                SB=s.SB
            )

        # ---- Baterias ----
        if self._battery_list:
            BatteryConstraints.add_constraints(
                model=self.model,
                sistema=s,
                T=T,
                battery_list=self._battery_list,
                CHARGE=self.CHARGE,
                DISCHARGE=self.DISCHARGE,
                SOC=self.SOC,
                BatteryOperation=self.BatteryOperation,
                soc_inicial_list=self._soc_inicial_list,
                soc_final_list=(self._soc_final_list
                                if self._soc_final_list else None),
                daily_reset_to_initial=True
            )

        # ---- Eólicos ----
        if s.NGER_GWD > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_GWD=s.NGER_GWD,
                PGWIND=self.PGWIND,
                CURTAILMENT=self.CURTAILMENT,
                PGWIND_AVAIL=self.PGWIND_AVAIL
            )

        # ---- Elétricos (balanço + fluxo DC) ----
        wind_gen_to_bar = (self.wind_bus.tolist() if s.NGER_GWD > 0 else None)

        self.balance_constraints = ElectricConstraints.add_constraints(
            model=self.model,
            sistema=s,
            T=T,
            ANG=self.ANG,
            FLUXO_LIN=self.FLUXO_LIN,
            DEFICIT=self.DEFICIT,
            PLOAD=self.PLOAD,
            PGER=self.PGER,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            PGWIND=self.PGWIND if s.NGER_GWD > 0 else None,
            wind_gen_to_bar=wind_gen_to_bar,
            CHARGE=self.CHARGE if self._battery_list else None,
            DISCHARGE=self.DISCHARGE if self._battery_list else None,
            battery_list=self._battery_list,
            PERDAS_BARRA=self._perdas_calculadas if self.considerar_perdas else None,
            considerar_perdas=self.considerar_perdas
        )

    # Alias de compatibilidade
    _add_all_constraints = _add_CONS

    # ==================================================================
    # OBJETIVO — quadrático + base + déficit + curtailment
    # ==================================================================
    def build_objective(self, cost_function: Optional[Callable] = None):
        if cost_function is not None:
            expr = cost_function(self)
            self.model.set_objective(expr, poi.ObjectiveSense.Minimize)
            return

        s = self.sistema
        T = self.horizon_time
        expr = 0.0

        # ---- Térmicos: a·P² + b·P + c (loop em t E g) ----
        for t in range(T):
            for g in range(s.NGER_UTE):
                P = self.PGER[t, g]
                expr += self.thermal_cost_quad[g] * P * P
                expr += self.thermal_cost_lin[g] * P
                expr += self.thermal_cost_base[g]      # <<< termo constante

        # ---- Déficit ----
        custo_deficit = float(getattr(s, 'custo_DEFICIT', 1000.0))
        for t in range(T):
            for b in range(s.NBAR):
                expr += custo_deficit * self.DEFICIT[t, b]

        # ---- Curtailment ----
        if s.NGER_GWD > 0:
            custo_curt_arr = getattr(s, 'custo_curtailment', None)
            for t in range(T):
                for w in range(s.NGER_GWD):
                    if custo_curt_arr is None:
                        c = 0.0
                    elif hasattr(custo_curt_arr, '__getitem__'):
                        c = float(custo_curt_arr[w])
                    else:
                        c = float(custo_curt_arr)
                    expr += c * self.CURTAILMENT[t, w]

        # ---- Baterias (custo de carga/descarga, se existir) ----
        if self._battery_list:
            arr_ch  = getattr(s, 'BATTERY_COST_CHARGE', None)
            arr_dch = getattr(s, 'BATTERY_COST_DISCHARGE', None)
            for t in range(T):
                for i, b in enumerate(self._battery_list):
                    if arr_ch is not None:
                        c = (float(arr_ch[i])
                             if hasattr(arr_ch, '__getitem__') and len(arr_ch) > 1
                             else float(arr_ch))
                        expr += c * self.CHARGE[t, b]
                    if arr_dch is not None:
                        c = (float(arr_dch[i])
                             if hasattr(arr_dch, '__getitem__') and len(arr_dch) > 1
                             else float(arr_dch))
                        expr += c * self.DISCHARGE[t, b]

        self.model.set_objective(expr, poi.ObjectiveSense.Minimize)

    # Alias de compatibilidade
    _add_FOB = build_objective

    # Alias de compatibilidade
    _add_FOB = build_objective

    # ==================================================================
    # PERDAS ITERATIVAS
    # ==================================================================
    def calculate_losses(self) -> np.ndarray:
        s = self.sistema
        T = self.horizon_time
        perdas_barra = np.zeros((T, s.NBAR))
        for t in range(T):
            for e in range(s.NLIN):
                i = s.line_fr[e]
                j = s.line_to[e]
                flow = self.model.get_value(self.FLUXO_LIN[t, e])
                r = s.r_line[e]
                loss = r * (flow ** 2)
                perdas_barra[t, i] += loss / 2
                perdas_barra[t, j] += loss / 2
        self._perdas_calculadas = perdas_barra
        return perdas_barra

    def update_losses(self, perdas_barra: np.ndarray) -> None:
        self._perdas_calculadas = perdas_barra

    def solve_iterative(self, solver_name: str = 'highs', tol: float = 1e-4,
                        max_iter: int = 50, write_lp: bool = False,
                        **solver_args):
        if not self.considerar_perdas:
            return self.solve(solver_name, write_lp=write_lp, **solver_args)

        self._perdas_calculadas = np.zeros((self.horizon_time, self.sistema.NBAR))
        raw = self.solve(solver_name, write_lp=write_lp, **solver_args)
        if not self._solved or self.model.get_model_attribute(
                poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
            print("Primeira iteração: solução não ótima.")
            return raw

        perdas_prev = self._perdas_calculadas.copy()
        for it in range(1, max_iter):
            perdas_atuais = self.calculate_losses()
            self.update_losses(perdas_atuais)

            # Recria as restrições de balanço com as perdas atualizadas
            for _, _, constr in self.balance_constraints:
                self.model.delete_constraint(constr)

            s = self.sistema
            T = self.horizon_time
            wind_gen_to_bar = self.wind_bus.tolist() if s.NGER_GWD > 0 else None
            self.balance_constraints = ElectricConstraints.add_constraints(
                model=self.model,
                sistema=s,
                T=T,
                ANG=self.ANG,
                FLUXO_LIN=self.FLUXO_LIN,
                DEFICIT=self.DEFICIT,
                PLOAD=self.PLOAD,
                PGER=self.PGER,
                conv_gen_to_bar=self.thermal_bus.tolist(),
                PGWIND=self.PGWIND if s.NGER_GWD > 0 else None,
                wind_gen_to_bar=wind_gen_to_bar,
                CHARGE=self.CHARGE if self._battery_list else None,
                DISCHARGE=self.DISCHARGE if self._battery_list else None,
                battery_list=self._battery_list,
                PERDAS_BARRA=perdas_atuais,
                considerar_perdas=self.considerar_perdas
            )

            raw = self.solve(solver_name, write_lp=False, **solver_args)
            if not self._solved or self.model.get_model_attribute(
                    poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
                print(f"Iteração {it+1}: solução não ótima.")
                break

            diff = np.max(np.abs(perdas_atuais - perdas_prev))
            print(f"Iteração {it+1}: diff_perdas = {diff:.6f}")
            if diff < tol:
                print(f"Convergência alcançada na iteração {it+1}.")
                break
            perdas_prev = perdas_atuais.copy()
        else:
            print("Número máximo de iterações atingido sem convergência.")

        return raw

    # ==================================================================
    # SOLVE
    # ==================================================================
    def solve(self, solver_name: str = 'highs', write_lp: bool = False,
              **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído. Chame build() primeiro.")
        if write_lp:
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_filename = caller_frame.f_code.co_filename
            base = os.path.splitext(os.path.basename(caller_filename))[0]
            lp_filename = f"DATA/output_CUR_Oficial/{base}_timecoupled.lp"
            os.makedirs(os.path.dirname(lp_filename), exist_ok=True)
            self.model.write(lp_filename)
            print(f"Modelo escrito em {lp_filename}")
        self.model.optimize()
        self._solved = True
        return self.model

    # ==================================================================
    # EXTRAÇÃO
    # ==================================================================
    def extract_results(self) -> TimeCoupled_OPF_Result:
        if not self._solved or self.model is None:
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
                # ---- Demanda ----
                PLOAD_vals = self.PLOAD[t, :].tolist()
                QLOAD_vals = [0.0] * s.NBAR          # DC: reativo nulo

                # ---- Térmicos ----
                PGER_vals = [self.model.get_value(self.PGER[t, g])
                             for g in range(s.NGER_UTE)]
                QGER_vals = [0.0] * s.NGER_UTE       # DC: reativo nulo

                # ---- Eólicos ----
                if s.NGER_GWD > 0:
                    PGWIND_disponivel = self.PGWIND_AVAIL[t, :].tolist()
                    PGWIND_vals = [self.model.get_value(self.PGWIND[t, w])
                                   for w in range(s.NGER_GWD)]
                    CURTAILMENT_vals = [self.model.get_value(self.CURTAILMENT[t, w])
                                        for w in range(s.NGER_GWD)]
                else:
                    PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

                # ---- Déficit ----
                DEFICIT_vals = [self.model.get_value(self.DEFICIT[t, b])
                                for b in range(s.NBAR)]

                # ---- Baterias ----
                SOC_init = [0.0] * s.NBAR
                SOC_atual = [0.0] * s.NBAR
                BESS_operation = [0.0] * s.NBAR
                if self._battery_list:
                    for i, b in enumerate(self._battery_list):
                        if t == 0:
                            SOC_init[b] = self._soc_inicial_list[i]
                        else:
                            SOC_init[b] = self.model.get_value(self.SOC[t - 1, b])
                        SOC_atual[b] = self.model.get_value(self.SOC[t, b])
                        ch  = self.model.get_value(self.CHARGE[t, b])
                        dch = self.model.get_value(self.DISCHARGE[t, b])
                        BESS_operation[b] = dch - ch

                # ---- Rede ----
                V   = [self.model.get_value(self.V[t, b]) for b in range(s.NBAR)]
                ANG = [self.model.get_value(self.ANG[t, b]) for b in range(s.NBAR)]
                FLUXO_LIN = [self.model.get_value(self.FLUXO_LIN[t, e])
                             for e in range(s.NLIN)]

                # ---- Custo do déficit ----
                custo_deficit_pu = getattr(s, 'custo_DEFICIT', 1000.0)
                CUSTO = [d * custo_deficit_pu for d in DEFICIT_vals]
                CMO = [0.0]

                # ---- Perdas totais ----
                if self.considerar_perdas and self._perdas_calculadas is not None:
                    PERDAS_TOTAIS = float(np.sum(self._perdas_calculadas[t, :]))
                else:
                    PERDAS_TOTAIS = 0.0

                snapshots.append(OPF_SnapshotResult(
                    dia=dia,
                    dia_semana=dia_semana,
                    hora=hora,
                    sucesso=True,
                    PLOAD=PLOAD_vals,
                    QLOAD=QLOAD_vals,
                    PGER=PGER_vals,
                    QGER=QGER_vals,
                    PGWIND_disponivel=PGWIND_disponivel,
                    PGWIND=PGWIND_vals,
                    CURTAILMENT=CURTAILMENT_vals,
                    SOC_init=SOC_init,
                    BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual,
                    DEFICIT=DEFICIT_vals,
                    V=V,
                    ANG=ANG,
                    FLUXO_LIN=FLUXO_LIN,
                    CUSTO=CUSTO,
                    CMO=CMO,
                    PERDAS_TOTAIS=PERDAS_TOTAIS,
                    dia_semana_nome=dia_semana_nome
                ))

            except Exception as e:
                print(f"Erro ao extrair snapshot t={t}: {e}")
                traceback.print_exc()
                snapshots.append(OPF_SnapshotResult(
                    dia=dia,
                    hora=hora,
                    sucesso=False,
                    mensagem=str(e),
                    dia_semana=dia_semana,
                    dia_semana_nome=dia_semana_nome,
                    PLOAD=[], QLOAD=[], PGER=[], QGER=[],
                    PGWIND_disponivel=[], PGWIND=[], CURTAILMENT=[],
                    SOC_init=[], BESS_operation=[], SOC_atual=[],
                    DEFICIT=[], V=[], ANG=[], FLUXO_LIN=[],
                    CUSTO=[], CMO=[], PERDAS_TOTAIS=0.0
                ))

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupled_OPF_Result(
            snapshots=snapshots,
            sucesso_global=sucesso_global,
            mensagem_global="OK" if sucesso_global else "Falhas na extração"
        )

    # ==================================================================
    # CONVENIÊNCIA
    # ==================================================================
    def solve_multiday(self,
                       solver_name: str = 'highs',
                       fator_carga: Optional[np.ndarray] = None,
                       fator_vento: Optional[np.ndarray] = None,
                       soc_inicial: Union[float, List[float]] = 0.5,
                       soc_final: Optional[Union[float, List[float]]] = None,
                       cost_function: Optional[Callable] = None,
                       cen_id: Optional[str] = None,
                       tol: float = 1e-4,
                       max_iter: int = 50,
                       write_lp: bool = False):
        self.build(fator_carga, fator_vento, soc_inicial, soc_final)
        self.build_objective(cost_function)

        if self.considerar_perdas:
            raw = self.solve_iterative(solver_name, tol=tol,
                                       max_iter=max_iter, write_lp=write_lp)
        else:
            raw = self.solve(solver_name, write_lp=write_lp)

        if self.db_handler is not None and cen_id is not None:
            resultados = self.extract_results()
            for snap in resultados.snapshots:
                dia_str = f"{snap.dia+1}"
                self.db_handler.save_hourly_result(
                    resultado=snap,
                    sistema=self.sistema,
                    hora=snap.hora,
                    solver_name=solver_name,
                    dia=dia_str,
                    cen_id=cen_id
                )

        return raw

    # Alias internos usados no _add_CONS
    @property
    def thermal_bus(self):
        return np.array(self.sistema.BAR_PGER_UTE, dtype=int)

    @property
    def wind_bus(self):
        return np.array(
            getattr(self.sistema, 'bus_wind',
                    getattr(self.sistema, 'BARPG_EOL', [])),
            dtype=int)


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
    print("SIMULAÇÃO TIME-COUPLED - DC OPF (custo quadrático)")
    print("=" * 70)

    json_path = "DATA/input/3barras_QUAD.json"
    if not os.path.exists(json_path):
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        sys.exit(1)

    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")
    print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
    print(f"   ✓ Barras: {sistema.NBAR}")
    print(f"   ✓ Linhas: {sistema.NLIN}")
    print(f"   ✓ Geradores convencionais: {sistema.NGER_UTE}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_GWD}")
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    n_dias, n_horas = 1, 24
    T = n_dias * n_horas
    SOC_inicial, SOC_final = 0.5, 0.5

    db_path = 'DATA/output_CUR_Oficial/resultados_timecoupled.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_handler = OPF_DBHandler(db_path)
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"   ✓ Cenário ID: {cen_id}")

    modelo = TimeCoupledOPFModel(
        sistema=sistema,
        n_horas=n_horas, n_dias=n_dias,
        db_handler=db_handler,
        considerar_perdas=True,
        dia_inicial=0
    )

    seed = secrets.randbits(32)
    avaliador = EvaluateFactors(
        sistema=sistema, n_dias=n_dias, n_horas=n_horas,
        carga_incerteza=0.05, vento_variacao=0.9, seed=seed)
    fatores_carga, fatores_vento = avaliador.gerar_tudo()

    print(f"\n4. Resolvendo time-coupled DC OPF (T={T})...")
    raw = modelo.solve_multiday(
        solver_name='highs',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cost_function=None,
        cen_id=cen_id,
        tol=1e-4, max_iter=5,
        write_lp=False
    )

    status = modelo.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    print(f"\nStatus da solução: {status}")

    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    custo_total = sum(sum(snap.CUSTO) for snap in resultados.snapshots
                      if snap.sucesso)
    print(f"Custo total (déficit): {custo_total:.2f} $")

    if modelo._battery_list:
        prim = modelo._battery_list[0]
        soc_final_val = modelo.model.get_value(modelo.SOC[T-1, prim])
        print(f"SOC final bateria {prim+1}: {soc_final_val:.3f} pu")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)