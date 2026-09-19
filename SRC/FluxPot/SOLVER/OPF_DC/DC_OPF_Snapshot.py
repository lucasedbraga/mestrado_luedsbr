#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DC-OPF SNAPSHOT (1 período) com custo quadrático — HiGHS.

Objetivo:
    min  Σ_g [ a_g·P² + b_g·P + c_g ] + custo_déficit·DEFICIT
         + custo_curtailment·CURTAILMENT

Coeficientes lidos do SistemaLoader:
    a_g = custo_GER_quad
    b_g = custo_GER_linear  (fallback: custo_GER)
    c_g = custo_GER_base

Nomenclatura padronizada conforme ACOPF_Snapshot_Model.
"""

import os
import sys
import traceback
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import List, Union, Optional, Dict, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_DC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_DC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_DC.RES.EletricConstraints import ElectricConstraints
from SOLVER.OPF_DC.RES.BatteryConstraints import BatteryConstraints
from DB.DBmodel_OPF import OPF_SnapshotResult


class DCOPFSnapshot:
    """
    Modelo DC-OPF para um único instante (snapshot) usando pyoptinterface.
    Padrão de nomenclatura idêntico ao ACOPF_Snapshot_Model.
    """

    def __init__(self, sistema, db_handler=None, considerar_perdas: bool = True):
        self.sistema = sistema
        self.db_handler = db_handler
        self.considerar_perdas = considerar_perdas

        self.model = None
        self._solved = False

        # --- Arrays do sistema ---
        self._build_SistemaEletrico_arrays()

        # --- Dicionários de variáveis ---
        self.var_lists: Dict[str, List] = {}
        self.var_indices: Dict[str, Dict] = {}

        self.PGER_dict: Dict[tuple, object] = {}
        self.PGWIND_dict: Dict[tuple, object] = {}
        self.CURTAILMENT_dict: Dict[tuple, object] = {}
        self.DEFICIT_dict: Dict[tuple, object] = {}
        self.V_dict: Dict[tuple, object] = {}
        self.ANG_dict: Dict[tuple, object] = {}
        self.FLOW_dict: Dict[tuple, object] = {}
        self.CHARGE_dict: Dict[tuple, object] = {}
        self.DISCHARGE_dict: Dict[tuple, object] = {}
        self.BatteryOperation_dict: Dict[tuple, object] = {}
        self.SOC_dict: Dict[tuple, object] = {}

        # Parâmetros atualizáveis (pu)
        self.PLOAD: Optional[np.ndarray] = None
        self.PGWIND_AVAIL: Optional[np.ndarray] = None
        self._losses: Optional[np.ndarray] = None

        # SOC inicial das baterias (parâmetro)
        self.soc_inicial_list: List[float] = []

        # Restrições de balanço (para iterações de perdas)
        self.balance_constraints = []

    # ------------------------------------------------------------------
    # 1. Arrays do sistema (com custos quadráticos)
    # ------------------------------------------------------------------
    def _build_SistemaEletrico_arrays(self):
        s = self.sistema
        self.NBAR  = s.NBAR
        self.NUTE  = s.NGER_UTE
        self.NGWD  = s.NGER_GWD
        self.NLIN  = s.NLIN
        self.NBESS = len(getattr(s, 'BARRAS_COM_BATERIA', []))

        self.thermal_bus = np.array(s.BAR_PGER_UTE, dtype=int)
        self.wind_bus = np.array(
            getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [])), dtype=int)

        self.line_from = np.array(s.line_fr, dtype=int)
        self.line_to = np.array(s.line_to, dtype=int)
        self.line_x = np.array(s.x_line, dtype=float)
        self.line_r = (np.array(s.r_line, dtype=float)
                       if hasattr(s, 'r_line')
                       else np.zeros(self.NLIN))
        self.line_flow_max = np.array(s.FLIM, dtype=float)

        self.thermal_pmin = np.array(s.PGER_MIN_UTE, dtype=float)
        self.thermal_pmax = np.array(s.PGER_MAX_UTE, dtype=float)

        # ---------- Custos quadráticos ----------
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
                    getattr(s, 'custo_startup', [0.0] * self.NUTE)),
            dtype=float)
        # alias retrocompatível
        self.thermal_cost = self.thermal_cost_lin.copy()

        # ---------- Baterias ----------
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

    # ------------------------------------------------------------------
    # 2. Fatores (carga, vento, SOC)
    # ------------------------------------------------------------------
    def _processa_multiplicadores(self, fator_carga, fator_vento,
                                  fator_soc_init):

        def processa_fator_carga(fator_carga):
            if fator_carga is None:
                fator_carga = 1.0
            if np.isscalar(fator_carga):
                fator_carga_array = np.ones(self.NBAR) * fator_carga
            else:
                fator_carga_array = np.asarray(fator_carga)
                assert fator_carga_array.shape == (self.NBAR,)
            self.PLOAD = self.base_Pload * fator_carga_array

        def processa_fator_vento(fator_vento):
            if self.NGWD > 0:
                if fator_vento is None:
                    fator_vento = 1.0
                if np.isscalar(fator_vento):
                    fator_vento_array = np.ones(self.NGWD) * fator_vento
                else:
                    fator_vento_array = np.asarray(fator_vento)
                    assert fator_vento_array.shape == (self.NGWD,)
                self.PGWIND_AVAIL = (np.array(self.sistema.PGWIND_disponivel)
                                     * fator_vento_array)
            else:
                self.PGWIND_AVAIL = np.array([])

        def processa_soc_init(fator_soc_init):
            if self.NBESS == 0:
                return
            self.soc_inicial_list = []
            for i, bus in enumerate(self.battery_buses):
                if isinstance(fator_soc_init, dict):
                    frac = fator_soc_init.get(bus, 0.5)
                elif fator_soc_init is None:
                    frac = 0.5
                else:
                    frac = float(fator_soc_init)
                self.soc_inicial_list.append(frac * self.battery_capacity[i])

        processa_fator_carga(fator_carga)
        processa_fator_vento(fator_vento)
        processa_soc_init(fator_soc_init)

    # ------------------------------------------------------------------
    # 3. Variáveis
    # ------------------------------------------------------------------
    def _add_VARS(self):

        def _create_VARx_V():
            self.var_lists['v_pu'] = []
            for b in range(self.NBAR):
                var = self.model.add_variable(
                    lb=0.95, ub=1.05, name=f"V_pu_BAR_{b+1}")
                self.var_lists['v_pu'].append(var)
                self.V_dict[(0, b)] = var
            self.var_indices['v_pu'] = {b: v for b, v in enumerate(self.var_lists['v_pu'])}

        def _create_VARx_ANG():
            self.var_lists['ang_pu'] = []
            for b in range(self.NBAR):
                var = self.model.add_variable(
                    lb=-np.pi, ub=np.pi, name=f"ANG_pu_BAR_{b+1}")
                self.var_lists['ang_pu'].append(var)
                self.ANG_dict[(0, b)] = var
            self.model.add_linear_constraint(
                self.var_lists['ang_pu'][self.slack_bus] == 0.0,
                name="fix_slack_angle")
            self.var_indices['ang_pu'] = {b: v for b, v in enumerate(self.var_lists['ang_pu'])}

        def _create_VARx_PGER():
            self.var_lists['PGER_UTE'] = []
            for g in range(self.NUTE):
                p_var = self.model.add_variable(
                    lb=self.thermal_pmin[g],
                    ub=self.thermal_pmax[g],
                    name=f"PGER_UTE_{g+1}")
                self.var_lists['PGER_UTE'].append(p_var)
                self.PGER_dict[(0, g)] = p_var
            self.var_indices['PGER_UTE'] = {g: v for g, v in enumerate(self.var_lists['PGER_UTE'])}

        def _create_VARx_DEF():
            self.var_lists['deficit'] = []
            for b in range(self.NBAR):
                var = self.model.add_variable(
                    lb=0, ub=1e6, name=f"DEFICT_BAR_{b+1}")
                self.var_lists['deficit'].append(var)
                self.DEFICIT_dict[(0, b)] = var
            self.var_indices['deficit'] = {b: v for b, v in enumerate(self.var_lists['deficit'])}

        def _create_VARx_GWD():
            if self.NGWD == 0:
                return
            self.var_lists['p_wind'] = []
            self.var_lists['curtailment'] = []
            for w in range(self.NGWD):
                avail = self.PGWIND_AVAIL[w]
                p_wind = self.model.add_variable(
                    lb=0, ub=avail, name=f"PGWD_{w}")
                curtail = self.model.add_variable(
                    lb=0, ub=avail, name=f"CURTAILMENT_{w}")
                self.var_lists['p_wind'].append(p_wind)
                self.var_lists['curtailment'].append(curtail)
                self.PGWIND_dict[(0, w)] = p_wind
                self.CURTAILMENT_dict[(0, w)] = curtail
            self.var_indices['p_wind']      = {w: v for w, v in enumerate(self.var_lists['p_wind'])}
            self.var_indices['curtailment'] = {w: v for w, v in enumerate(self.var_lists['curtailment'])}

        def _create_VARx_FLUXO():
            self.var_lists['flow'] = []
            for e in range(self.NLIN):
                var = self.model.add_variable(
                    lb=-self.line_flow_max[e],
                    ub=self.line_flow_max[e],
                    name=f"FLOW_{e}")
                self.var_lists['flow'].append(var)
                self.FLOW_dict[(0, e)] = var
            self.var_indices['flow'] = {e: v for e, v in enumerate(self.var_lists['flow'])}

        def _create_VARx_BESS():
            if self.NBESS == 0:
                return
            self.var_lists['charge'] = []
            self.var_lists['discharge'] = []
            self.var_lists['soc'] = []
            self.var_lists['battery_op'] = []
            for i, bus in enumerate(self.battery_buses):
                power_limit = self.battery_power_limit[i]
                cap = self.battery_capacity[i]
                min_soc = self.battery_min_soc_frac[i] * cap

                ch = self.model.add_variable(
                    lb=0, ub=power_limit, name=f"charge_{bus+1}")
                dch = self.model.add_variable(
                    lb=0, ub=power_limit, name=f"discharge_{bus+1}")
                soc = self.model.add_variable(
                    lb=min_soc, ub=cap, name=f"soc_{bus+1}")
                op = self.model.add_variable(
                    lb=-power_limit, ub=power_limit, name=f"battery_op_{bus+1}")

                self.var_lists['charge'].append(ch)
                self.var_lists['discharge'].append(dch)
                self.var_lists['soc'].append(soc)
                self.var_lists['battery_op'].append(op)

                self.CHARGE_dict[(0, bus)] = ch
                self.DISCHARGE_dict[(0, bus)] = dch
                self.SOC_dict[(0, bus)] = soc
                self.BatteryOperation_dict[(0, bus)] = op

                self.model.add_linear_constraint(
                    op == dch - ch, name=f"battery_link_{bus}")

            self.var_indices['charge']     = {b: v for b, v in zip(self.battery_buses, self.var_lists['charge'])}
            self.var_indices['discharge']  = {b: v for b, v in zip(self.battery_buses, self.var_lists['discharge'])}
            self.var_indices['soc']        = {b: v for b, v in zip(self.battery_buses, self.var_lists['soc'])}
            self.var_indices['battery_op'] = {b: v for b, v in zip(self.battery_buses, self.var_lists['battery_op'])}

        _create_VARx_V()
        _create_VARx_ANG()
        _create_VARx_PGER()
        _create_VARx_DEF()
        _create_VARx_GWD()
        _create_VARx_FLUXO()
        _create_VARx_BESS()

    # ------------------------------------------------------------------
    # 4. Restrições
    # ------------------------------------------------------------------
    def _add_CONS(self):

        wind_gen_to_bar = None
        battery_list = None

        if self.NUTE > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model,
                T=1,
                NGER_CONV=self.NUTE,
                PGER=self.PGER_dict,
                PGER_MIN_UTE=self.thermal_pmin,
                PGER_MAX_UTE=self.thermal_pmax,
                PGER_INICIAL_UTE=self.sistema.PGER_INICIAL_UTE,
                RAMP_UP=self.sistema.RAMP_UP,
                RAMP_DOWN=self.sistema.RAMP_DOWN,
                SB=self.sistema.SB
            )

        if self.NGWD > 0:
            wind_gen_to_bar = self.wind_bus.tolist()
            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=1,
                NGER_GWD=self.NGWD,
                PGWIND=self.PGWIND_dict,
                CURTAILMENT=self.CURTAILMENT_dict,
                PGWIND_AVAIL=self.PGWIND_AVAIL.reshape(1, -1)
            )

        if self.NBESS > 0:
            battery_list = self.battery_buses.tolist()
            BatteryConstraints.add_constraints(
                model=self.model,
                sistema=self.sistema,
                T=1,
                battery_list=self.battery_buses.tolist(),
                battery_index=None,
                CHARGE=self.CHARGE_dict,
                DISCHARGE=self.DISCHARGE_dict,
                SOC=self.SOC_dict,
                BatteryOperation=self.BatteryOperation_dict,
                soc_inicial_list=self.soc_inicial_list,
                soc_final_list=None
            )

        PLOAD_2d = self.PLOAD.reshape(1, -1)
        losses_2d = None
        if self.considerar_perdas and self._losses is not None:
            losses_2d = self._losses.reshape(1, -1)

        self.balance_constraints = ElectricConstraints.add_constraints(
            model=self.model,
            sistema=self.sistema,
            T=1,
            ANG=self.ANG_dict,
            FLUXO_LIN=self.FLOW_dict,
            DEFICIT=self.DEFICIT_dict,
            PLOAD=PLOAD_2d,
            PGER=self.PGER_dict,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            PGWIND=self.PGWIND_dict if self.NGWD > 0 else None,
            wind_gen_to_bar=wind_gen_to_bar,
            CHARGE=self.CHARGE_dict if self.NBESS > 0 else None,
            DISCHARGE=self.DISCHARGE_dict if self.NBESS > 0 else None,
            battery_list=battery_list,
            PERDAS_BARRA=losses_2d,
            considerar_perdas=self.considerar_perdas
        )

    # ------------------------------------------------------------------
    # 5. Objetivo — quadrático + base + déficit
    # ------------------------------------------------------------------
    def _add_FOB(self):
        expr = 0.0

        # ---- Térmicos: a·P² + b·P ----
        for g in range(self.NUTE):
            P = self.PGER_dict[(0, g)]
            expr += self.thermal_cost_quad[g] * P * P
            expr += self.thermal_cost_lin[g] * P

        # ---- Termo constante c_g (snapshot sem UC → sempre ligado) ----
        expr += float(np.sum(self.thermal_cost_base))

        # ---- Déficit ----
        custo_deficit = getattr(self.sistema, 'custo_DEFICIT', 1000.0)
        for b in range(self.NBAR):
            expr += custo_deficit * self.DEFICIT_dict[(0, b)]

        # ---- Curtailment ----
        if self.NGWD > 0:
            custo_curt = getattr(self.sistema, 'custo_curtailment', None)
            for w in range(self.NGWD):
                if custo_curt is None:
                    c = 0.0
                elif hasattr(custo_curt, '__getitem__'):
                    c = float(custo_curt[w])
                else:
                    c = float(custo_curt)
                expr += c * self.CURTAILMENT_dict[(0, w)]

        # ---- Baterias ----
        if self.NBESS > 0:
            arr_ch  = getattr(self.sistema, 'BATTERY_COST_CHARGE', None)
            arr_dch = getattr(self.sistema, 'BATTERY_COST_DISCHARGE', None)
            for i, bus in enumerate(self.battery_buses):
                if arr_ch is not None:
                    c = (float(arr_ch[i]) if hasattr(arr_ch, '__getitem__')
                         and len(arr_ch) > 1 else float(arr_ch))
                    expr += c * self.CHARGE_dict[(0, bus)]
                if arr_dch is not None:
                    c = (float(arr_dch[i]) if hasattr(arr_dch, '__getitem__')
                         and len(arr_dch) > 1 else float(arr_dch))
                    expr += c * self.DISCHARGE_dict[(0, bus)]

        self.model.set_objective(expr, poi.ObjectiveSense.Minimize)

    # ------------------------------------------------------------------
    # 6. Build
    # ------------------------------------------------------------------
    def _build_DC_OPF(self,
                      fator_carga=None,
                      fator_vento=None,
                      soc_baterias=None) -> None:
        self.model = highs.Model()
        self._processa_multiplicadores(fator_carga, fator_vento, soc_baterias)
        self._add_VARS()
        self._add_CONS()
        self._add_FOB()
        self._solved = False

    def _build_Cenario(self, fator_carga=None, fator_vento=None,
                       soc_baterias=None):
        self._build_DC_OPF(fator_carga, fator_vento, soc_baterias)

    # ------------------------------------------------------------------
    # 7. Solve
    # ------------------------------------------------------------------
    def solve(self, solver_name: str = 'highs', write_lp: bool = False,
              **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        if write_lp:
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_filename = caller_frame.f_code.co_filename
            base = os.path.splitext(os.path.basename(caller_filename))[0]
            lp_filename = f"DATA/output_CUR_Oficial/{base}_snapshot.lp"
            os.makedirs(os.path.dirname(lp_filename), exist_ok=True)
            self.model.write(lp_filename)
            print(f"Modelo escrito em {lp_filename}")
        self.model.optimize()
        self._solved = True
        return self.model

    def solve_iterative(self, solver_name: str = 'highs', tol: float = 1e-4,
                        max_iter: int = 50, write_lp: bool = False,
                        **solver_args):
        if not self.considerar_perdas:
            return self.solve(solver_name, write_lp=write_lp, **solver_args)

        self._losses = np.zeros(self.NBAR)
        raw = self.solve(solver_name, write_lp=write_lp, **solver_args)
        if not self._solved or self.model.get_model_attribute(
                poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
            print("Primeira iteração: solução não ótima.")
            return raw

        ang_prev = np.array([self.model.get_value(self.var_lists['ang_pu'][b])
                             for b in range(self.NBAR)])

        for it in range(1, max_iter):
            perdas = self.calculate_losses_from_flow()
            self.update_losses(perdas)

            for _, _, constr in self.balance_constraints:
                self.model.delete_constraint(constr)

            losses_2d = perdas.reshape(1, -1)
            wind_gen_to_bar = self.wind_bus.tolist() if self.NGWD > 0 else None
            self.balance_constraints = ElectricConstraints.add_constraints(
                model=self.model,
                sistema=self.sistema,
                T=1,
                ANG=self.ANG_dict,
                FLUXO_LIN=self.FLOW_dict,
                DEFICIT=self.DEFICIT_dict,
                PLOAD=self.PLOAD.reshape(1, -1),
                PGER=self.PGER_dict,
                conv_gen_to_bar=self.thermal_bus.tolist(),
                PGWIND=self.PGWIND_dict if self.NGWD > 0 else None,
                wind_gen_to_bar=wind_gen_to_bar,
                CHARGE=self.CHARGE_dict if self.NBESS > 0 else None,
                DISCHARGE=self.DISCHARGE_dict if self.NBESS > 0 else None,
                battery_list=self.battery_buses.tolist() if self.NBESS > 0 else None,
                PERDAS_BARRA=losses_2d,
                considerar_perdas=self.considerar_perdas
            )

            raw = self.solve(solver_name, write_lp=False, **solver_args)
            if not self._solved or self.model.get_model_attribute(
                    poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
                print(f"Iteração {it+1}: solução não ótima.")
                break

            ang_curr = np.array([self.model.get_value(self.var_lists['ang_pu'][b])
                                 for b in range(self.NBAR)])
            diff = np.max(np.abs(ang_curr - ang_prev))
            print(f"Iteração {it+1}: diff = {diff:.6f}")
            if diff < tol:
                print(f"Convergência alcançada na iteração {it+1}.")
                break
            ang_prev = ang_curr.copy()

        return raw

    def calculate_losses_from_flow(self):
        perdas_barra = np.zeros(self.NBAR)
        for e in range(self.NLIN):
            flow = self.model.get_value(self.var_lists['flow'][e])
            r = self.line_r[e]
            loss = r * (flow ** 2)
            i = self.line_from[e]
            j = self.line_to[e]
            perdas_barra[i] += loss / 2
            perdas_barra[j] += loss / 2
        return perdas_barra

    def update_losses(self, perdas_barra):
        self._losses = perdas_barra

    # ------------------------------------------------------------------
    # 8. solve_snapshot
    # ------------------------------------------------------------------
    def solve_snapshot(self,
                       solver_name: str = 'highs',
                       fator_carga=None,
                       fator_vento=None,
                       soc_baterias=None,
                       cost_function: Optional[Callable] = None,
                       hora: int = 0,
                       dia: int = 0,
                       cen_id: Optional[str] = None,
                       tol: float = 1e-2,
                       max_iter: int = 50,
                       write_lp: bool = False,
                       verify: bool = False):
        self._build_Cenario(fator_carga, fator_vento, soc_baterias)

        if self.considerar_perdas:
            raw = self.solve_iterative(solver_name, tol=tol,
                                       max_iter=max_iter, write_lp=write_lp)
        else:
            raw = self.solve(solver_name, write_lp=write_lp)

        if self.db_handler is not None and cen_id is not None:
            resultado = self.extract_results(hora=hora, dia=dia, cen_id=cen_id)
            dia_str = f"{dia+1}"
            self.db_handler.save_hourly_result(
                resultado=resultado, sistema=self.sistema,
                hora=hora, solver_name=solver_name,
                dia=dia_str, cen_id=cen_id
            )

        if verify:
            self.print_verification_report(tol=tol)

        return raw

    # ------------------------------------------------------------------
    # 9. Extração
    # ------------------------------------------------------------------
    def extract_results(self, hora: int = 0, dia: int = 0,
                        cen_id: Optional[str] = None) -> OPF_SnapshotResult:
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")

        s = self.sistema
        dias_nomes = ["domingo", "segunda", "terça", "quarta",
                      "quinta", "sexta", "sábado"]
        dia_semana = ((dia) % 7) + 1
        dia_semana_nome = dias_nomes[dia_semana - 1]

        try:
            PLOAD_vals = (self.PLOAD).tolist() if self.PLOAD is not None else []
            PGER_vals  = [self.model.get_value(self.var_lists['PGER_UTE'][g])
                          for g in range(self.NUTE)]

            if self.NGWD > 0:
                PGWIND_disponivel = (self.PGWIND_AVAIL).tolist()
                PGWIND_vals = [self.model.get_value(self.var_lists['p_wind'][w])
                               for w in range(self.NGWD)]
                CURTAILMENT_vals = [self.model.get_value(self.var_lists['curtailment'][w])
                                    for w in range(self.NGWD)]
            else:
                PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

            DEFICIT_vals = [self.model.get_value(self.var_lists['deficit'][b])
                            for b in range(self.NBAR)]

            SOC_init = [0.0] * self.NBAR
            SOC_atual = [0.0] * self.NBAR
            BESS_operation = [0.0] * self.NBAR
            if self.NBESS > 0:
                for i, bus in enumerate(self.battery_buses):
                    SOC_init[bus] = self.soc_inicial_list[i]
                    SOC_atual[bus] = self.model.get_value(self.SOC_dict[(0, bus)])
                    charge    = self.model.get_value(self.CHARGE_dict[(0, bus)])
                    discharge = self.model.get_value(self.DISCHARGE_dict[(0, bus)])
                    BESS_operation[bus] = discharge - charge

            V   = [self.model.get_value(self.var_lists['v_pu'][b])
                   for b in range(self.NBAR)]
            ANG = [self.model.get_value(self.var_lists['ang_pu'][b])
                   for b in range(self.NBAR)]
            FLUXO_LIN = [self.model.get_value(self.var_lists['flow'][e])
                         for e in range(self.NLIN)]

            # ---- Perdas TOTAIS (escalar) ----
            if self.considerar_perdas and self._losses is not None:
                PERDAS_TOTAIS = float(np.sum(self._losses))
            else:
                PERDAS_TOTAIS = 0.0

            DEFICIT_pu = [self.model.get_value(self.var_lists['deficit'][b])
                          for b in range(self.NBAR)]
            custo_deficit_pu = getattr(s, 'custo_DEFICIT', 1000.0)
            CUSTO = [d_pu * custo_deficit_pu for d_pu in DEFICIT_pu]
            CMO = [0.0]

            return OPF_SnapshotResult(
                dia=dia, dia_semana=dia_semana, hora=hora, sucesso=True,
                PLOAD=PLOAD_vals, PGER=PGER_vals,
                PGWIND_disponivel=PGWIND_disponivel,
                PGWIND=PGWIND_vals, CURTAILMENT=CURTAILMENT_vals,
                SOC_init=SOC_init, BESS_operation=BESS_operation,
                SOC_atual=SOC_atual, DEFICIT=DEFICIT_vals,
                V=V, ANG=ANG, FLUXO_LIN=FLUXO_LIN,
                CUSTO=CUSTO, CMO=CMO,
                PERDAS_TOTAIS=PERDAS_TOTAIS,
                dia_semana_nome=dia_semana_nome
            )

        except Exception as e:
            print(f"Erro ao extrair snapshot: {e}")
            traceback.print_exc()
            return OPF_SnapshotResult(
                dia=dia, hora=hora, sucesso=False, mensagem=str(e),
                dia_semana=dia_semana, dia_semana_nome=dia_semana_nome
            )


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
    print("SIMULAÇÃO SNAPSHOT - DC OPF (custo quadrático)")
    print("=" * 70)

    json_path = "DATA/input/3barras_QUAD.json"
    if not os.path.exists(json_path):
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        sys.exit(1)

    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")
    print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
    print(f"   ✓ Barras: {sistema.NBAR}")
    print(f"   ✓ Geradores convencionais: {sistema.NGER_UTE}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_GWD}")
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    db_path = 'DATA/output_CUR_Oficial/resultados_snapshot.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_handler = OPF_DBHandler(db_path)
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S') + "_snapshot"
    print(f"   ✓ Cenário ID: {cen_id}")

    modelo = DCOPFSnapshot(
        sistema=sistema,
        db_handler=db_handler,
        considerar_perdas=True
    )

    hora_desejada = 0
    seed = secrets.randbits(32)
    avaliador = EvaluateFactors(
        sistema=sistema, n_dias=1, n_horas=24,
        carga_incerteza=0.2, vento_variacao=0.1, seed=seed)
    fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()

    fator_carga_hora = fatores_carga_completo[0, hora_desejada, :]
    fator_vento_hora = (fatores_vento_completo[0, hora_desejada, :]
                        if sistema.NGER_GWD > 0 else 1.0)

    print(f"\nParâmetros para Hora {hora_desejada}:")
    print(f"   Fator de carga médio: {np.mean(fator_carga_hora):.3f}")

    soc_baterias = {b: 0.5 for b in sistema.BARRAS_COM_BATERIA}

    raw = modelo.solve_snapshot(
        solver_name='highs',
        fator_carga=fator_carga_hora,
        fator_vento=fator_vento_hora,
        soc_baterias=soc_baterias,
        hora=hora_desejada, dia=0, cen_id=cen_id,
        tol=1e-4, max_iter=10, write_lp=True, verify=False
    )

    status = modelo.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    print(f"\nStatus da solução: {status}")

    if status == poi.TerminationStatusCode.OPTIMAL:
        resultado = modelo.extract_results(hora=hora_desejada, cen_id=cen_id)
        print(f"\nResultados para Hora {hora_desejada}:")
        print(f"   Demanda total: {sum(resultado.PLOAD):.3f} pu")
        print(f"   Perdas: {resultado.PERDAS_TOTAIS:.3f} pu")
        print(f"   Geração térmica total: {sum(resultado.PGER):.3f} pu")
        if sistema.NGER_GWD > 0:
            print(f"   Geração eólica total: {sum(resultado.PGWIND):.3f} pu")
            print(f"   Curtailment total: {sum(resultado.CURTAILMENT):.3f} pu")
        print(f"   Déficit total: {sum(resultado.DEFICIT):.3f} pu")

        print(f"\n   Despacho por gerador (pu):")
        for g in range(sistema.NGER_UTE):
            print(f"     G{g+1}: P = {resultado.PGER[g]:.4f}  "
                  f"(Pmin={sistema.PGER_MIN_UTE[g]:.2f}, "
                  f"Pmax={sistema.PGER_MAX_UTE[g]:.2f})")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)