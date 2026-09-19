#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para baterias em modelo multi‑período (Pyomo).
Todas as grandezas em pu.
"""
from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Optional, Tuple


class BatteryConstraints:
    """Restrições de bateria para modelos AC e DC, com horizonte de tempo."""
    
    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        sistema,
        T: int,
        battery_list: List[int],
        CHARGE: Dict[Tuple[int, int], pyo.Var],
        DISCHARGE: Dict[Tuple[int, int], pyo.Var],
        SOC: Dict[Tuple[int, int], pyo.Var],
        BatteryOperation: Dict[Tuple[int, int], pyo.Var],
        soc_inicial_list: List[float],
        soc_final_list: Optional[List[float]] = None,
        daily_reset_to_initial: bool = False
    ) -> None:
        """
        Adiciona restrições de bateria ao modelo Pyomo.

        Parâmetros
        ----------
        model : pyo.ConcreteModel
            Modelo Pyomo.
        sistema : SistemaLoader
            Objeto com dados do sistema.
        T : int
            Número de períodos.
        battery_list : list
            Lista de índices das barras com bateria.
        CHARGE, DISCHARGE, SOC, BatteryOperation : dict
            Dicionários com variáveis Pyomo (chave (t, b)).
        soc_inicial_list : list
            SOC inicial (pu) para cada bateria, na ordem de battery_list.
        soc_final_list : list, opcional
            SOC final desejado (pu) para cada bateria.
        daily_reset_to_initial : bool
            Se True e soc_final_list não fornecida, impõe SOC[T-1] = SOC_inicial.
        """
        if not battery_list:
            return

        eff_carga = getattr(sistema, 'BATTERY_CHARGE_EFF', 1.0)
        eff_descarga = getattr(sistema, 'BATTERY_DISCHARGE_EFF', 1.0)

        # 1. Relação entre operação líquida e potências
        for t in range(T):
            for b in battery_list:
                setattr(
                    model,
                    f"BatteryOperation_def_{t}_{b}",
                    pyo.Constraint(expr=BatteryOperation[t, b] == DISCHARGE[t, b] - CHARGE[t, b])
                )

        # 2. Condição inicial (t = 0)
        for i, b in enumerate(battery_list):
            setattr(
                model,
                f"SOC_init_{b}",
                pyo.Constraint(
                    expr=SOC[0, b] == soc_inicial_list[i]
                    + eff_carga * CHARGE[0, b]
                    - DISCHARGE[0, b] / eff_descarga
                )
            )

        # 3. Evolução do SOC para t >= 1
        for t in range(1, T):
            for b in battery_list:
                setattr(
                    model,
                    f"SOC_evolution_{t}_{b}",
                    pyo.Constraint(
                        expr=SOC[t, b] == SOC[t-1, b]
                        + eff_carga * CHARGE[t, b]
                        - DISCHARGE[t, b] / eff_descarga
                    )
                )

        # 4. SOC final (se especificado ou reset diário)
        if soc_final_list is not None:
            for i, b in enumerate(battery_list):
                setattr(
                    model,
                    f"SOC_final_{b}",
                    pyo.Constraint(expr=SOC[T-1, b] == soc_final_list[i])
                )
        elif daily_reset_to_initial:
            for i, b in enumerate(battery_list):
                setattr(
                    model,
                    f"SOC_final_reset_{b}",
                    pyo.Constraint(expr=SOC[T-1, b] == soc_inicial_list[i])
                )