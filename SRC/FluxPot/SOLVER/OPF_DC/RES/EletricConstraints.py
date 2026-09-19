#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições elétricas: fluxo nas linhas, limites, balanço de potência (PyOptInterface).
Versão adaptada para trabalhar com dicionários de variáveis indexadas por (t, idx).
"""
from __future__ import annotations

import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import Dict, List, Optional, Tuple


class ElectricConstraints:
    """Restrições da rede elétrica para PyOptInterface."""

    @staticmethod
    def add_constraints(
        model: highs.Model,
        sistema,
        T: int,
        ANG: Dict[Tuple[int, int], poi.Variable],
        FLUXO_LIN: Dict[Tuple[int, int], poi.Variable],
        DEFICIT: Dict[Tuple[int, int], poi.Variable],
        PLOAD: np.ndarray,
        PGER: Dict[Tuple[int, int], poi.Variable],
        conv_gen_to_bar: List[int],
        PGWIND: Optional[Dict[Tuple[int, int], poi.Variable]] = None,
        wind_gen_to_bar: Optional[List[int]] = None,
        CHARGE: Optional[Dict[Tuple[int, int], poi.Variable]] = None,
        DISCHARGE: Optional[Dict[Tuple[int, int], poi.Variable]] = None,
        battery_list: Optional[List[int]] = None,
        PERDAS_BARRA: Optional[np.ndarray] = None,
        considerar_perdas: bool = True
    ) -> List[Tuple[int, int, poi.Constraint]]:
        """
        Adiciona restrições de fluxo DC, limites de linha e balanço de potência.
        Retorna a lista de restrições de balanço (cada elemento é uma tupla (t, b, constraint)).

        Parâmetros:
        -----------
        model : highs.Model
            Modelo PyOptInterface.
        sistema : SistemaLoader
            Objeto com dados do sistema.
        T : int
            Número de períodos.
        ANG : dict
            Ângulos das barras, chave (t, b).
        FLUXO_LIN : dict
            Fluxo nas linhas, chave (t, e).
        DEFICIT : dict
            Déficit por barra, chave (t, b).
        PLOAD : np.ndarray
            Demanda por barra, shape (T, NBAR) [pu].
        PGER : dict
            Geração térmica, chave (t, g).
        conv_gen_to_bar : list
            Mapeamento gerador térmico -> barra (índice).
        PGWIND : dict, opcional
            Geração eólica, chave (t, w).
        wind_gen_to_bar : list, opcional
            Mapeamento gerador eólico -> barra.
        CHARGE, DISCHARGE : dict, opcional
            Potências de carga/descarga das baterias, chave (t, b).
        battery_list : list, opcional
            Lista de barras com bateria.
        PERDAS_BARRA : np.ndarray, opcional
            Perdas alocadas por barra, shape (T, NBAR) [pu].
        considerar_perdas : bool
            Se True, inclui PERDAS_BARRA no balanço.

        Retorna:
        --------
        balance_constraints : list
            Lista de tuplas (t, b, constraint) para cada restrição de balanço.
        """
        balance_constraints = []

        # 1. Definição do fluxo de potência (DC)
        for t in range(T):
            for e in range(sistema.NLIN):
                i = sistema.line_fr[e]
                j = sistema.line_to[e]
                x = sistema.x_line[e]
                # expr: FLUXO_LIN[t,e] == (ANG[t,i] - ANG[t,j]) / x
                model.add_linear_constraint(
                    FLUXO_LIN[t, e] - (ANG[t, i] - ANG[t, j]) / x == 0,
                    name=f"flow_def_{t}_{e}"
                )

        # 2. Limites de fluxo (já estão nos bounds das variáveis, mas podemos reforçar)
        # (opcional, comentado para evitar redundância)
        # for t in range(T):
        #     for e in range(sistema.NLIN):
        #         flim = sistema.FLIM[e]
        #         model.add_linear_constraint(FLUXO_LIN[t, e] >= -flim, name=f"flow_lb_{t}_{e}")
        #         model.add_linear_constraint(FLUXO_LIN[t, e] <= flim, name=f"flow_ub_{t}_{e}")

        # 3. Limites de déficit (opcional)
        for t in range(T):
            for b in range(sistema.NBAR):
                model.add_linear_constraint(
                    DEFICIT[t, b] <= 2.0 * PLOAD[t, b],
                    name=f"deficit_limit_{t}_{b}"
                )

        # 4. Balanço de potência em cada barra
        # Pré-calcular índices para agilizar
        thermal_at_bus = [[] for _ in range(sistema.NBAR)]
        for g, bus in enumerate(conv_gen_to_bar):
            thermal_at_bus[bus].append(g)

        wind_at_bus = [[] for _ in range(sistema.NBAR)]
        if PGWIND is not None and wind_gen_to_bar is not None:
            for w, bus in enumerate(wind_gen_to_bar):
                wind_at_bus[bus].append(w)

        battery_at_bus = set(battery_list) if battery_list else set()

        for t in range(T):
            for b in range(sistema.NBAR):
                # Geração térmica
                thermal_sum = 0.0
                for g in thermal_at_bus[b]:
                    thermal_sum += PGER[t, g]

                # Geração eólica
                wind_sum = 0.0
                if PGWIND is not None:
                    for w in wind_at_bus[b]:
                        wind_sum += PGWIND[t, w]

                # Bateria (descarga - carga)
                battery_net = 0.0
                if battery_list is not None and CHARGE is not None and DISCHARGE is not None:
                    if b in battery_at_bus:
                        battery_net = DISCHARGE[t, b] - CHARGE[t, b]

                # Déficit
                deficit = DEFICIT[t, b]

                # Fluxo líquido entrando na barra (positivo se entra)
                flow_net = 0.0
                for e in range(sistema.NLIN):
                    if sistema.line_to[e] == b:
                        flow_net += FLUXO_LIN[t, e]      # fluxo chegando
                    elif sistema.line_fr[e] == b:
                        flow_net -= FLUXO_LIN[t, e]      # fluxo saindo

                # Carga
                load = PLOAD[t, b]

                # Perdas
                losses = PERDAS_BARRA[t, b] if (considerar_perdas and PERDAS_BARRA is not None) else 0.0

                # Equação: geração + bateria + deficit + fluxo_entrando == carga + perdas
                expr = thermal_sum + wind_sum + battery_net + deficit + flow_net - load - losses
                constr = model.add_linear_constraint(expr == 0, name=f"balance_{t}_{b}")
                balance_constraints.append((t, b, constr))

        return balance_constraints