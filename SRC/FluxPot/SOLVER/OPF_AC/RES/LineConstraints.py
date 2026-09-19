#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições de fluxo de potência aparente nas linhas AC (limite térmico).
Utiliza coordenadas polares (V, θ) e a matriz de admitância.
"""
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Optional, Tuple


class AC_LineConstraints:
    """Restrições de capacidade das linhas (S_ij ≤ FLIM)."""

    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        sistema,
        HORA: int,
        V: Dict[Tuple[int, int], pyo.Var],            # (t, bus) → variável de magnitude
        ANG: Dict[Tuple[int, int], pyo.Var],          # (t, bus) → variável de ângulo
        line_from: np.ndarray,                        # barra de origem de cada linha
        line_to: np.ndarray,                          # barra de destino de cada linha
        line_r: np.ndarray,                           # resistência série (pu)
        line_x: np.ndarray,                           # reatância série (pu)
        line_flow_max: np.ndarray,                    # capacidade térmica (pu) – FLIM
    ) -> List[Tuple[int, int, pyo.Constraint]]:
        """
        Adiciona ao modelo as restrições de fluxo aparente nas linhas:
            P_flow[t, e]^2 + Q_flow[t, e]^2 <= FLIM[e]^2

        Retorna uma lista de tuplas (hora, linha, constraint) para referência.
        """
        NBAR = sistema.NBAR
        NLIN = sistema.NLIN

        # Conjunto de índices (hora, linha) para as restrições
        model.TimeLineSet = pyo.Set(initialize=[(t, e) for t in range(HORA) for e in range(NLIN)])

        # Pré‑calcular parâmetros constantes das linhas
        z2 = line_r**2 + line_x**2
        # Evita divisão por zero – linhas com impedância nula são ignoradas
        g = np.where(z2 > 0, line_r / z2, 0.0)
        b = np.where(z2 > 0, -line_x / z2, 0.0)

        def line_flow_rule(m, t, e):
            i = line_from[e]
            j = line_to[e]
            Vi = V[t, i]
            Vj = V[t, j]
            delta = ANG[t, i] - ANG[t, j]

            # Cálculo dos fluxos ativo e reativo (sentido i → j)
            P_flow = Vi**2 * g[e] - Vi * Vj * (g[e] * pyo.cos(delta) + b[e] * pyo.sin(delta))
            Q_flow = -Vi**2 * b[e] - Vi * Vj * (g[e] * pyo.sin(delta) - b[e] * pyo.cos(delta))

            # Restrição de potência aparente
            return P_flow**2 + Q_flow**2 <= line_flow_max[e]**2

        model.LineFlowLimit = pyo.Constraint(model.TimeLineSet, rule=line_flow_rule)

        # Preenche lista de retorno (para eventuais manipulações posteriores)
        constraints_list = []
        for t in range(HORA):
            for e in range(NLIN):
                constraints_list.append((t, e, model.LineFlowLimit[t, e]))

        return constraints_list