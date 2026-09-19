#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições de balanço de potência AC (P e Q) – coordenadas polares (V, θ).
As rotinas de balanço foram separadas em add_P_Balance e add_Q_Balance.
"""
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Optional, Tuple


class AC_BalanceConstraints:
    """Restrições de balanço de potência ativa e reativa."""

    # ------------------------------------------------------------------
    # Rotinas individuais de balanço (chamadas por add_constraints)
    # ------------------------------------------------------------------
    @staticmethod
    def add_P_Balance(
        m: pyo.ConcreteModel,
        t: int,
        i: int,
        V: Dict[Tuple[int, int], pyo.Var],
        ANG: Dict[Tuple[int, int], pyo.Var],
        G: np.ndarray,
        B: np.ndarray,
        PGER: Dict[Tuple[int, int], pyo.Var],
        PGWIND: Optional[Dict[Tuple[int, int], pyo.Var]],
        DEFICIT: Dict[Tuple[int, int], pyo.Var],
        PLOAD: np.ndarray,
        UTE_na_BARRA: List[List[int]],
        GWD_na_BARRA: List[List[int]],
        BESS_na_BARRA: List[List[int]],
        BESS_SOC_op: Optional[Dict[Tuple[int, int], pyo.Var]]
    ):
        """
        Retorna a expressão da restrição de balanço de potência ativa
        para a barra i no instante t:
        P_inj = P_ger + P_BESS_op + P_def - P_load
        """
        NBAR = G.shape[0]

        Vi = V[t, i]
        theta_i = ANG[t, i]

        sum_active = 0.0
        for j in range(NBAR):
            g_ij = G[i, j]
            b_ij = B[i, j]
            if g_ij == 0.0 and b_ij == 0.0:
                continue
            Vj = V[t, j]
            delta = theta_i - ANG[t, j]
            sum_active += Vj * (g_ij * pyo.cos(delta) + b_ij * pyo.sin(delta))
        P_inj = Vi * sum_active

        # Geração térmica
        P_ger = sum(PGER[t, g] for g in UTE_na_BARRA[i])

        # Geração eólica
        if PGWIND is not None:
            P_ger += sum(PGWIND[t, w] for w in GWD_na_BARRA[i])

        # Operação líquida da bateria (já é descarga - carga)
        P_BESS_op = 0.0
        if BESS_na_BARRA and i in BESS_na_BARRA and BESS_SOC_op is not None:
            P_BESS_op = BESS_SOC_op[t, i]

        P_def = DEFICIT[t, i]
        P_load = PLOAD[t, i]

        # Balanço: P_inj - (geração + bateria + déficit - carga) = 0
        return P_inj - (P_ger + P_BESS_op + P_def - P_load) == 0

    # ------------------------------------------------------------------
    @staticmethod
    def add_Q_Balance(
        m: pyo.ConcreteModel,
        t: int,
        i: int,
        V: Dict[Tuple[int, int], pyo.Var],
        ANG: Dict[Tuple[int, int], pyo.Var],
        G: np.ndarray,
        B: np.ndarray,
        QGER: Dict[Tuple[int, int], pyo.Var],
        QLOAD: np.ndarray,
        UTE_na_BARRA: List[List[int]]
    ):
        """
        Retorna a expressão da restrição de balanço de potência reativa
        para a barra i no instante t:
        Q_inj = Q_ger - Q_load
        """
        NBAR = G.shape[0]

        Vi = V[t, i]
        theta_i = ANG[t, i]

        sum_reactive = 0.0
        for j in range(NBAR):
            g_ij = G[i, j]
            b_ij = B[i, j]
            if g_ij == 0.0 and b_ij == 0.0:
                continue
            Vj = V[t, j]
            delta = theta_i - ANG[t, j]
            sum_reactive += Vj * (g_ij * pyo.sin(delta) - b_ij * pyo.cos(delta))
        Q_inj = Vi * sum_reactive

        Q_ger = sum(QGER[t, g] for g in UTE_na_BARRA[i])
        Q_load = QLOAD[t, i]

        return Q_inj - (Q_ger - Q_load) == 0

    # ------------------------------------------------------------------
    # Método principal que utiliza as duas rotinas acima
    # ------------------------------------------------------------------
    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        sistema,
        HORA: int,
        G: np.ndarray,
        B: np.ndarray,
        V: Dict[Tuple[int, int], pyo.Var],
        ANG: Dict[Tuple[int, int], pyo.Var],
        PGER: Dict[Tuple[int, int], pyo.Var],
        QGER: Dict[Tuple[int, int], pyo.Var],
        PLOAD: np.ndarray,            # shape (HORA, n_bus)
        QLOAD: np.ndarray,
        DEFICIT: Dict[Tuple[int, int], pyo.Var],
        PGWIND: Optional[Dict[Tuple[int, int], pyo.Var]] = None,
        BESS_SOC_op: Optional[Dict[Tuple[int, int], pyo.Var]] = None,
        conv_gen_to_bar: Optional[List[int]] = None,
        wind_gen_to_bar: Optional[List[int]] = None,
        battery_list: Optional[List[int]] = None
    ) -> List[Tuple[int, int, pyo.Constraint]]:
        """
        Adiciona ao modelo as restrições de balanço P e Q para todas as horas e barras.
        Retorna uma lista de tuplas (hora, barra, constraint) para uso externo.
        """
        NBAR = sistema.NBAR
        RestricoesDeBalanco = []

        # Conjunto de índices (hora, i) para as restrições
        model.TimeBusSet = pyo.Set(initialize=[(t, i) for t in range(HORA) for i in range(NBAR)])

        # Mapeamentos gerador → barra
        UTE_na_BARRA = [[] for _ in range(NBAR)]
        if conv_gen_to_bar is not None:
            for g, bus in enumerate(conv_gen_to_bar):
                UTE_na_BARRA[bus].append(g)

        GWD_na_BARRA = [[] for _ in range(NBAR)]
        if PGWIND is not None and wind_gen_to_bar is not None:
            for w, bus in enumerate(wind_gen_to_bar):
                GWD_na_BARRA[bus].append(w)

        BESS_na_BARRA = [[] for _ in range(NBAR)]
        if battery_list is not None and BESS_SOC_op is not None:
            for b in battery_list:
                BESS_na_BARRA[b].append(b)   # apenas marcamos que a barra tem bateria

        # Regras para as constraints que chamam as rotinas estáticas
        def C_Balanco_P(m, t, i):
            return AC_BalanceConstraints.add_P_Balance(
                m, t, i, V, ANG, G, B, PGER, PGWIND, DEFICIT, PLOAD,
                UTE_na_BARRA, GWD_na_BARRA, BESS_na_BARRA, BESS_SOC_op
            )

        def C_Balanco_Q(m, t, i):
            return AC_BalanceConstraints.add_Q_Balance(
                m, t, i, V, ANG, G, B, QGER, QLOAD,
                UTE_na_BARRA
            )

        # Criação das constraints indexadas
        model.Balanco_PotenciaAtiva = pyo.Constraint(model.TimeBusSet, rule=C_Balanco_P)
        model.Balanco_PotenciaReativa = pyo.Constraint(model.TimeBusSet, rule=C_Balanco_Q)

        # Preenche lista de retorno
        for t in range(HORA):
            for i in range(NBAR):
                RestricoesDeBalanco.append((t, i, model.Balanco_PotenciaAtiva[t, i]))
                RestricoesDeBalanco.append((t, i, model.Balanco_PotenciaReativa[t, i]))

        return RestricoesDeBalanco