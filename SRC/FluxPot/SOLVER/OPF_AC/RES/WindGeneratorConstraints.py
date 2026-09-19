#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores eólicos em modelo multi-período (Pyomo).
Todas as grandezas de potência estão em pu (por unidade).
"""
from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from typing import Dict, Union


class WindGeneratorConstraints:
    """Restrições de geradores eólicos para Pyomo."""

    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        T: int,
        NGER_GWD: int,
        PGWIND: Dict,
        CURTAILMENT: Dict,
        PGWIND_AVAIL: Union[np.ndarray, Dict],
    ):
        """
        Adiciona restrições de balanço para geradores eólicos.

        Parâmetros
        ----------
        model : pyo.ConcreteModel
            Modelo Pyomo.
        T : int
            Número de períodos.
        NGER_GWD : int
            Número de geradores eólicos.
        PGWIND : dict
            Dicionário de variáveis Pyomo (chave (t, w)) → pyo.Var.
        CURTAILMENT : dict
            Dicionário de variáveis Pyomo (chave (t, w)) → pyo.Var.
        PGWIND_AVAIL : np.ndarray ou dict
            Disponibilidade eólica (pu) para cada (t, w).
            Se array, shape (T, NGER_GWD); se dict, chave (t, w).
        """
        if NGER_GWD == 0:
            return

        # Função de acesso à disponibilidade
        if isinstance(PGWIND_AVAIL, np.ndarray):
            def get_avail(t, w):
                return PGWIND_AVAIL[t, w]
        else:
            def get_avail(t, w):
                return PGWIND_AVAIL.get((t, w), 0.0)

        for t in range(T):
            for w in range(NGER_GWD):
                avail = get_avail(t, w)
                setattr(
                    model,
                    f"wind_balance_{t}_{w}",
                    pyo.Constraint(expr=PGWIND[t, w] + CURTAILMENT[t, w] == avail)
                )