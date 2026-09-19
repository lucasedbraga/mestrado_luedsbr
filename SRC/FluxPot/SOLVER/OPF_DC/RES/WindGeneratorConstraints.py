#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores eólicos em modelo multi-período (PyOptInterface).
Todas as grandezas de potência estão em pu (por unidade).
"""
from __future__ import annotations
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import Dict, Union


class WindGeneratorConstraints:
    """Restrições de geradores eólicos para PyOptInterface."""

    @staticmethod
    def add_constraints(
        model: highs.Model,
        T: int,
        NGER_EOL: int,
        PGWIND: Dict,
        CURTAILMENT: Dict,
        PGWIND_AVAIL: Union[np.ndarray, Dict],
    ):
        """
        Adiciona restrições de balanço para geradores eólicos.

        Parâmetros
        ----------
        model : highs.Model
            Modelo PyOptInterface.
        T : int
            Número de períodos.
        NGER_EOL : int
            Número de geradores eólicos.
        PGWIND : dict
            Dicionário de variáveis de geração eólica, chave (t, w) (valores em pu).
        CURTAILMENT : dict
            Dicionário de variáveis de curtailment, chave (t, w) (valores em pu).
        PGWIND_AVAIL : np.ndarray ou dict
            Disponibilidade eólica (em pu) para cada (t, w).
            Se for array, deve ter shape (T, NGER_EOL). Se for dict, chave (t, w).
        """
        if NGER_EOL == 0:
            return

        # Função de acesso à disponibilidade
        if isinstance(PGWIND_AVAIL, np.ndarray):
            def get_avail(t, w):
                return PGWIND_AVAIL[t, w]
        else:
            # assume dicionário com chave (t, w)
            def get_avail(t, w):
                return PGWIND_AVAIL.get((t, w), 0.0)

        for t in range(T):
            for w in range(NGER_EOL):
                avail = get_avail(t, w)

                # Balanço: geração + curtailment = disponibilidade
                model.add_linear_constraint(
                    PGWIND[t, w] + CURTAILMENT[t, w] == avail,
                    name=f"wind_balance_{t}_{w}"
                )

                # (Opcional) limite superior de curtailment – já implícito pela igualdade,
                # mas pode ser mantido por clareza.
                # model.add_linear_constraint(CURTAILMENT[t, w] <= avail, name=f"curtailment_limit_{t}_{w}")