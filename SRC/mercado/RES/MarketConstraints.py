#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para Mercado de Energia – versão Pyomo.
Balanço de contratos: entrega + déficit = quantidade contratada.
"""
from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Tuple, Optional, Any


class MarketConstraints:
    """
    Restrições de contratos de mercado para modelos Pyomo.

    Parâmetros
    ----------
    model              : modelo Pyomo
    T                  : horizonte de tempo (int)
    contratos          : lista de dicts, cada um com pelo menos a chave 'Amount'
    CONTRATO_ENTREGA   : dict {(t, c_idx): var/expr} energia efetivamente entregue
    CONTRATO_DEFICIT   : dict {(t, c_idx): var/expr} déficit do contrato
    """

    @staticmethod
    def add_constraints(
        model,
        T: int,
        contratos: List[Dict[str, Any]],
        CONTRATO_ENTREGA: Dict[Tuple[int, int], Any],
        CONTRATO_DEFICIT: Dict[Tuple[int, int], Any],
    ):

        for t in range(T):
            for c_idx, c in enumerate(contratos):

                setattr(
                    model,
                    f"CONT_BAL_T{t}_C{c_idx+1}",
                    pyo.Constraint(
                        expr=CONTRATO_ENTREGA[(t, c_idx)]
                             + CONTRATO_DEFICIT[(t, c_idx)]
                             == c['Amount']
                    )
                )