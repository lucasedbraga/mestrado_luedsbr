#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores térmicos – versão Pyomo (sem UC).

  • Limites fixos: Pmin ≤ P ≤ Pmax
  • Reativo acoplado ao fator de potência: QGER = FP · PGER
  • Limites de reativo: Qmin ≤ Q ≤ Qmax
  • Rampas clássicas:
        t = 0 → compara com PGER_INICIAL_UTE
        t > 0 → compara com PGER[t-1]
"""
from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Tuple, Optional, Union


class ThermalGeneratorConstraints:
    """
    Restrições de geradores térmicos para modelos Pyomo — sem UC.
    """

    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        T: int,
        NGER_UTE: int,
        PGER: Dict[Tuple[int, int], pyo.Var],
        QGER: Dict[Tuple[int, int], pyo.Var],
        FATOR_POTENCIA: Union[float, List[float]],
        PGER_MIN_UTE: Optional[List[float]] = None,
        PGER_MAX_UTE: Optional[List[float]] = None,
        QGER_MIN_UTE: Optional[List[float]] = None,
        QGER_MAX_UTE: Optional[List[float]] = None,
        PGER_INICIAL_UTE: Optional[List[float]] = None,
        RAMP_UP: Optional[List[float]] = None,
        RAMP_DOWN: Optional[List[float]] = None,
        SB: float = 1.0,
    ) -> None:
        """
        Adiciona limites de geração ativa/reativa, acoplamento Q = FP·P e rampas.

        Parâmetros
        ----------
        model : pyo.ConcreteModel
            Modelo Pyomo.
        T : int
            Número de períodos.
        NGER_UTE : int
            Número de geradores convencionais.
        PGER : dict[(t, g), Var]
            Variáveis de potência ativa.
        QGER : dict[(t, g), Var]
            Variáveis de potência reativa.
        FATOR_POTENCIA : float ou list[float]
            Fator de potência fixo. Escalar aplica a todos; lista deve
            ter NGER_UTE valores em (0, 1].
        PGER_MIN_UTE, PGER_MAX_UTE : list[float], opcional
            Limites de potência ativa (pu).
        QGER_MIN_UTE, QGER_MAX_UTE : list[float], opcional
            Limites de potência reativa (pu).
        PGER_INICIAL_UTE : list[float], opcional
            Geração ativa inicial (antes do período 0) para rampa.
        RAMP_UP, RAMP_DOWN : list[float], opcional
            Taxas de rampa (pu/h).
        SB : float
            Potência base (MVA).
        """
        if NGER_UTE == 0:
            return

        if PGER_MIN_UTE is None:
            PGER_MIN_UTE = [0.0] * NGER_UTE
        if PGER_MAX_UTE is None:
            PGER_MAX_UTE = [1.0] * NGER_UTE

        if np.isscalar(FATOR_POTENCIA):
            fp_list = [float(np.acos(FATOR_POTENCIA))] * NGER_UTE
        else:
            fp_list = list(FATOR_POTENCIA)
            if len(fp_list) != NGER_UTE:
                raise ValueError(
                    f"FATOR_POTENCIA deve ter {NGER_UTE} valores "
                    f"(ou ser escalar). Recebido: {len(fp_list)}")

        # Validação numérica de cada FP
        for g in range(NGER_UTE):
            fp = float(fp_list[g])
            if fp <= 0.0 or fp > 1.0:
                raise ValueError(
                    f"FATOR_POTENCIA[{g}] = {fp} inválido. "
                    f"Deve estar em (0, 1].")

        # =========================================================
        # 1. Limites de potência ativa
        # =========================================================
        for t in range(T):
            for g in range(NGER_UTE):
                setattr(model, f"PGER_MIN_UTE_{t}_{g}",
                        pyo.Constraint(expr=PGER[t, g] >= PGER_MIN_UTE[g]))
                setattr(model, f"PGER_MAX_UTE_{t}_{g}",
                        pyo.Constraint(expr=PGER[t, g] <= PGER_MAX_UTE[g]))

        # =========================================================
        # 2. Acoplamento reativo: QGER = FP · PGER
        # =========================================================
        for t in range(T):
            for g in range(NGER_UTE):
                setattr(model, f"QGER_FP_{t}_{g}",
                        pyo.Constraint(
                            expr=QGER[t, g] == fp_list[g] * PGER[t, g]))

        # =========================================================
        # 3. Limites de potência reativa
        # =========================================================
        if QGER_MIN_UTE is not None and QGER_MAX_UTE is not None:
            for t in range(T):
                for g in range(NGER_UTE):
                    setattr(model, f"QGER_MIN_UTE_{t}_{g}",
                            pyo.Constraint(
                                expr=QGER[t, g] >= QGER_MIN_UTE[g]))
                    setattr(model, f"QGER_MAX_UTE_{t}_{g}",
                            pyo.Constraint(
                                expr=QGER[t, g] <= QGER_MAX_UTE[g]))

        # =========================================================
        # 4. Rampas 
        # =========================================================
        if (RAMP_UP is not None and RAMP_DOWN is not None
                and PGER_INICIAL_UTE is not None):

            # t = 0 : a partir da geração inicial
            for g in range(NGER_UTE):
                setattr(model, f"first_ramp_up_{g}",
                        pyo.Constraint(
                            expr=PGER[0, g]
                                 <= PGER_INICIAL_UTE[g] + RAMP_UP[g]))
                setattr(model, f"first_ramp_down_{g}",
                        pyo.Constraint(
                            expr=PGER[0, g]
                                 >= PGER_INICIAL_UTE[g] - RAMP_DOWN[g]))

            # t ≥ 1 : entre períodos consecutivos
            for t in range(1, T):
                for g in range(NGER_UTE):
                    setattr(model, f"ramp_up_{t}_{g}",
                            pyo.Constraint(
                                expr=PGER[t, g]
                                     <= PGER[t - 1, g] + RAMP_UP[g]))
                    setattr(model, f"ramp_down_{t}_{g}",
                            pyo.Constraint(
                                expr=PGER[t, g]
                                     >= PGER[t - 1, g] - RAMP_DOWN[g]))