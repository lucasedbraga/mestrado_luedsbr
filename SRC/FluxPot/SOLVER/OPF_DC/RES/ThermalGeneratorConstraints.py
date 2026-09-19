#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores térmicos com mínimo desligamento (3 períodos).
Todos os valores de potência em pu (por unidade).
"""
from __future__ import annotations
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import Dict, List, Tuple, Optional


class ThermalGeneratorConstraints:
    """
    Restrições de geradores térmicos: limites de geração, rampas e mínimo desligamento.
    """

    @staticmethod
    def add_constraints(
        model: highs.Model,
        T: int,
        NGER_CONV: int,
        PGER: Dict[Tuple[int, int], poi.Variable],
        PGER_MIN_UTE: List[float],
        PGER_MAX_UTE: List[float],
        PGER_INICIAL_UTE: List[float],
        RAMP_UP: List[float],
        RAMP_DOWN: List[float],
        SB: float
    ) -> None:
        """
        (Método original) Adiciona apenas restrições de limites e rampa.
        """
        if NGER_CONV == 0:
            return

        # Converte rampas de MW/h para pu/h
        ramp_up_pu = [r / SB for r in RAMP_UP]
        ramp_down_pu = [r / SB for r in RAMP_DOWN]

        # Limites de geração
        for t in range(T):
            for g in range(NGER_CONV):
                model.add_linear_constraint(
                    PGER[t, g] >= PGER_MIN_UTE[g],
                    name=f"gen_lb_{t}_{g}"
                )
                model.add_linear_constraint(
                    PGER[t, g] <= PGER_MAX_UTE[g],
                    name=f"gen_ub_{t}_{g}"
                )

        # Rampa no primeiro período (comparação com inicial)
        for g in range(NGER_CONV):
            model.add_linear_constraint(
                PGER[0, g] <= PGER_INICIAL_UTE[g] + ramp_up_pu[g],
                name=f"first_ramp_up_{g}"
            )
            model.add_linear_constraint(
                PGER[0, g] >= PGER_INICIAL_UTE[g] - ramp_down_pu[g],
                name=f"first_ramp_down_{g}"
            )

        # Rampas nos demais períodos
        for t in range(1, T):
            for g in range(NGER_CONV):
                model.add_linear_constraint(
                    PGER[t, g] <= PGER[t-1, g] + ramp_up_pu[g],
                    name=f"ramp_up_{t}_{g}"
                )
                model.add_linear_constraint(
                    PGER[t, g] >= PGER[t-1, g] - ramp_down_pu[g],
                    name=f"ramp_down_{t}_{g}"
                )

    @staticmethod
    def add_constraints_with_min_downtime(
        model: highs.Model,
        T: int,
        NGER_CONV: int,
        PGER: Dict[Tuple[int, int], poi.Variable],
        PGER_MIN_UTE: List[float],
        PGER_MAX_UTE: List[float],
        PGER_INICIAL_UTE: List[float],
        RAMP_UP: List[float],
        RAMP_DOWN: List[float],
        SB: float,
        min_downtime: int = 4
    ) -> Dict[Tuple[int, int], poi.Variable]:
        """
        Adiciona restrições de limites, rampa e mínimo desligamento.
        Retorna um dicionário com as variáveis binárias de commitment u[t, g].

        Parâmetros adicionais:
        ----------------------
        min_downtime : int
            Número mínimo de períodos que o gerador deve permanecer desligado
            após atingir zero (padrão = 4).
        """
        if NGER_CONV == 0:
            return {}

        # Converte rampas para pu
        ramp_up_pu = [r / SB for r in RAMP_UP]
        ramp_down_pu = [r / SB for r in RAMP_DOWN]

        # 1. Criar variáveis binárias de commitment u[t, g]
        u = {}
        for t in range(T):
            for g in range(NGER_CONV):
                u[t, g] = model.add_variable(
                    domain=poi.VariableDomain.Binary,
                    name=f"u_{t}_{g}"
                )

        # 2. Vínculos entre PGER e u (acoplamento)
        for t in range(T):
            for g in range(NGER_CONV):
                # Se u = 0, PGER deve ser 0
                model.add_linear_constraint(
                    PGER[t, g] <= PGER_MAX_UTE[g] * u[t, g],
                    name=f"link_ub_{t}_{g}"
                )
                # Se u = 1, PGER deve ser >= pgmin (se pgmin > 0)
                if PGER_MIN_UTE[g] > 0:
                    model.add_linear_constraint(
                        PGER[t, g] >= PGER_MIN_UTE[g] * u[t, g],
                        name=f"link_lb_{t}_{g}"
                    )
                # Limites originais (já nos bounds) são redundantes, mas mantidos por segurança
                model.add_linear_constraint(
                    PGER[t, g] >= 0,  # já implícito, mas explícito
                    name=f"gen_lb_{t}_{g}"
                )
                model.add_linear_constraint(
                    PGER[t, g] <= PGER_MAX_UTE[g],
                    name=f"gen_ub_{t}_{g}"
                )

        # 3. Determinar estado inicial (antes do período 0)
        u_inicial = [1 if p > 0 else 0 for p in PGER_INICIAL_UTE]

        # 4. Restrição de permanência mínima de 2 horas no mesmo nível
        #(a geração não pode mudar em dois períodos consecutivos)
        #A primeira transição (t=0 -> t=1) é livre.
        M_big = [PGER_MAX_UTE[g] for g in range(NGER_CONV)]  # valor máximo para big-M

        # Criar variáveis binárias para indicar mudança entre t-1 e t
        delta = {}
        for t in range(1, T): 
            for g in range(NGER_CONV):
                delta[t, g] = model.add_variable(
                    domain=poi.VariableDomain.Binary,
                    name=f"delta_{t}_{g}"
                )

        # Relacionar delta com a diferença de geração
        for t in range(1, T):
            for g in range(NGER_CONV):
                # Diferença entre período atual e anterior
                diff = PGER[t, g] - PGER[t-1, g]
                # -M * delta <= diff <= M * delta
                model.add_linear_constraint(
                    diff >= -M_big[g] * delta[t, g],
                    name=f"delta_low_{t}_{g}"
                )
                model.add_linear_constraint(
                    diff <= M_big[g] * delta[t, g],
                    name=f"delta_up_{t}_{g}"
                )

        # Proibir duas mudanças consecutivas (para t >= 2)
        for t in range(2, T):
            for g in range(NGER_CONV):
                model.add_linear_constraint(
                    delta[t, g] + delta[t-1, g] <= 1,
                    name=f"no_two_changes_{t}_{g}"
                )


        # 5. Restrições de rampa (originais, agora com u implícito)
        for g in range(NGER_CONV):
            # Primeiro período
            model.add_linear_constraint(
                PGER[0, g] <= PGER_INICIAL_UTE[g] + ramp_up_pu[g],
                name=f"first_ramp_up_{g}"
            )
            model.add_linear_constraint(
                PGER[0, g] >= PGER_INICIAL_UTE[g] - ramp_down_pu[g],
                name=f"first_ramp_down_{g}"
            )

        for t in range(1, T):
            for g in range(NGER_CONV):
                model.add_linear_constraint(
                    PGER[t, g] <= PGER[t-1, g] + ramp_up_pu[g],
                    name=f"ramp_up_{t}_{g}"
                )
                model.add_linear_constraint(
                    PGER[t, g] >= PGER[t-1, g] - ramp_down_pu[g],
                    name=f"ramp_down_{t}_{g}"
                )
        
        # =====================================================================
        # Restrição de mínimo desligamento: se desligar, deve ficar 2h off
        # =====================================================================
        min_downtime = 2

        # Determinar estado inicial (antes do período 0) a partir da geração inicial
        u_inicial = [1 if p > 0 else 0 for p in PGER_INICIAL_UTE]

        for g in range(NGER_CONV):
            # Para todos os t onde a janela de 2 períodos cabe no horizonte
            for t in range(0, T - min_downtime + 1):
                # u_prev representa o estado no período anterior a t:
                # se t == 0, usa o estado inicial; caso contrário, usa u[t-1, g]
                if t == 0:
                    u_prev = u_inicial[g]
                else:
                    u_prev = u[t-1, g]

                # Soma de (1 - u) nos períodos t e t+1
                sum_off = (1 - u[t, g]) + (1 - u[t+1, g])

                # Restrição: sum_off >= 2 * (u_prev - u[t, g])
                model.add_linear_constraint(
                    sum_off >= min_downtime * (u_prev - u[t, g]),
                    name=f"min_downtime_2_{g}_t{t}"
                )

        return u