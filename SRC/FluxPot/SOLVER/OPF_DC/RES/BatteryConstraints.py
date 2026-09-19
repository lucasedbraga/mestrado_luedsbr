#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para baterias em modelo multi‑período (PyOptInterface).
Atende aos requisitos:
    - Parâmetro SOC_inicial (fixo) para o primeiro período.
    - Variável SOC[t,b] representando o SOC ao final do período t.
    - Restrição opcional de SOC_final aplicada ao último período.
Todas as grandezas em pu.
"""
from __future__ import annotations
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import Dict, List, Optional, Tuple


class BatteryConstraints:
    """Restrições de bateria para modelos com horizonte de tempo."""
    
    @staticmethod
    def add_constraints(
        model: highs.Model,
        sistema,
        T: int,
        battery_list: List[int],
        CHARGE: Dict[Tuple[int, int], poi.Variable],
        DISCHARGE: Dict[Tuple[int, int], poi.Variable],
        SOC: Dict[Tuple[int, int], poi.Variable],
        BatteryOperation: Dict[Tuple[int, int], poi.Variable],
        soc_inicial_list: List[float],
        soc_final_list: Optional[List[float]] = None,
        daily_reset_to_initial: bool = False
    ) -> None:
        """
        Adiciona variáveis e restrições de bateria ao modelo PyOptInterface.

        Parâmetros
        ----------
        model : highs.Model
            Modelo PyOptInterface.
        sistema : SistemaLoader
            Objeto com dados do sistema (BATTERY_CAPACITY, BATTERY_MIN_SOC, etc.).
        T : int
            Número de períodos.
        battery_list : list
            Lista de índices das barras que possuem bateria.
        battery_index : dict, opcional
            Mapeamento barra -> índice (não utilizado, mantido por compatibilidade).
        CHARGE, DISCHARGE, SOC, BatteryOperation : dict
            Dicionários que serão preenchidos com as variáveis criadas.
            Chave (t, b) para t em range(T) e b em battery_list.
        soc_inicial_list : list
            Valores iniciais de SOC (pu) para cada bateria, na ordem de battery_list.
        soc_final_list : list, opcional
            Valores finais desejados de SOC (pu) para cada bateria.
            Se fornecido, adiciona restrição SOC[T-1, b] == valor.
        """
        if not battery_list:
            return

        # Eficiências (adimensionais)
        eff_carga = getattr(sistema, 'BATTERY_CHARGE_EFF', 1.0)
        eff_descarga = getattr(sistema, 'BATTERY_DISCHARGE_EFF', 1.0)

        # Função auxiliar para obter parâmetros por barra
        def get_val(attr, barra, default=0.0):
            val = getattr(sistema, attr, None)
            if val is None:
                return default
            if isinstance(val, (list, tuple, np.ndarray)):
                return val[barra] if barra < len(val) else default
            if isinstance(val, dict):
                return val.get(barra, default)
            return val  # valor único

        # --------------------------------------------------------------------
        # 1. Criação das variáveis (já deve ter sido feita pelo modelo principal)
        #    Aqui apenas garantimos que os dicionários estão preenchidos.
        #    Caso contrário, seria necessário criar as variáveis.
        #    Por simplicidade, assumimos que as variáveis já existem nos dicionários.
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # 2. Restrições
        # --------------------------------------------------------------------

        # 2a. Relação entre operação líquida e potências
        for t in range(T):
            for b in battery_list:
                model.add_linear_constraint(
                    BatteryOperation[t, b] == DISCHARGE[t, b] - CHARGE[t, b],
                    name=f"BatteryOperation_def_{t}_{b}"
                )

        # 2b. Condição inicial (t = 0) – usa o parâmetro soc_inicial_list
        for i, b in enumerate(battery_list):
            model.add_linear_constraint(
                SOC[0, b] == soc_inicial_list[i]
                + eff_carga * CHARGE[0, b]
                - DISCHARGE[0, b] / eff_descarga,
                name=f"SOC_init_{b}"
            )

        # 2c. Evolução do SOC para t >= 1
        for t in range(1, T):
            for b in battery_list:
                model.add_linear_constraint(
                    SOC[t, b] == SOC[t-1, b]
                    + eff_carga * CHARGE[t, b]
                    - DISCHARGE[t, b] / eff_descarga,
                    name=f"SOC_evolution_{t}_{b}"
                )

        # 2d. SOC Final do ultimo dia
        for i, b in enumerate(battery_list):
            model.add_linear_constraint(
                SOC[T-1, b] == soc_inicial_list[i],
                name=f"SOC_final_{b}"
            )