#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Função objetivo para minimização do custo total de operação (PyOptInterface).
"""
import pyoptinterface as poi

class ObjectiveFunction:
    """Função objetivo para minimização do custo total de operação."""

    @staticmethod
    def add_objective_pyopt(model_instance):
        """
        Adiciona a função objetivo ao modelo PyOptInterface.

        Parâmetros:
        -----------
        model_instance : TimeCoupledOPFModelPyOptInterface
            Instância do modelo contendo as variáveis e sistema.
        """
        s = model_instance.sistema
        T = model_instance.horizon_time
        expr = 0.0

        # 1. Custo dos geradores térmicos
        if hasattr(s, 'CPG_CONV'):
            for t in range(T):
                for g in range(s.NGER_CONV):
                    expr += s.CPG_CONV[g] * model_instance.PGER[t, g]

        # 2. Penalidade por corte eólico
        if (hasattr(s, 'CPG_CURTAILMENT') and 
            hasattr(model_instance, 'CURTAILMENT') and 
            model_instance.CURTAILMENT):
            for t in range(T):
                for w in range(s.NGER_EOL):
                    expr += s.CPG_CURTAILMENT[w] * model_instance.CURTAILMENT[t, w]

        # 3. Custo do déficit (penalidade alta)
        if hasattr(s, 'CPG_DEFICIT'):
            for t in range(T):
                for b in range(s.NBAR):
                    expr += s.CPG_DEFICIT * model_instance.DEFICIT[t, b]

        # 4. Custo de operação das baterias (desgaste)
        if (hasattr(model_instance, 'CHARGE') and model_instance.CHARGE and 
            hasattr(s, 'BATTERY_COST_CHARGE') and hasattr(s, 'BATTERY_COST_DISCHARGE')):
            for t in range(T):
                for b in model_instance._battery_list:
                    # Se BATTERY_COST_CHARGE for um vetor indexado por barra, pegar o valor correspondente
                    if hasattr(s.BATTERY_COST_CHARGE, '__getitem__') and b < len(s.BATTERY_COST_CHARGE):
                        custo_bateria = s.BATTERY_COST_CHARGE[b]
                    else:
                        custo_bateria = s.BATTERY_COST_CHARGE  # assume escalar
                    expr += custo_bateria * model_instance.CHARGE[t, b]

                for b in model_instance._battery_list:
                    # Se BATTERY_COST_DISCHARGE for um vetor indexado por barra, pegar o valor correspondente
                    if hasattr(s.BATTERY_COST_DISCHARGE, '__getitem__') and b < len(s.BATTERY_COST_DISCHARGE):
                        custo_bateria = s.BATTERY_COST_DISCHARGE[b]
                    else:
                        custo_bateria = s.BATTERY_COST_DISCHARGE  # assume escalar
                    expr += custo_bateria * model_instance.DISCHARGE[t, b]

        # Define a função objetivo no modelo (minimização)
        model_instance.model.set_objective(expr, poi.ObjectiveSense.Minimize)