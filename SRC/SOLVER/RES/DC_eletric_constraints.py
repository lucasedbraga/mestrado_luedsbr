from pyomo.environ import *
import numpy as np
from typing import Dict, List, Tuple

class DCElectricConstraints:
    """Restrições elétricas DC"""
    
    @staticmethod
    def add_power_balance_constraints(model, sistema, considerar_perdas=False):
        """Adiciona restrições de balanço de potência"""
        def power_balance_rule(m, i):
            # Soma geração na barra
            geracao_total = 0.0
            for g in m.GENERATORS:
                if sistema.BARPG[g] == i:
                    if g in m.GWD_GENERATORS:
                        geracao_total += m.PG_WIND_USED[g]
                    else:
                        geracao_total += m.PG[g]
            
            # Déficit na barra
            deficit_barra = m.DEFICIT[i]
            
            # Fluxos líquidos
            fluxo_liquido = 0.0
            for e in m.LINES:
                if sistema.line_fr[e] == i:
                    fluxo_liquido += m.FLUXO[e]
                elif sistema.line_to[e] == i:
                    fluxo_liquido -= m.FLUXO[e]
            
            # Perdas (se considerar)
            perdas = 0.0
            if considerar_perdas:
                perdas = m.perdas_barra[i]
            
            # Contribuição das baterias (se houver)
            contribuicao_bateria = 0.0
            if hasattr(m, 'BATTERIES') and i in m.BATTERIES:
                # Descarga adiciona potência, carga consome potência
                # Nota: i é o índice da barra, que também é o índice da bateria neste caso
                contribuicao_bateria = m.DISCHARGE[i] - m.CHARGE[i]
            
            return geracao_total + deficit_barra + contribuicao_bateria - fluxo_liquido - perdas == sistema.PLOAD[i]
        
        model.PowerBalance = Constraint(model.BUSES, rule=power_balance_rule)
    
    @staticmethod
    def add_line_flow_constraints(model, sistema):
        """Adiciona restrições de fluxo nas linhas"""
        # Definição de fluxo
        def fluxo_definition_rule(m, e):
            i = sistema.line_fr[e]
            j = sistema.line_to[e]
            return m.FLUXO[e] == (m.ANG[i] - m.ANG[j]) / sistema.x_line[e]
        
        model.FluxoDefinition = Constraint(model.LINES, rule=fluxo_definition_rule)
        
        # Limites de fluxo
        def fluxo_max_pos_rule(m, e):
            return m.FLUXO[e] <= sistema.FLIM[e]
        
        def fluxo_max_neg_rule(m, e):
            return m.FLUXO[e] >= -sistema.FLIM[e]
        
        model.FluxoMaxPos = Constraint(model.LINES, rule=fluxo_max_pos_rule)
        model.FluxoMaxNeg = Constraint(model.LINES, rule=fluxo_max_neg_rule)
    
    @staticmethod
    def add_generator_limits_constraints(model, sistema):
        """Adiciona limites de geração para geradores convencionais"""
        def generation_limits_rule(m, g):
            if g in m.GWD_GENERATORS:
                return m.PG[g] == sistema.PGMAX_EFETIVO[g]
            else:
                # Para outros geradores, limites normais
                return inequality(sistema.PGMIN[g], m.PG[g], sistema.PGMAX[g])
        
        model.GenerationLimits = Constraint(model.GENERATORS, rule=generation_limits_rule)
    
