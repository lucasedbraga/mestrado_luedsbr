from pyomo.environ import *
import numpy as np

class DCEnergyConstraints:
    """Restrições energéticas"""
    
    @staticmethod
    def add_deficit_constraints(model, sistema):
        """Adiciona restrições para déficit"""
        def deficit_limits_rule(m, i):
            # Déficit limitado pela carga da barra
            carga_barra = sistema.PLOAD[i]
            return m.DEFICIT[i] <= carga_barra * 2.0
        
        model.DeficittLimits = Constraint(model.BUSES, rule=deficit_limits_rule)
    
    @staticmethod
    def add_wind_generator_constraints(model, sistema):
        """Adiciona restrições para geradores eólicos"""
        # Geração utilizada não pode exceder disponível
        def wind_balance_rule(m, g):
            return m.PG_WIND_USED[g] <= m.PG[g]
        
        model.WindBalance = Constraint(model.GWD_GENERATORS, rule=wind_balance_rule)
        
        # Definição de curtailment
        def curtailment_definition_rule(m, g):
            return m.CURTAILMENT[g] == m.PG[g] - m.PG_WIND_USED[g]
        
        model.CurtailmentDefinition = Constraint(model.GWD_GENERATORS, rule=curtailment_definition_rule)
    
    @staticmethod
    def add_battery_constraints(model, sistema):
        """Adiciona restrições para baterias (se aplicável)"""

        # is_charging[b] = 1 → Modo CARGA
        # is_charging[b] = 0 → Modo DESCARGA ou PARADO
        model.is_charging = Var(model.BATTERIES, within=Binary, initialize=0)
        
        # Parâmetro Big-M (suficientemente grande)
        # Deve ser maior que o máximo de carga/descarga possível
        BIG_M = 1000  # Ajuste conforme necessário

        # Verificar se o conjunto BATTERIES existe
        if not hasattr(model, 'BATTERIES') or len(model.BATTERIES) == 0:
            return
        
        # Verificar se variáveis existem
        if not hasattr(model, 'CHARGE') or not hasattr(model, 'DISCHARGE'):
            print("  ⚠️  Variáveis de bateria não definidas - pulando restrições")
            return
        
        def battery_soc_limits_rule(m, b):
            """Limites de estado de carga da bateria"""
            return m.SOC[b] <= sistema.BATTERY_CAPACITY[b]
        
        def battery_soc_min_rule(m, b):
            """Limite mínimo de estado de carga"""
            return m.SOC[b] >= 0.0
        
        def battery_soc_initial_rule(m, b):
            """Define o estado de carga inicial da bateria"""
            return m.SOC_PREV[b] == sistema.BATTERY_INITIAL_SOC[b]
        
        def battery_soc_update_rule(m, b):
            """Atualiza o estado de carga da bateria após carga/descarga"""
            # SOC atual = SOC anterior + eficiência_carga * carga - descarga/eficiência_descarga
            eficiencia_carga = 0.95  # 95% de eficiência na carga
            eficiencia_descarga = 0.95  # 95% de eficiência na descarga
            return m.SOC[b] == m.SOC_PREV[b] + eficiencia_carga * m.CHARGE[b] - m.DISCHARGE[b]*eficiencia_descarga
        
        def battery_charge_limit_rule(m, b):
            """Limite de potência de carga"""
            return m.CHARGE[b] <= sistema.BATTERY_POWER_LIMIT[b] * m.is_charging[b]

        def battery_discharge_limit_rule(m, b):
            """Limite de potência de descarga"""
            return m.DISCHARGE[b] <= sistema.BATTERY_POWER_OUT[b]* (1 - m.is_charging[b])

        # Adicionar constraints
        model.BatterySOCMax = Constraint(model.BATTERIES, rule=battery_soc_limits_rule)
        model.BatterySOCMin = Constraint(model.BATTERIES, rule=battery_soc_min_rule)
        model.BatterySOCInitial = Constraint(model.BATTERIES, rule=battery_soc_initial_rule)
        model.BatterySOCUpdate = Constraint(model.BATTERIES, rule=battery_soc_update_rule)
        model.BatteryChargeLimit = Constraint(model.BATTERIES, rule=battery_charge_limit_rule)
        model.BatteryDischargeLimit = Constraint(model.BATTERIES, rule=battery_discharge_limit_rule)
