from pyomo.environ import *
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OPFResult:
    """Resultado do OPF"""
    sucesso: bool
    PG: List[float]
    ANG: List[float]
    FLUXO: List[float]
    DEFICIT: List[float]
    CURTAILMENT: List[float]
    custo_total: float
    deficit_total: float
    curtailment_total: float
    cmo_total: float
    perdas: float
    iteracoes: int = 0
    tempo_execucao: float = 0.0
    mensagem: str = ""
    timestamp: Optional[datetime] = None
    SOC: List[float] = field(default_factory=list)
    BATTERY_OPERATION: List[str] = field(default_factory=list)
    BATTERY_POWER: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Converte resultado para dicionário"""
        return {
            'sucesso': self.sucesso,
            'PG': self.PG,
            'ANG': self.ANG,
            'FLUXO': self.FLUXO,
            'DEFICIT': self.DEFICIT,
            'CURTAILMENT': self.CURTAILMENT,
            'custo_total': self.custo_total,
            'cmo_total': self.cmo_total,
            'perdas': self.perdas,
            'iteracoes': self.iteracoes,
            'tempo_execucao': self.tempo_execucao,
            'mensagem': self.mensagem,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'SOC': self.SOC,
            'BATTERY_OPERATION': self.BATTERY_OPERATION,
            'BATTERY_POWER': self.BATTERY_POWER
        }

@dataclass
class OPF_Model:
    """Modelo físico do sistema (apenas restrições, sem objetivo)"""
    model: ConcreteModel = field(default_factory=ConcreteModel)
    sistema: Optional[object] = None
    considerar_perdas: bool = False
    
    def build(self, sistema, considerar_perdas=False):
        """Constrói modelo físico com todas as restrições"""
        self.sistema = sistema
        self.considerar_perdas = considerar_perdas

        m = self.model
        s = self.sistema
        
        # Limpar modelo se já existir
        m.clear()
        
        # === CONJUNTOS ===
        m.BUSES = Set(initialize=range(s.NBAR))
        m.GENERATORS = Set(initialize=range(s.NGER))
        m.LINES = Set(initialize=range(s.NLIN))
        m.GWD_GENERATORS = Set(initialize=s.BAR_GWD)
        
        # Conjunto de baterias (se houver)
        if hasattr(s, 'BATTERIES') and len(s.BATTERIES) > 0:
            m.BATTERIES = Set(initialize=s.BATTERIES)
        else:
            m.BATTERIES = Set(initialize=[])  # Conjunto vazio
        
        # === VARIÁVEIS ===
        m.PG = Var(m.GENERATORS, within=NonNegativeReals)
        m.ANG = Var(m.BUSES, within=Reals, bounds=(-3.14, 3.14))
        m.FLUXO = Var(m.LINES, within=Reals)
        m.DEFICIT = Var(m.BUSES, within=NonNegativeReals)
        m.PG_WIND_USED = Var(m.GWD_GENERATORS, within=NonNegativeReals)
        m.CURTAILMENT = Var(m.GWD_GENERATORS, within=NonNegativeReals)
        
        # Variáveis de bateria (apenas se houver baterias)
        if len(m.BATTERIES) > 0:
            m.CHARGE = Var(m.BATTERIES, within=NonNegativeReals)  # Carga
            m.DISCHARGE = Var(m.BATTERIES, within=NonNegativeReals)  # Descarga
            m.SOC = Var(m.BATTERIES, within=NonNegativeReals)  # Estado de carga atual
            m.SOC_PREV = Var(m.BATTERIES, within=NonNegativeReals)  # Estado de carga anterior
        
        # Parâmetro para perdas (mutável para iteração)
        if considerar_perdas:
            m.perdas_barra = Param(m.BUSES, mutable=True, default=0.0)
        
        # Fixar ângulo da barra slack
        m.ANG[s.slack_idx].fix(0.0)
        
        # === RESTRIÇÕES ===
        self.add_constraints()
        
        return m
    
    def add_constraints(self):
        """Adiciona todas as restrições ao modelo"""
        # Importar módulos de restrições
        from SOLVER.RES.DC_eletric_constraints import DCElectricConstraints
        from SOLVER.RES.DC_energy_constraints import DCEnergyConstraints
        
        # Adicionar restrições elétricas
        DCElectricConstraints.add_power_balance_constraints(
            self.model, self.sistema, self.considerar_perdas
        )
        DCElectricConstraints.add_line_flow_constraints(
            self.model, self.sistema
        )
        DCElectricConstraints.add_generator_limits_constraints(
            self.model, self.sistema
        )
        
        DCEnergyConstraints.add_wind_generator_constraints(
            self.model, self.sistema
        )
        
        # Adicionar restrições energéticas
        DCEnergyConstraints.add_deficit_constraints(
            self.model, self.sistema
        )
        
        # Adicionar restrições de bateria (se houver)
        if len(self.model.BATTERIES) > 0:
            DCEnergyConstraints.add_battery_constraints(
                self.model, self.sistema
            )


    
    def update_losses(self, perdas_barra: np.ndarray):
        """Atualiza parâmetros de perdas no modelo"""
        if not self.considerar_perdas:
            return
        
        m = self.model
        for i in m.BUSES:
            if i < len(perdas_barra):
                m.perdas_barra[i] = perdas_barra[i]
    
    def calculate_losses(self) -> np.ndarray:
        """Calcula perdas nas linhas baseadas na solução atual"""
        s = self.sistema
        perdas_barra = np.zeros(s.NBAR)
        
        for e in self.model.LINES:
            i = s.line_fr[e]
            j = s.line_to[e]
            
            # Obter fluxo da solução
            fluxo_val = value(self.model.FLUXO[e]) if hasattr(self.model, 'FLUXO') else 0
            
            # Calcular perdas na linha (DC simplificado)
            r = s.r_line[e]
            perdas_linha = r * (fluxo_val ** 2)
            
            # Distribuir igualmente entre barras
            perdas_barra[i] += perdas_linha / 2
            perdas_barra[j] += perdas_linha / 2
        
        return perdas_barra
    
    def extract_results(self, results, iteracoes: int = 1) -> OPFResult:
        """Extrai resultados da solução"""
        m = self.model
        s = self.sistema
        
        # Verificar status
        if results.solver.termination_condition != TerminationCondition.optimal:
            return OPFResult(
                sucesso=False,
                PG=[0.0] * s.NGER,
                ANG=[0.0] * s.NBAR,
                FLUXO=[0.0] * s.NLIN,
                DEFICIT=[0.0] * s.NBAR,
                CURTAILMENT=[0.0] * len(s.BAR_GWD),
                custo_total=0.0,
                defict_total=0.0,
                curtailment_total=0.0,
                cmo_total=0.0,
                perdas=0.0,
                iteracoes=iteracoes,
                mensagem=f"Solver terminou com: {results.solver.termination_condition}",
                timestamp=datetime.now()
            )
        
        # Extrair valores (Geração Bruta e outros)
        PG_raw = [value(m.PG[g]) for g in m.GENERATORS]  # Geração bruta de todos os geradores
        ANG_val = [value(m.ANG[b]) for b in m.BUSES]
        DEFICIT_val = [value(m.DEFICIT[b]) for b in m.BUSES]
        FLUXO_val = [value(m.FLUXO[e]) for e in m.LINES]

        # Calcular o valor líquido para cada gerador
        PG_val = [] 
        curtailment_dict = {g: value(m.CURTAILMENT[g]) for g in m.GWD_GENERATORS}

        for i, g in enumerate(m.GENERATORS):
            if g in m.GWD_GENERATORS:
                # Para geradores GWD: Geração Bruta - Curtailment
                liquid_value = PG_raw[i] - curtailment_dict[g]
                PG_val.append(liquid_value)
            else:
                # Para outros geradores: mantém a geração bruta
                PG_val.append(PG_raw[i])

        # Curtailment (se ainda precisar dos valores individuais e totais)
        curtailment_vals = [value(m.CURTAILMENT[g]) for g in m.GWD_GENERATORS]
        curtailment_total = sum(curtailment_vals)
        
        # Déficit total
        deficit_total = sum(DEFICIT_val)
        
        # Custos
        custo_total = value(m.obj) if hasattr(m, 'obj') else 0.0
        
        # CMO (Custo Marginal de Operação) - dual da restrição de balanço
        cmo_total = 0.0
        if hasattr(m, 'dual') and hasattr(m, 'PowerBalance'):
            try:
                cmo_total = m.dual[m.PowerBalance[s.slack_idx]]
            except:
                cmo_total = 0.0
        
        # Perdas
        perdas_totais = 0.0
        if self.considerar_perdas:
            perdas_totais = sum(value(m.perdas_barra[b]) for b in m.BUSES)
        
        # Adiciona extração de estado da bateria (SOC, operação, potência)
        BATTERY_SOC = []
        BATTERY_OPERATION = []
        BATTERY_POWER = []
        if hasattr(m, 'BUSES') and len(m.BUSES) > 0:
            for b in m.BUSES:
                if b in m.BATTERIES:
                    BATTERY_SOC.append(value(m.SOC[b]) if hasattr(m, 'SOC') else 0.0)
                    # Determina operação: 'charge', 'discharge', 'idle'
                    charge = value(m.CHARGE[b]) if hasattr(m, 'CHARGE') else 0.0
                    discharge = value(m.DISCHARGE[b]) if hasattr(m, 'DISCHARGE') else 0.0
                    if charge > 0.01:
                        BATTERY_OPERATION.append('charge')
                    elif discharge > 0.01:
                        BATTERY_OPERATION.append('discharge')
                    else:
                        BATTERY_OPERATION.append('idle')
                    BATTERY_POWER.append(discharge - charge)
                else:
                    BATTERY_SOC.append(0.0)
                    BATTERY_OPERATION.append('none')
                    BATTERY_POWER.append(0.0)

        return OPFResult(
            sucesso=True,
            PG=PG_val,
            ANG=ANG_val,
            FLUXO=FLUXO_val,
            DEFICIT=DEFICIT_val,
            CURTAILMENT=curtailment_vals,
            custo_total=custo_total,
            deficit_total=deficit_total,
            curtailment_total=curtailment_total,
            cmo_total=cmo_total,
            perdas=perdas_totais,
            iteracoes=iteracoes,
            tempo_execucao=results.solver.time if hasattr(results.solver, 'time') else 0.0,
            mensagem="",
            timestamp=datetime.now(),
            SOC=BATTERY_SOC,
            BATTERY_OPERATION=BATTERY_OPERATION,
            BATTERY_POWER=BATTERY_POWER
        )