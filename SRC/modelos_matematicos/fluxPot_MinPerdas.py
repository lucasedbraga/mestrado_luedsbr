import pyomo.environ as pyo
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from scipy.stats import norm

@dataclass
class ResultadoOPF:
    """Estrutura para armazenar resultados do OPF"""
    sucesso: bool
    PG: List[float]
    ANG: List[float] 
    lambda_val: List[float]
    custo_total: float
    fluxos: List[float]
    deficit_total: float
    curtailment_total: float
    curtailment_por_gerador: List[float]
    dual_power_balance: List[float]
    BATin: List[float]
    BATout: List[float]
    BATarm: List[float]

class Sistema:
    """Classe para armazenar dados do sistema de potência"""
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.load_data()
        self.process_base_data()
        self.initialize_system()
        
    def load_data(self):
        """Carrega e processa dados do arquivo JSON"""
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.barras = self.data["BARRAS"]
        self.geradores_data = self.data["GERADORES"]
        self.demandas_data = self.data["DEMANDAS"]
        self.linhas = self.data["LINHAS"]
        self.baterias_data = self.data.get("BATERIAS", [])
        
    def process_base_data(self):
        """Processa dados de base do sistema"""
        self.SB = self.data["S_base"]
        self.PB = self.data["P_base"]
        self.VB = self.data["V_base"]
        self.FB = self.data["f_base"]
        self.ZB = (self.VB ** 2) / self.PB
        self.YB = 1 / self.ZB
        
    def initialize_system(self):
        """Inicializa e processa todos os componentes do sistema"""
        self.convert_to_pu()
        self.create_bus_mapping()
        self.process_lines()
        self.build_susceptance_matrix()
        self.process_generators()
        self.process_loads()
        self.identify_slack_bus()
        self.add_curtailment_and_deficit_generators()
        self.process_batteries()
    
    def convert_to_pu(self):
        """Converte todos os dados para pu"""
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
            b["Q_carga_pu"] = b.get("Q_carga_MVAr", 0.0) / self.SB
        
        # Geradores
        for g in self.geradores_data:
            g["Pmax_pu"] = g["PGERmax_MW"] / self.SB
            g["Pmin_pu"] = g["PGERmin_MW"] / self.SB
            g["Qmax_pu"] = g.get("Qmax_MVAr", 200.0) / self.SB
            g["Qmin_pu"] = g.get("Qmin_MVAr", -200.0) / self.SB
            g["Pg_ref_pu"] = g.get("Pg_ref_MW", 0.0) / self.SB
            g["Qg_ref_pu"] = g.get("Qg_ref_MVAr", 0.0) / self.SB
            g["custo_var_pu"] = g.get("custo_var_USD_MWh", 0.0) * self.SB
        
        # Linhas
        for l in self.linhas:
            l["R_pu"] = l["R"] / self.ZB
            l["X_pu"] = l["X"] / self.ZB
            l["B_pu"] = l.get("Bsh", 0.0) * self.ZB
            l["Fmax_pu"] = l["LIM_Fluxo"] / self.SB
        
        # Baterias
        for bat in self.baterias_data:
            bat["Pmax_carga_pu"] = bat.get("Pmax_carga_MW", 0.0) / self.SB
            bat["Pmax_descarga_pu"] = bat.get("Pmax_descarga_MW", 0.0) / self.SB
            if "capacidade_armazenamento_MWh" in bat:
                bat["capacidade_armazenamento_pu"] = bat["capacidade_armazenamento_MWh"] / self.SB
            else:
                bat["capacidade_armazenamento_pu"] = bat["Pmax_carga_MW"] * 4 / self.SB
    
    def create_bus_mapping(self):
        """Cria mapeamento de IDs de barras para índices"""
        self.bus_ids = [b["ID_Barra"] for b in self.barras]
        self.idx_map = {id: i for i, id in enumerate(self.bus_ids)}
        self.NBAR = len(self.bus_ids)
        self.NLIN = len(self.linhas)
    
    def process_lines(self):
        """Processa dados das linhas de transmissão"""
        self.line_fr = []
        self.line_to = []
        self.r_line = np.zeros(self.NLIN)
        self.x_line = np.zeros(self.NLIN)
        self.y_line = np.zeros(self.NLIN)
        self.g_line = np.zeros(self.NLIN)
        self.FLIM = np.zeros(self.NLIN)
        
        for e, ln in enumerate(self.linhas):
            fr = ln["ID_Barra_Origem"]
            to = ln["ID_Barra_Destino"]
            self.line_fr.append(self.idx_map[fr])
            self.line_to.append(self.idx_map[to])
            
            r = ln["R_pu"]
            x = ln["X_pu"]
            
            self.r_line[e] = r
            self.x_line[e] = x
            denom = r**2 + x**2
            self.g_line[e] = r / denom if denom > 0 else 0.0
            self.y_line[e] = 1.0 / x if abs(x) > 0 else 0.0
            self.FLIM[e] = ln["Fmax_pu"]
    
    def build_susceptance_matrix(self):
        """Constrói a matriz de susceptância Bbus"""
        self.Bbus = np.zeros((self.NBAR, self.NBAR))
        
        for e in range(self.NLIN):
            i = self.line_fr[e]
            j = self.line_to[e]
            y = self.y_line[e]
            
            self.Bbus[i, i] += y
            self.Bbus[j, j] += y
            self.Bbus[i, j] -= y
            self.Bbus[j, i] -= y
    
    def process_generators(self):
        """Processa dados dos geradores"""
        self.NGER_ORIGINAL = len(self.geradores_data)
        self.BARPG_ORIGINAL = []
        self.BAR_GWD = []
        self.PGMIN_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMIN_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.CPG_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        
        # Identificar tipos de geradores
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            self.BARPG_ORIGINAL.append(self.idx_map[id_barra])
            if g["Tipo"] == "GWD":
                self.BAR_GWD.append(self.idx_map[id_barra])
        
        # Processar todos os geradores
        for i, g in enumerate(self.geradores_data):
            tipo_ger = g["Tipo"]
            
            self.PGMIN_ORIGINAL[i] = g["Pmin_pu"]
            self.PGMAX_ORIGINAL[i] = g["Pmax_pu"]
            
            if tipo_ger in ["UTE", "UTH", "GWD"]:
                self.CPG_ORIGINAL[i] = g.get("custo_var_USD_MWh", 0.0) * self.SB
            else:
                self.CPG_ORIGINAL[i] = 0.0
            
            # Valores iniciais
            self.PGMAX_EFETIVO[i] = self.PGMAX_ORIGINAL[i]
            self.PGMIN_EFETIVO[i] = self.PGMIN_ORIGINAL[i]
    
    def process_loads(self):
        """Processa dados das cargas"""
        self.PLOAD = np.zeros(self.NBAR)
        for d in self.demandas_data:
            id_barra = d["ID_Barra"]
            idx = self.idx_map[id_barra]
            potencia_demanda = d.get("PLOAD", 0.0) / self.SB
            self.PLOAD[idx] += potencia_demanda

    def process_batteries(self):
        """Processa dados das baterias do JSON"""
        self.BARRAS_COM_BATERIA = []
        self.BATmax_in = np.zeros(self.NBAR)
        self.BATmax_out = np.zeros(self.NBAR)
        self.BATcusto_in = np.zeros(self.NBAR)
        self.BATcusto_out = np.zeros(self.NBAR)
        self.BATcapacidade = np.zeros(self.NBAR)
        self.BATarm_inicial = np.zeros(self.NBAR)
        
        for bat in self.baterias_data:
            id_barra = bat["ID_Barra"]
            if id_barra not in self.idx_map:
                continue
                
            idx = self.idx_map[id_barra]
            self.BARRAS_COM_BATERIA.append(idx)
            
            self.BATmax_in[idx] = bat["Pmax_carga_pu"]
            self.BATmax_out[idx] = bat["Pmax_descarga_pu"]
            self.BATcapacidade[idx] = bat["capacidade_armazenamento_pu"]
            self.BATarm_inicial[idx] = 0.5 * self.BATcapacidade[idx]
            self.BATcusto_in[idx] = bat.get("custo_carga_USD_MWh", 5.0)
            self.BATcusto_out[idx] = bat.get("custo_descarga_USD_MWh", 5.0)
    
    def identify_slack_bus(self):
        """Identifica a barra slack"""
        slack_list = [b for b in self.barras if b["tipo"] == "Slack"]
        if len(slack_list) != 1:
            raise ValueError("Deve haver exatamente 1 barra Slack no JSON")
        slack_id = slack_list[0]["ID_Barra"]
        self.slack_idx = self.idx_map[slack_id]
    
    def add_curtailment_and_deficit_generators(self):
        """Adiciona geradores de curtailment e déficit"""
        # Identificar barras PQ sem gerador
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        barras_com_gerador = set(self.BARPG_ORIGINAL)
        self.barras_PQ_sem_gerador = [b for b in barras_PQ 
                                    if self.idx_map[b["ID_Barra"]] not in barras_com_gerador]
        
        # Curtailment
        self.NGER_CURTAILMENT = len(self.BAR_GWD)
        custo_maximo_existente = np.max(self.CPG_ORIGINAL) if len(self.CPG_ORIGINAL) > 0 else 1000.0
        self.CUSTO_CURTAILMENT = 10.0 * custo_maximo_existente
        self.CUSTO_DEFICIT = 100.0 * custo_maximo_existente
        
        self.BARPG_CURTAILMENT = []
        self.PGMIN_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        self.PGMAX_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        self.CPG_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        
        for i, barra_idx in enumerate(self.BAR_GWD):
            self.BARPG_CURTAILMENT.append(barra_idx)
            self.PGMIN_CURTAILMENT[i] = 0.0
            
            gwd_idx = next((j for j, g in enumerate(self.geradores_data) 
                          if self.idx_map[g["ID_Barra"]] == barra_idx and g["Tipo"] == "GWD"), None)
            
            if gwd_idx is not None:
                self.PGMAX_CURTAILMENT[i] = self.PGMAX_EFETIVO[gwd_idx]
                self.CPG_CURTAILMENT[i] = self.geradores_data[gwd_idx].get(
                    "custo_curtailment_USD_MWh", self.CUSTO_CURTAILMENT)
        
        # Déficit
        self.NGER_DEFICIT = len(self.barras_PQ_sem_gerador)
        self.BARPG_DEFICIT = []
        self.PGMIN_DEFICIT = np.zeros(self.NGER_DEFICIT)
        self.PGMAX_DEFICIT = np.zeros(self.NGER_DEFICIT)
        self.CPG_DEFICIT = np.zeros(self.NGER_DEFICIT)
        
        for i, b in enumerate(self.barras_PQ_sem_gerador):
            idx = self.idx_map[b["ID_Barra"]]
            self.BARPG_DEFICIT.append(idx)
            self.PGMIN_DEFICIT[i] = 0.0
            self.PGMAX_DEFICIT[i] = self.PLOAD[idx] * 2 if self.PLOAD[idx] > 0 else 1.0
            self.CPG_DEFICIT[i] = self.CUSTO_DEFICIT
        
        # Combinar todos os geradores
        self.NGER = self.NGER_ORIGINAL + self.NGER_CURTAILMENT + self.NGER_DEFICIT
        self.BARPG = self.BARPG_ORIGINAL + self.BARPG_CURTAILMENT + self.BARPG_DEFICIT
        self.PGMIN = np.concatenate([self.PGMIN_EFETIVO, self.PGMIN_CURTAILMENT, self.PGMIN_DEFICIT])
        self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, self.PGMAX_CURTAILMENT, self.PGMAX_DEFICIT])
        self.CPG = np.concatenate([self.CPG_ORIGINAL, self.CPG_CURTAILMENT, self.CPG_DEFICIT])

def resolver_opf(sistema, Bbus, linhas_ativas=None, nome_modelo="opf"):
    """OPF DC completo com curtailment, déficit, baterias"""
    try:
        model = pyo.ConcreteModel()
        
        # Conjuntos
        model.GER = pyo.Set(initialize=range(sistema.NGER))
        model.BAR = pyo.Set(initialize=range(sistema.NBAR))
        
        # LINHAS ATIVAS
        if linhas_ativas is None:
            indices_linhas_ativas = list(range(sistema.NLIN))
        else:
            indices_linhas_ativas = [i for i, ativa in enumerate(linhas_ativas) if ativa]
        
        model.LIN_ATV = pyo.Set(initialize=indices_linhas_ativas)
        
        # Geradores GWD
        gwd_indices = [g for g in range(sistema.NGER_ORIGINAL) 
                      if sistema.geradores_data[g]["Tipo"] == "GWD"]
        model.GWD = pyo.Set(initialize=gwd_indices)
        
        # Baterias
        model.BAT_BAR = pyo.Set(initialize=sistema.BARRAS_COM_BATERIA)
        
        # Variáveis
        model.PG = pyo.Var(model.GER, within=pyo.NonNegativeReals)
        model.ANG = pyo.Var(model.BAR, within=pyo.Reals, bounds=(-3.14, 3.14))
        model.BATin = pyo.Var(model.BAR, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.BATout = pyo.Var(model.BAR, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.BATarm = pyo.Var(model.BAT_BAR, within=pyo.NonNegativeReals)
        model.DEFICIT = pyo.Var(model.BAR, within=pyo.NonNegativeReals)
        model.CURTAILMENT = pyo.Var(model.GWD, within=pyo.NonNegativeReals)
        
        # Fixar ângulo da barra slack
        model.ANG[sistema.slack_idx].fix(0.0)
        
        # LIMITES DE GERAÇÃO
        def pg_limits_rule(m, g):
            return (sistema.PGMIN[g], m.PG[g], sistema.PGMAX[g])
        model.pg_limits = pyo.Constraint(model.GER, rule=pg_limits_rule)
        
        # LIMITES DE CURTAILMENT
        def curtailment_limits_rule(m, g):
            return m.CURTAILMENT[g] <= sistema.PGMAX_EFETIVO[g]
        model.curtailment_limits = pyo.Constraint(model.GWD, rule=curtailment_limits_rule)
        
        # LIMITES DE BATERIA
        def bat_in_limits_rule(m, i):
            if i in m.BAT_BAR:
                return m.BATin[i] <= sistema.BATmax_in[i]
            else:
                return m.BATin[i] == 0
        model.bat_in_limits = pyo.Constraint(model.BAR, rule=bat_in_limits_rule)
        
        def bat_out_limits_rule(m, i):
            if i in m.BAT_BAR:
                return m.BATout[i] <= sistema.BATmax_out[i]
            else:
                return m.BATout[i] == 0
        model.bat_out_limits = pyo.Constraint(model.BAR, rule=bat_out_limits_rule)
        
        def bat_arm_limits_rule(m, i):
            return (0, m.BATarm[i], sistema.BATcapacidade[i])
        model.bat_arm_limits = pyo.Constraint(model.BAT_BAR, rule=bat_arm_limits_rule)
        
        # BALANÇO ENERGÉTICO DA BATERIA
        eficiencia_carga = 1
        eficiencia_descarga = 1
        
        def bat_energy_balance_rule(m, i):
            return m.BATarm[i] == (sistema.BATarm_inicial[i] + 
                                 eficiencia_carga * m.BATin[i] - 
                                 eficiencia_descarga * m.BATout[i])
        model.bat_energy_balance = pyo.Constraint(model.BAT_BAR, rule=bat_energy_balance_rule)
        
        # PENALIDADES
        custo_maximo = max(sistema.CPG) if len(sistema.CPG) > 0 else 1000.0
        PENALIDADE_CARGA = 0.1 * custo_maximo
        PENALIDADE_DESCARGA = 1.1 * custo_maximo
        PENALIDADE_CURTAILMENT = 10 * custo_maximo
        PENALIDADE_DEFICIT = 100 * custo_maximo
        
        # BALANÇO DE POTÊNCIA
        def power_balance_rule(m, i):
            geracao_total = 0.0
            for g in m.GER:
                if sistema.BARPG[g] == i:
                    if g in m.GWD:
                        geracao_total += m.PG[g] - m.CURTAILMENT[g]
                    else:
                        geracao_total += m.PG[g]
            
            contribuicao_bateria = m.BATout[i] - m.BATin[i]
            deficit_barra = m.DEFICIT[i]
            fluxo_saindo = 0.0
            for j in m.BAR:
                fluxo_saindo += Bbus[i, j] * m.ANG[j]
            
            return geracao_total + contribuicao_bateria + deficit_barra - fluxo_saindo == sistema.PLOAD[i]

        model.power_balance = pyo.Constraint(model.BAR, rule=power_balance_rule)
        
        # LIMITES DE FLUXO
        def line_flow_limits_rule(m, e):
            i = sistema.line_fr[e]
            j = sistema.line_to[e]
            fluxo = sistema.y_line[e] * (m.ANG[i] - m.ANG[j])
            return (-sistema.FLIM[e], fluxo, sistema.FLIM[e])
        
        model.line_flow_limits = pyo.Constraint(model.LIN_ATV, rule=line_flow_limits_rule)
        
        # FUNÇÃO OBJETIVO
        def objective_rule(m):
            custo_geracao = sum(sistema.CPG[g] * m.PG[g] for g in m.GER)
            custo_curtailment = PENALIDADE_CURTAILMENT * sum(m.CURTAILMENT[g] for g in m.GWD)
            custo_deficit = PENALIDADE_DEFICIT * sum(m.DEFICIT[b] for b in m.BAR)
            custo_bateria_in = sum(PENALIDADE_CARGA*sistema.BATcusto_in[i] * m.BATin[i] for i in m.BAT_BAR)
            custo_bateria_out = sum(PENALIDADE_DESCARGA*sistema.BATcusto_out[i] * m.BATout[i] for i in m.BAT_BAR)
            
            return (custo_geracao + custo_curtailment + custo_deficit + 
                   custo_bateria_in + custo_bateria_out)
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # RESOLVER
        solver = pyo.SolverFactory('glpk')
        results = solver.solve(model, tee=False)
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Extrair resultados
            PG_val = [pyo.value(model.PG[g]) for g in model.GER]
            ANG_val = [pyo.value(model.ANG[b]) for b in model.BAR]
            DEFICIT_val = [pyo.value(model.DEFICIT[b]) for b in model.BAR]
            BATin_val = [pyo.value(model.BATin[b]) for b in model.BAR]
            BATout_val = [pyo.value(model.BATout[b]) for b in model.BAR]
            BATarm_val = [0.0] * sistema.NBAR
            for i in model.BAT_BAR:
                BATarm_val[i] = pyo.value(model.BATarm[i])
            
            # Curtailment
            curtailment_por_gerador = [0.0] * sistema.NGER
            curtailment_total = 0.0
            for g in model.GWD:
                curtailment_val = pyo.value(model.CURTAILMENT[g])
                curtailment_por_gerador[g] = curtailment_val
                curtailment_total += curtailment_val
            
            deficit_total = sum(DEFICIT_val)
            
            # Custo
            custo_geracao_real = sum(sistema.CPG[g] * PG_val[g] for g in range(sistema.NGER))

            custo_bateria_real = (sum(sistema.BATcusto_in[i] * BATin_val[i] for i in sistema.BARRAS_COM_BATERIA) +
                                sum(sistema.BATcusto_out[i] * BATout_val[i] for i in sistema.BARRAS_COM_BATERIA))
            
            custo_total = (custo_geracao_real + 
                         (PENALIDADE_CURTAILMENT * curtailment_total) + 
                         (PENALIDADE_DEFICIT * deficit_total) +
                         custo_bateria_real)
            
            # Fluxos
            fluxos_val = [0.0] * sistema.NLIN
            for e in indices_linhas_ativas:
                i, j = sistema.line_fr[e], sistema.line_to[e]
                fluxos_val[e] = sistema.y_line[e] * (ANG_val[i] - ANG_val[j])
            
            lambda_val = [0.0] * sistema.NBAR
            
            return ResultadoOPF(True, PG_val, ANG_val, lambda_val, custo_total, 
                              fluxos_val, deficit_total, curtailment_total,
                              curtailment_por_gerador, [0.0] * sistema.NBAR,
                              BATin_val, BATout_val, BATarm_val)
        else:
            print(f"OPF nao convergiu: {results.solver.termination_condition}")
            return ResultadoOPF(False, 
                               [0.0] * sistema.NGER, 
                               [0.0] * sistema.NBAR, 
                               [0.0] * sistema.NBAR, 
                               0.0, 
                               [0.0] * sistema.NLIN,
                               0.0, 0.0, [0.0] * sistema.NGER, [0.0] * sistema.NBAR,
                               [0.0] * sistema.NBAR, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR)
    
    except Exception as e:
        print(f"Erro no OPF: {e}")
        return ResultadoOPF(False, 
                           [0.0] * sistema.NGER, 
                           [0.0] * sistema.NBAR, 
                           [0.0] * sistema.NBAR, 
                           0.0, 
                           [0.0] * sistema.NLIN,
                           0.0, 0.0, [0.0] * sistema.NGER, [0.0] * sistema.NBAR,
                           [0.0] * sistema.NBAR, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR)

# ==============================================================================
# SIMULAÇÃO DESACOPLADA COM ESTADO DAS BATERIAS
# ==============================================================================

class GerenciadorBaterias:
    """Gerencia o estado das baterias entre simulações desacopladas"""
    
    def __init__(self):
        self.estado_baterias = None
        self.historico_estados = []
        
    def inicializar(self, sistema):
        """Inicializa com o estado do sistema"""
        self.estado_baterias = sistema.BATarm_inicial.copy()
        self.historico_estados = [self.estado_baterias.copy()]
        
    def atualizar_estado_baterias(self, novo_estado):
        """Atualiza o estado das baterias após uma simulação"""
        self.estado_baterias = novo_estado.copy()
        self.historico_estados.append(self.estado_baterias.copy())
        
    def preparar_sistema(self, sistema):
        """Prepara o sistema com o estado atual das baterias"""
        for i in sistema.BARRAS_COM_BATERIA:
            if i < len(self.estado_baterias):
                sistema.BATarm_inicial[i] = self.estado_baterias[i]

def criar_load_shape_24h():
    """Cria um perfil típico de carga para 24 horas"""
    # Perfil típico de demanda (valores normalizados)
    return [
        0.6, 0.5, 0.5, 0.5, 0.5, 0.6,  # 00-05h: Madrugada
        0.7, 0.8, 0.9, 0.9, 0.8, 0.8,  # 06-11h: Manhã
        0.9, 1.0, 1.0, 1.0, 1.0, 0.9,  # 12-17h: Tarde
        1.1, 2.5, 1.1, 1.0, 0.8, 0.7   # 18-23h: Noite
    ]

def simular_despacho_1hora(json_file_path, fator_demanda=1.0, disponibilidade_geradores=None, 
                          gerenciador_baterias=None, hora=0):
    """
    Simula despacho econômico para 1 hora única
    
    Args:
        json_file_path: Caminho do arquivo JSON do sistema
        fator_demanda: Fator para ajustar a demanda
        disponibilidade_geradores: Dict com disponibilidade por gerador
        gerenciador_baterias: Gerenciador para estado das baterias
        hora: Hora do dia (para relatório)
    """
    
    # Carregar sistema
    sistema = Sistema(json_file_path)
    
    # Aplicar estado das baterias se fornecido
    if gerenciador_baterias and gerenciador_baterias.estado_baterias is not None:
        gerenciador_baterias.preparar_sistema(sistema)
    
    # Ajustar demanda se especificado
    if fator_demanda != 1.0:
        sistema.PLOAD = sistema.PLOAD * fator_demanda
    
    # Ajustar disponibilidade dos geradores se especificado
    if disponibilidade_geradores:
        for i, gerador in enumerate(sistema.geradores_data):
            id_gerador = gerador['ID_Gerador']
            if id_gerador in disponibilidade_geradores:
                disponibilidade = disponibilidade_geradores[id_gerador]
                sistema.PGMAX_EFETIVO[i] = min(disponibilidade * sistema.PGMAX_ORIGINAL[i], 
                                             sistema.PGMAX_ORIGINAL[i])
                sistema.PGMAX_EFETIVO[i] = max(sistema.PGMAX_EFETIVO[i], sistema.PGMIN_ORIGINAL[i])
                sistema.PGMAX[i] = sistema.PGMAX_EFETIVO[i]
    
    # Resolver OPF para 1 hora
    resultado = resolver_opf(sistema, sistema.Bbus)
    
    # RELATÓRIO INDIVIDUAL
    print(f"\nHORA {hora:02d}:00 - Fator Demanda: {fator_demanda}")
    print("-" * 50)
    
    if resultado.sucesso:
        print(f"✓ Custo: {resultado.custo_total:.2f} USD")
        print(f"  Geração: {sum(resultado.PG):.4f} pu, Demanda: {sum(sistema.PLOAD):.4f} pu")
        print(f"  Curtailment: {resultado.curtailment_total:.4f} pu, Déficit: {resultado.deficit_total:.4f} pu")
        print(f"  Baterias: Carga={sum(resultado.BATin):.4f}, Descarga={sum(resultado.BATout):.4f}")
        
        # Mostrar estado das baterias
        if sistema.BARRAS_COM_BATERIA:
            print("  Estado baterias:")
            for idx in sistema.BARRAS_COM_BATERIA:
                estado_anterior = sistema.BATarm_inicial[idx]
                estado_atual = resultado.BATarm[idx]
                variacao = estado_atual - estado_anterior
                print(f"    Barra {idx}: {estado_anterior:.4f} → {estado_atual:.4f} ({variacao:+.4f})")
        
        # Atualizar estado das baterias se gerenciador fornecido
        if gerenciador_baterias and resultado.sucesso:
            gerenciador_baterias.atualizar_estado_baterias(resultado.BATarm)
            
    else:
        print(f"✗ Falha na convergência")
    
    # Retornar resultados detalhados
    return {
        'hora': hora,
        'fator_demanda': fator_demanda,
        'sucesso': resultado.sucesso,
        'custo': resultado.custo_total if resultado.sucesso else 0,
        'geracao_total': sum(resultado.PG) if resultado.sucesso else 0,
        'demanda_total': sum(sistema.PLOAD),
        'curtailment': resultado.curtailment_total if resultado.sucesso else 0,
        'deficit': resultado.deficit_total if resultado.sucesso else sum(sistema.PLOAD),
        'carga_bateria': sum(resultado.BATin) if resultado.sucesso else 0,
        'descarga_bateria': sum(resultado.BATout) if resultado.sucesso else 0,
        'estado_baterias_final': resultado.BATarm.copy() if resultado.sucesso else sistema.BATarm_inicial.copy(),
        'sistema': sistema,
        'resultado': resultado
    }

def trabalho_08():
    """
    TRABALHO 08 - Simulação desacoplada com múltiplos fatores de demanda
    Mantém o estado das baterias entre simulações
    """
    print("=" * 70)
    print("TRABALHO 08 - DESPACHO ECONÔMICO DESACOPLADO")
    print("=" * 70)
    
    # Criar gerenciador de baterias
    gerenciador = GerenciadorBaterias()
    
    # Inicializar gerenciador com sistema
    sistema_inicial = Sistema("DATA/input/B6L8_carregado.json")
    gerenciador.inicializar(sistema_inicial)
    
    # Load shape para 24 horas
    load_shape = criar_load_shape_24h()
    
    # Lista para armazenar todos os resultados
    todos_resultados = []
    
    print(f"\nSIMULAÇÃO DE 24 HORAS COM PERFIL DE CARGA")
    print("=" * 50)
    
    # Simular cada hora
    for hora, fator in enumerate(load_shape):
        resultado_hora = simular_despacho_1hora(
            "DATA/input/B6L8_carregado.json",
            fator_demanda=fator,
            gerenciador_baterias=gerenciador,
            hora=hora
        )
        todos_resultados.append(resultado_hora)
    
    # RELATÓRIO FINAL
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL - 24 HORAS")
    print("=" * 70)
    
    # Criar DataFrame com resultados
    df = pd.DataFrame(todos_resultados)
    
    # Estatísticas gerais
    print(f"\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   Total de horas simuladas: {len(df)}")
    print(f"   Horas com sucesso: {sum(df['sucesso'])}")
    print(f"   Custo total acumulado: {df['custo'].sum():.2f} USD")
    print(f"   Geração total: {df['geracao_total'].sum():.4f} pu")
    print(f"   Demanda total: {df['demanda_total'].sum():.4f} pu")
    print(f"   Curtailment total: {df['curtailment'].sum():.4f} pu")
    print(f"   Déficit total: {df['deficit'].sum():.4f} pu")
    print(f"   Operação baterias - Carga: {df['carga_bateria'].sum():.4f} pu")
    print(f"   Operação baterias - Descarga: {df['descarga_bateria'].sum():.4f} pu")
    
    # Evolução das baterias
    print(f"\n🔋 EVOLUÇÃO DAS BATERIAS:")
    estado_inicial = sistema_inicial.BATarm_inicial
    estado_final = todos_resultados[-1]['estado_baterias_final']
    
    for idx in sistema_inicial.BARRAS_COM_BATERIA:
        if idx < len(estado_inicial) and idx < len(estado_final):
            variacao_total = estado_final[idx] - estado_inicial[idx]
            print(f"   Barra {idx}: {estado_inicial[idx]:.4f} → {estado_final[idx]:.4f} ({variacao_total:+.4f})")
    
    # CORREÇÃO: Converter hora para inteiro antes de formatar
    print(f"\n💰 CUSTOS POR HORA (TOP 5 MAIS CAROS):")
    custos_por_hora = df[['hora', 'custo', 'fator_demanda']].copy()
    custos_por_hora = custos_por_hora.sort_values('custo', ascending=False).head(5)
    for _, row in custos_por_hora.iterrows():
        hora_int = int(row['hora'])  # Converter para inteiro
        print(f"   Hora {hora_int:02d}:00 - Custo: {row['custo']:.2f} USD (Demanda: {row['fator_demanda']:.2f})")
    
    # Resumo das baterias
    print(f"\n📈 RESUMO DA OPERAÇÃO DAS BATERIAS:")
    carga_total = df['carga_bateria'].sum()
    descarga_total = df['descarga_bateria'].sum()
    saldo_energetico = descarga_total - carga_total
    print(f"   Energia total carregada: {carga_total:.4f} pu")
    print(f"   Energia total descarregada: {descarga_total:.4f} pu")
    print(f"   Saldo energético: {saldo_energetico:+.4f} pu")
    
    return todos_resultados, gerenciador

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    # Executar trabalho 08
    resultados, gerenciador = trabalho_08()