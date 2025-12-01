import pyomo.environ as pyo
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
import json

@dataclass
class ResultadoOPF:
    """Estrutura para armazenar resultados do OPF"""
    sucesso: bool
    PG: List[float]
    ANG: List[float] 
    custo_total: float
    fluxos: List[float]
    deficit_total: float
    curtailment_total: float
    BATin: List[float]
    BATout: List[float]
    BATarm: List[float]

class Sistema:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.load_data()
        self.process_base_data()
        self.initialize_system()
        
    def load_data(self):
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.barras = self.data["BARRAS"]
        self.geradores_data = self.data["GERADORES"]
        self.demandas_data = self.data["DEMANDAS"]
        self.linhas = self.data["LINHAS"]
        self.baterias_data = self.data.get("BATERIAS", [])
        
    def process_base_data(self):
        self.SB = self.data["S_base"]
        self.PB = self.data["P_base"]
        self.VB = self.data["V_base"]
        self.ZB = (self.VB ** 2) / self.PB
        
    def initialize_system(self):
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
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
        
        # Geradores
        for g in self.geradores_data:
            g["Pmax_pu"] = g["PGERmax_MW"] / self.SB
            g["Pmin_pu"] = g["PGERmin_MW"] / self.SB
            g["custo_var_pu"] = g.get("custo_var_USD_MWh", 0.0) * self.SB
        
        # Linhas
        for l in self.linhas:
            l["X_pu"] = l["X"] / self.ZB
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
        self.bus_ids = [b["ID_Barra"] for b in self.barras]
        self.idx_map = {id: i for i, id in enumerate(self.bus_ids)}
        self.NBAR = len(self.bus_ids)
        self.NLIN = len(self.linhas)
    
    def process_lines(self):
        self.line_fr = []
        self.line_to = []
        self.x_line = np.zeros(self.NLIN)
        self.y_line = np.zeros(self.NLIN)
        self.FLIM = np.zeros(self.NLIN)
        
        for e, ln in enumerate(self.linhas):
            fr = ln["ID_Barra_Origem"]
            to = ln["ID_Barra_Destino"]
            self.line_fr.append(self.idx_map[fr])
            self.line_to.append(self.idx_map[to])
            
            x = ln["X_pu"]
            self.x_line[e] = x
            self.y_line[e] = 1.0 / x if abs(x) > 0 else 0.0
            self.FLIM[e] = ln["Fmax_pu"]
    
    def build_susceptance_matrix(self):
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
        self.NGER_ORIGINAL = len(self.geradores_data)
        self.BARPG_ORIGINAL = []
        self.BAR_GWD = []
        self.PGMIN_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMIN_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.CPG_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            self.BARPG_ORIGINAL.append(self.idx_map[id_barra])
            if g["Tipo"] == "GWD":
                self.BAR_GWD.append(self.idx_map[id_barra])
        
        for i, g in enumerate(self.geradores_data):
            self.PGMIN_ORIGINAL[i] = g["Pmin_pu"]
            self.PGMAX_ORIGINAL[i] = g["Pmax_pu"]
            self.CPG_ORIGINAL[i] = g.get("custo_var_USD_MWh", 0.0) * self.SB
            self.PGMAX_EFETIVO[i] = self.PGMAX_ORIGINAL[i]
            self.PGMIN_EFETIVO[i] = self.PGMIN_ORIGINAL[i]
    
    def process_loads(self):
        self.PLOAD = np.zeros(self.NBAR)
        for d in self.demandas_data:
            id_barra = d["ID_Barra"]
            idx = self.idx_map[id_barra]
            self.PLOAD[idx] += d.get("PLOAD", 0.0) / self.SB

    def process_batteries(self):
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
            self.BATarm_inicial[idx] = 0.5 * self.BATcapacidade[idx]  # SOC 50%
            
            # CORREÇÃO: Custos POSITIVOS
            self.BATcusto_in[idx] = bat.get("custo_carga_USD_MWh", 5.0)    # POSITIVO
            self.BATcusto_out[idx] = bat.get("custo_descarga_USD_MWh", 10.0) # POSITIVO
    
    def identify_slack_bus(self):
        slack_list = [b for b in self.barras if b["tipo"] == "Slack"]
        if len(slack_list) != 1:
            raise ValueError("Deve haver exatamente 1 barra Slack")
        slack_id = slack_list[0]["ID_Barra"]
        self.slack_idx = self.idx_map[slack_id]
    
    def add_curtailment_and_deficit_generators(self):
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        barras_com_gerador = set(self.BARPG_ORIGINAL)
        self.barras_PQ_sem_gerador = [b for b in barras_PQ 
                                    if self.idx_map[b["ID_Barra"]] not in barras_com_gerador]
        
        # Curtailment
        self.NGER_CURTAILMENT = len(self.BAR_GWD)
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
                self.CPG_CURTAILMENT[i] = 100  # Custo fixo
        
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
            self.CPG_DEFICIT[i] = 1000  # Custo fixo
        
        # Combinar todos os geradores
        self.NGER = self.NGER_ORIGINAL + self.NGER_CURTAILMENT + self.NGER_DEFICIT
        self.BARPG = self.BARPG_ORIGINAL + self.BARPG_CURTAILMENT + self.BARPG_DEFICIT
        self.PGMIN = np.concatenate([self.PGMIN_EFETIVO, self.PGMIN_CURTAILMENT, self.PGMIN_DEFICIT])
        self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, self.PGMAX_CURTAILMENT, self.PGMAX_DEFICIT])
        self.CPG = np.concatenate([self.CPG_ORIGINAL, self.CPG_CURTAILMENT, self.CPG_DEFICIT])

def SolveOPF(sistema, Bbus):
    """OPF DC simplificado"""
    try:
        model = pyo.ConcreteModel()
        
        # Conjuntos
        model.GER = pyo.Set(initialize=range(sistema.NGER))
        model.BAR = pyo.Set(initialize=range(sistema.NBAR))
        
        # Geradores GWD
        gwd_indices = [g for g in range(sistema.NGER_ORIGINAL) 
                      if sistema.geradores_data[g]["Tipo"] == "GWD"]
        model.GWD = pyo.Set(initialize=gwd_indices)
        
        # Baterias
        model.BAT_BAR = pyo.Set(initialize=sistema.BARRAS_COM_BATERIA)
        
        # Variáveis
        model.PG = pyo.Var(model.GER, within=pyo.NonNegativeReals)
        model.ANG = pyo.Var(model.BAR, within=pyo.Reals, bounds=(-3.14, 3.14))
        model.BATin = pyo.Var(model.BAR, within=pyo.NonNegativeReals)
        model.BATout = pyo.Var(model.BAR, within=pyo.NonNegativeReals)
        model.BATarm = pyo.Var(model.BAT_BAR, within=pyo.NonNegativeReals)
        model.DEFICIT = pyo.Var(model.BAR, within=pyo.NonNegativeReals)
        model.CURTAILMENT = pyo.Var(model.GWD, within=pyo.NonNegativeReals)
        
        # Fixar ângulo da barra slack
        model.ANG[sistema.slack_idx].fix(0.0)
        
        # LIMITES DE GERAÇÃO
        def C_LimiteGER(m, g):
            return (sistema.PGMIN[g], m.PG[g], sistema.PGMAX[g])
        model.C_LimiteGER = pyo.Constraint(model.GER, rule=C_LimiteGER)
        
        # LIMITES DE CURTAILMENT
        def C_LimiteCURT(m, g):
            return m.CURTAILMENT[g] <= sistema.PGMAX_EFETIVO[g]
        model.C_LimiteCURT = pyo.Constraint(model.GWD, rule=C_LimiteCURT)
        
        # LIMITES DE BATERIA
        def C_LimiteCarregamentoBateria(m, i):
            if i in m.BAT_BAR:
                return m.BATin[i] <= sistema.BATmax_in[i]
            else:
                return m.BATin[i] == 0
        model.C_LimiteCarregamentoBateria = pyo.Constraint(model.BAR, rule=C_LimiteCarregamentoBateria)
        
        def C_LimiteDescargaBateria(m, i):
            if i in m.BAT_BAR:
                return m.BATout[i] <= sistema.BATmax_out[i]
            else:
                return m.BATout[i] == 0
        model.C_LimiteDescargaBateria = pyo.Constraint(model.BAR, rule=C_LimiteDescargaBateria)
        
        def C_LimiteArmazBateria(m, i):
            return (0, m.BATarm[i], sistema.BATcapacidade[i])
        model.C_LimiteArmazBateria = pyo.Constraint(model.BAT_BAR, rule=C_LimiteArmazBateria)
        
        # BALANÇO ENERGÉTICO DA BATERIA
        def C_BalancoPotBateria(m, i):
            return m.BATarm[i] == sistema.BATarm_inicial[i] + m.BATin[i] - m.BATout[i]
        model.C_BalancoPotBateria = pyo.Constraint(model.BAT_BAR, rule=C_BalancoPotBateria)
        
        # BALANÇO DE POTÊNCIA
        def C_BalancoPotencia(m, i):
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

        model.C_BalancoPotencia = pyo.Constraint(model.BAR, rule=C_BalancoPotencia)
        
        # FUNÇÃO OBJETIVO SIMPLIFICADA
        def FOB(m):
            custo_geracao = sum(sistema.CPG[g] * m.PG[g] for g in m.GER)
            custo_curtailment = 100 * sum(m.CURTAILMENT[g] for g in m.GWD)
            custo_deficit = 1000 * sum(m.DEFICIT[b] for b in m.BAR)
            custo_bateria_in = sum(sistema.BATcusto_in[i] * m.BATin[i] for i in m.BAT_BAR)
            custo_bateria_out = sum(sistema.BATcusto_out[i] * m.BATout[i] for i in m.BAT_BAR)
            
            return custo_geracao + custo_curtailment + custo_deficit + custo_bateria_in + custo_bateria_out
        
        model.FOB = pyo.Objective(rule=FOB, sense=pyo.minimize)
        
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
            curtailment_total = 0.0
            for g in model.GWD:
                curtailment_total += pyo.value(model.CURTAILMENT[g])
            
            deficit_total = sum(DEFICIT_val)
            
            # Custo
            custo_total = pyo.value(model.FOB)
            
            # Fluxos
            fluxos_val = [0.0] * sistema.NLIN
            for e in range(sistema.NLIN):
                i, j = sistema.line_fr[e], sistema.line_to[e]
                fluxos_val[e] = sistema.y_line[e] * (ANG_val[i] - ANG_val[j])
            
            return ResultadoOPF(True, PG_val, ANG_val, custo_total, 
                              fluxos_val, deficit_total, curtailment_total,
                              BATin_val, BATout_val, BATarm_val)
        else:
            print(f"OPF não convergiu: {results.solver.termination_condition}")
            return ResultadoOPF(False, 
                               [0.0] * sistema.NGER, 
                               [0.0] * sistema.NBAR, 
                               0.0, 
                               [0.0] * sistema.NLIN,
                               0.0, 0.0, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR)
    
    except Exception as e:
        print(f"Erro no OPF: {e}")
        return ResultadoOPF(False, 
                           [0.0] * sistema.NGER, 
                           [0.0] * sistema.NBAR, 
                           0.0, 
                           [0.0] * sistema.NLIN,
                           0.0, 0.0, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR)

def criar_load_shape_24h():
    """Cria um perfil típico de carga para 24 horas"""
    return [1,2,3]
#  return [
#         0.7, 0.8, 0.8, 0.9, 1.0, 1.2,  # 00-05h: Madrugada
#         1.3, 0.8, 0.9, 1.0, 1.2, 1.0,  # 06-11h: Manhã
#         0.9, 1.0, 1.0, 1.2, 1.8, 1.5,  # 12-17h: Tarde
#         1.7, 1.5, 1.2, 1.3, 0.8, 0.7   # 18-23h: Noite
#     ]

def SolveDespachoEconomicoCenario(sistema, fator_demanda=1.0, hora=0, primeiro_periodo=True):
    """
    Simula despacho econômico para 1 hora
    
    Args:
        sistema: Sistema elétrico
        fator_demanda: Fator de multiplicação da demanda
        hora: Hora atual da simulação
        primeiro_periodo: Se True, usa carga como percentual da capacidade máxima
                         Se False, considera armazenamento da hora anterior
    """
    # Salvar demanda original
    PLOAD_original = sistema.PLOAD.copy()
    
    # Ajustar demanda
    if fator_demanda != 1.0:
        sistema.PLOAD = PLOAD_original * fator_demanda
    
    # Resolver OPF
    resultado = SolveOPF(sistema, sistema.Bbus)
    
    # Restaurar demanda original
    sistema.PLOAD = PLOAD_original
    
    # Relatório
    print(f"\nHORA {hora:02d}:00 - Fator Demanda: {fator_demanda}")
    print("-" * 40)
    
    if resultado.sucesso:
        print(f"✓ Custo: {resultado.custo_total:.2f} USD")
        print(f"  Geração: {sum(resultado.PG):.3f} pu, Demanda: {sum(sistema.PLOAD):.3f} pu")
        print(f"  Curtailment: {resultado.curtailment_total:.3f} pu, Déficit: {resultado.deficit_total:.3f} pu")
        print(f"  Baterias: Carga={sum(resultado.BATin):.3f}, Descarga={sum(resultado.BATout):.3f}")
        
        if sistema.BARRAS_COM_BATERIA:
            print("  Estado baterias:")
            for idx in sistema.BARRAS_COM_BATERIA:
                estado_anterior = sistema.BATarm_inicial[idx]
                estado_atual = resultado.BATarm[idx]
                variacao = estado_atual - estado_anterior
                print(f"    Barra {idx}: {estado_anterior:.3f} → {estado_atual:.3f} ({variacao:+.3f})")
    else:
        print(f"✗ Falha na convergência")
    
    return {
        'hora': hora,
        'fator_demanda': fator_demanda,
        'sucesso': resultado.sucesso,
        'custo': resultado.custo_total if resultado.sucesso else 0,
        'geracao_total': sum(resultado.PG) if resultado.sucesso else 0,
        'demanda_total': sum(sistema.PLOAD),
        'curtailment': resultado.curtailment_total if resultado.sucesso else 0,
        'deficit': resultado.deficit_total if resultado.sucesso else 0,
        'carga_bateria': sum(resultado.BATin) if resultado.sucesso else 0,
        'descarga_bateria': sum(resultado.BATout) if resultado.sucesso else 0,
        'BATarm_atual': resultado.BATarm.copy() if resultado.sucesso else sistema.BATarm_inicial.copy()
    }

def SolveCenarioDespachoDia(json_file_path, load_shape=None):
    """Executa despacho econômico para um dia inteiro"""
    
    if load_shape is None:
        load_shape = criar_load_shape_24h()
    
    # Carregar sistema
    sistema = Sistema(json_file_path)
    
    print("=" * 50)
    print("DESPACHO ECONÔMICO - 24 HORAS")
    print("=" * 50)
    
    resultados = []
    
    for hora, fator in enumerate(load_shape):
        # Primeira hora: usa carga como percentual da capacidade máxima
        # Horas seguintes: considera armazenamento da hora anterior
        primeiro_periodo = (hora == 0)
        
        resultado_hora = SolveDespachoEconomicoCenario(
            sistema=sistema,
            fator_demanda=fator,
            hora=hora,
            primeiro_periodo=primeiro_periodo
        )
        resultados.append(resultado_hora)
        
        # Atualizar estado das baterias para a próxima hora
        if resultado_hora['sucesso'] and hora < len(load_shape) - 1:
            sistema.BATarm_inicial = resultado_hora['BATarm_atual']
    
    # Relatório consolidado
    df = pd.DataFrame(resultados)
    
    print("\n" + "=" * 50)
    print("RELATÓRIO CONSOLIDADO - 24 HORAS")
    print("=" * 50)
    
    print(f"Horas com sucesso: {sum(df['sucesso'])}/{len(df)}")
    print(f"Custo total: ${df['custo'].sum():.2f} USD")
    print(f"Geração total: {df['geracao_total'].sum():.3f} pu")
    print(f"Demanda total: {df['demanda_total'].sum():.3f} pu")
    print(f"Curtailment total: {df['curtailment'].sum():.3f} pu")
    print(f"Déficit total: {df['deficit'].sum():.3f} pu")
    print(f"Operação baterias - Carga: {df['carga_bateria'].sum():.3f} pu")
    print(f"Operação baterias - Descarga: {df['descarga_bateria'].sum():.3f} pu")
    
    return df

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    try:
        # Executar despacho econômico para 24 horas
        resultados_df = SolveCenarioDespachoDia("DATA/input/3barras_TESTE.json")
        
        print("\n✅ Despacho econômico concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a simulação: {e}")
        import traceback
        traceback.print_exc()