import pyomo.environ as pyo
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import copy
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

@dataclass
class ResultadoOPF:
    """Estrutura para armazenar resultados do OPF"""
    sucesso: bool
    PG: List[float]
    ANG: List[float] 
    custo_total: float
    cmo_total: float
    fluxos: List[float]
    deficit_total: float
    curtailment_total: float
    BATin: List[float]
    BATout: List[float]
    BATarm: List[float]

class SistemaTransmissao:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.carregaSistema()
        self.ProcessSistemaBase()
        self.MontaCenarioOPF()
        
    def carregaSistema(self):
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.barras = self.data["BARRAS"]
        self.geradores_data = self.data["GERADORES"]
        self.demandas_data = self.data["DEMANDAS"]
        self.linhas = self.data["LINHAS"]
        self.baterias_data = self.data.get("BATERIAS", [])
        
    def ProcessSistemaBase(self):
        self.SB = self.data["S_base"]
        self.PB = self.data["P_base"]
        self.VB = self.data["V_base"]
        self.ZB = (self.VB ** 2) / self.PB

    def MontaCenarioOPF(self):
        self.ProcessaPU()
        self.ProcessaDBAR()
        self.IdentificaBarraSlack()
        self.ProcessaDLIN()
        self.MontaBbus()
        self.ProcessaDGER()
        self.ProcessaLOAD()
        self.ProcessaDEF_GWD()
        self.ProcessaBAT()
    
    def ProcessaPU(self):
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
        
        # Geradores
        for g in self.geradores_data:
            g["Pmax_pu"] = g["PGERmax_MW"] / self.SB
            g["Pmin_pu"] = g["PGERmin_MW"] / self.SB
            g["custo_var_pu"] = g.get("custo_var_USD_MWh", 0.0) * self.SB
        
        # Linhas - armazenar valores originais para expansão
        for l in self.linhas:
            l["X_pu"] = l["X"] / self.ZB
            l["R_pu"] = l.get("R", 0.0) / self.ZB  # Adicionar resistência
            l["Fmax_pu_original"] = l["LIM_Fluxo"] / self.SB
            l["Fmax_pu"] = l["LIM_Fluxo"] / self.SB
            # Custo de investimento: 5 vezes o limite da linha
            l["custo_investimento"] = 5 * l["LIM_Fluxo"]  # $/MW
        
        # Baterias
        for bat in self.baterias_data:
            bat["Pmax_carga_base_pu"] = bat.get("Pmax_carga_MW", 0.0) / self.SB
            bat["Pmax_descarga_base_pu"] = bat.get("Pmax_descarga_MW", 0.0) / self.SB
            if "capacidade_armazenamento_MWh" in bat:
                bat["capacidade_base_pu"] = bat["capacidade_armazenamento_MWh"] / self.SB
            else:
                bat["capacidade_base_pu"] = bat["Pmax_carga_MW"] * 4 / self.SB
    
    def ProcessaDBAR(self):
        self.bus_ids = [b["ID_Barra"] for b in self.barras]
        self.idx_map = {id: i for i, id in enumerate(self.bus_ids)}
        self.NBAR = len(self.bus_ids)
        self.NLIN = len(self.linhas)

    def IdentificaBarraSlack(self):
        slack_list = [b for b in self.barras if b["tipo"] == "Slack"]
        if len(slack_list) != 1:
            raise ValueError("Deve haver exatamente 1 barra Slack")
        slack_id = slack_list[0]["ID_Barra"]
        self.slack_idx = self.idx_map[slack_id]
            
    def ProcessaDLIN(self):
        self.line_fr = []
        self.line_to = []
        self.x_line = np.zeros(self.NLIN)
        self.r_line = np.zeros(self.NLIN)
        self.y_line = np.zeros(self.NLIN)
        self.FLIM = np.zeros(self.NLIN)
        self.FLIM_original = np.zeros(self.NLIN)
        self.custo_investimento = np.zeros(self.NLIN)
        
        for e, ln in enumerate(self.linhas):
            fr = ln["ID_Barra_Origem"]
            to = ln["ID_Barra_Destino"]
            self.line_fr.append(self.idx_map[fr])
            self.line_to.append(self.idx_map[to])
            
            x = ln["X_pu"]
            r = ln.get("R_pu", 0.0)
            self.x_line[e] = x
            self.r_line[e] = r
            self.y_line[e] = 1.0 / x if abs(x) > 0 else 0.0
            self.FLIM_original[e] = ln["Fmax_pu_original"]
            self.FLIM[e] = ln["Fmax_pu"]
            self.custo_investimento[e] = ln["custo_investimento"]
    
    def atualizar_expansao_linhas(self, n_circuitos):
        """
        Atualiza parâmetros das linhas com base na expansão (nº de circuitos)
        n_circuitos: lista com número de circuitos em cada linha [n12, n13, n23]
        """
        for e in range(self.NLIN):
            n = n_circuitos[e]
            # Reatância equivalente para n circuitos em paralelo
            self.x_line[e] = self.linhas[e]["X_pu"] / n
            self.y_line[e] = 1.0 / self.x_line[e]
            # Limite de fluxo aumenta linearmente com nº de circuitos
            self.FLIM[e] = self.FLIM_original[e] * n
        # Recalcular matriz Bbus
        self.MontaBbus()
    
    def MontaBbus(self):
        self.Bbus = np.zeros((self.NBAR, self.NBAR))
        for e in range(self.NLIN):
            i = self.line_fr[e]
            j = self.line_to[e]
            y = self.y_line[e]
            self.Bbus[i, i] += y
            self.Bbus[j, j] += y
            self.Bbus[i, j] -= y
            self.Bbus[j, i] -= y
    
    def ProcessaDGER(self):
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
    
    def ProcessaLOAD(self):
        self.PLOAD = np.zeros(self.NBAR)
        for d in self.demandas_data:
            id_barra = d["ID_Barra"]
            idx = self.idx_map[id_barra]
            self.PLOAD[idx] += d.get("PLOAD", 0.0) / self.SB

    def atualizar_perfis_horarios(self, perfil_carga, perfil_eolica, hora):
        """
        Atualiza carga e geração eólica para uma hora específica
        """
        # Atualizar carga na barra 3 (índice 2 se barra 1=0, 2=1, 3=2)
        idx_barra3 = self.idx_map[3]
        self.PLOAD[idx_barra3] = perfil_carga[hora]
        
        # Atualizar geração eólica disponível na barra 2
        idx_eolico = None
        for i, g in enumerate(self.geradores_data):
            if g["Tipo"] == "GWD":
                idx_eolico = i
                break
        
        if idx_eolico is not None:
            self.PGMAX_EFETIVO[idx_eolico] = perfil_eolica[hora]
            # Reconstruir PGMAX
            self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, 
                                        self.PGMAX_CURTAILMENT, 
                                        self.PGMAX_DEFICIT])

    def ProcessaBAT(self):
        self.BARRAS_COM_BATERIA = []
        self.BATmax_in = np.zeros(self.NBAR)
        self.BATmax_out = np.zeros(self.NBAR)
        self.BATcapacidade = np.zeros(self.NBAR)
        self.BATcusto_in = np.zeros(self.NBAR)
        self.BATcusto_out = np.zeros(self.NBAR)
        self.BATarm_inicial = np.zeros(self.NBAR)
        
        for bat in self.baterias_data:
            id_barra = bat["ID_Barra"]
            if id_barra not in self.idx_map:
                continue
            idx = self.idx_map[id_barra]
            self.BARRAS_COM_BATERIA.append(idx)
            self.BATmax_in[idx] = bat["Pmax_carga_base_pu"]
            self.BATmax_out[idx] = bat["Pmax_descarga_base_pu"]
            self.BATcapacidade[idx] = bat["capacidade_base_pu"]
            self.BATarm_inicial[idx] = 0.5 * self.BATcapacidade[idx]  # SOC 50% inicial
            
            self.BATcusto_in[idx] = -0.1*bat.get("custo_carga_USD_MWh", 0.0)
            self.BATcusto_out[idx] = bat.get("custo_descarga_USD_MWh", 0.0)
    
    def ProcessaDEF_GWD(self):
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        barras_com_gerador = set(self.BARPG_ORIGINAL)
        self.barras_PQ_sem_gerador = [b for b in barras_PQ 
                                    if self.idx_map[b["ID_Barra"]] not in barras_com_gerador]
        
        # Curtailment (corte de vento)
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
                self.CPG_CURTAILMENT[i] = 1000  # Custo alto para desencorajar corte
        
        # Déficit (corte de carga)
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
            self.CPG_DEFICIT[i] = 5000  # Custo muito alto para corte de carga
        
        # Combinar todos os geradores
        self.NGER = self.NGER_ORIGINAL + self.NGER_CURTAILMENT + self.NGER_DEFICIT
        self.BARPG = self.BARPG_ORIGINAL + self.BARPG_CURTAILMENT + self.BARPG_DEFICIT
        self.PGMIN = np.concatenate([self.PGMIN_EFETIVO, self.PGMIN_CURTAILMENT, self.PGMIN_DEFICIT])
        self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, self.PGMAX_CURTAILMENT, self.PGMAX_DEFICIT])
        self.CPG = np.concatenate([self.CPG_ORIGINAL, self.CPG_CURTAILMENT, self.CPG_DEFICIT])

def SolveOPF(sistema, considerar_perdas=False):
    """OPF DC com restrições de transmissão e bateria"""
    try:
        model = pyo.ConcreteModel()
        
        # Conjuntos
        model.GER = pyo.Set(initialize=range(sistema.NGER))
        model.BAR = pyo.Set(initialize=range(sistema.NBAR))
        model.LIN = pyo.Set(initialize=range(sistema.NLIN))
        
        # Geradores GWD para curtailment
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
        model.FLUXO = pyo.Var(model.LIN, within=pyo.Reals)
        
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
        
        # FLUXO NAS LINHAS
        def C_DefinicaoFluxo(m, e):
            i = sistema.line_fr[e]
            j = sistema.line_to[e]
            return m.FLUXO[e] == (m.ANG[i] - m.ANG[j]) / sistema.x_line[e]
        model.C_DefinicaoFluxo = pyo.Constraint(model.LIN, rule=C_DefinicaoFluxo)
        
        # LIMITES DE FLUXO
        def C_LimiteFluxoPos(m, e):
            return m.FLUXO[e] <= sistema.FLIM[e]
        model.C_LimiteFluxoPos = pyo.Constraint(model.LIN, rule=C_LimiteFluxoPos)
        
        def C_LimiteFluxoNeg(m, e):
            return m.FLUXO[e] >= -sistema.FLIM[e]
        model.C_LimiteFluxoNeg = pyo.Constraint(model.LIN, rule=C_LimiteFluxoNeg)
        
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
            
            # Fluxos 
            fluxo_liquido = 0.0
            for e in m.LIN:
                if sistema.line_fr[e] == i:
                    fluxo_liquido += m.FLUXO[e]
                elif sistema.line_to[e] == i:
                    fluxo_liquido -= m.FLUXO[e]
            
            # Perdas nas linhas
            perdas = 0.0
            if considerar_perdas:
                for e in m.LIN:
                    if sistema.line_fr[e] == i or sistema.line_to[e] == i:
                        perdas += sistema.r_line[e] * (m.FLUXO[e]**2) / 2
            
            return geracao_total + contribuicao_bateria + deficit_barra - fluxo_liquido - perdas == sistema.PLOAD[i]

        model.C_BalancoPotencia = pyo.Constraint(model.BAR, rule=C_BalancoPotencia)
        
        # FUNÇÃO OBJETIVO
        def FOB(m):
            # Custo de geração
            custo_geracao = sum(sistema.CPG[g] * m.PG[g] for g in m.GER)
            
            # Custo de curtailment (corte de vento)
            custo_curtailment = 1000 * sum(m.CURTAILMENT[g] for g in m.GWD)
            
            # Custo de déficit (corte de carga)
            custo_deficit = 5000 * sum(m.DEFICIT[b] for b in m.BAR)
            
            # Custo de operação da bateria
            custo_bateria_in = sum(sistema.BATcusto_in[i] * m.BATin[i] for i in m.BAT_BAR)
            custo_bateria_out = sum(sistema.BATcusto_out[i] * m.BATout[i] for i in m.BAT_BAR)
            
            return custo_geracao + custo_curtailment + custo_deficit + custo_bateria_in + custo_bateria_out
        
        model.FOB = pyo.Objective(rule=FOB, sense=pyo.minimize)
        # Habilitar duais
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        # RESOLVER
        solver = pyo.SolverFactory('couenne', executable='../../../scripts/couenne/couenne')
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
            #TODO:
            # #print("\n=== CUSTOS MARGINAIS OPERACIONAIS (CMOs) ===")
            # for i in model.BAR:
            #     try:
            #         # Access using .get() with None as default
            #         cmo_valor = model.dual.get(model.C_BalancoPotencia[i])
            #         if cmo_valor is not None:
            #             print(f"Barra {i}: CMO = R$ {cmo_valor:.2f}/MWh")
            #         else:
            #             print(f"Barra {i}: Dual não retornado pelo solver")
            #     except Exception as e:
            #         print(f"Barra {i}: Erro ao acessar dual - {str(e)}")
            cmo_total = 0#float(sum(cmo))
            # Fluxos
            fluxos_val = [0.0] * sistema.NLIN
            for e in range(sistema.NLIN):
                fluxos_val[e] = pyo.value(model.FLUXO[e])
            
            return ResultadoOPF(True, PG_val, ANG_val, custo_total, cmo_total,
                              fluxos_val, deficit_total, curtailment_total,
                              BATin_val, BATout_val, BATarm_val)
        else:
            print(f"OPF não convergiu: {results.solver.termination_condition}")
            return ResultadoOPF(False, 
                               [0.0] * sistema.NGER, 
                               [0.0] * sistema.NBAR, 
                               0.0,
                               0.0,
                               [0.0] * sistema.NLIN,
                               0.0, 0.0, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR)
    
    except Exception as e:
        print(f"Erro no OPF: {e}")
        import traceback
        traceback.print_exc()
        return ResultadoOPF(False, 
                           [0.0] * sistema.NGER, 
                           [0.0] * sistema.NBAR, 
                           0.0,
                           0.0,
                           [0.0] * sistema.NLIN,
                           0.0, 0.0, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR, [0.0] * sistema.NBAR)

def calcular_custo_operacao_24h(sistema, perfil_carga, perfil_eolica):
    """
    Calcula custo de operação para 24 horas
    """
    custo_total = 0
    resultados_horarios = []
    
    # Salvar estado inicial da bateria
    bat_arm_inicial_original = sistema.BATarm_inicial.copy()
    
    for hora in range(24):
        # Atualizar perfis para a hora
        sistema.atualizar_perfis_horarios(perfil_carga, perfil_eolica, hora)
        
        # Resolver OPF
        resultado = SolveOPF(sistema, considerar_perdas=True)
        
        if resultado.sucesso:
            custo_total += resultado.custo_total
            resultados_horarios.append({
                'hora': hora,
                'custo': resultado.custo_total,
                'PGB_in': sum(resultado.BATin),
                'PGB_out': sum(resultado.BATout),
                'curtailment': resultado.curtailment_total,
                'deficit': resultado.deficit_total
            })
            
            # Atualizar estado da bateria para próxima hora
            sistema.BATarm_inicial = resultado.BATarm.copy()
        else:
            # Retornar custo muito alto se falhar
            return float('inf'), []
    
    # Restaurar estado inicial
    sistema.BATarm_inicial = bat_arm_inicial_original
    
    return custo_total, resultados_horarios

def fitness(investimento, sistema_base, perfil_carga, perfil_eolica):
    """
    Função de fitness para o algoritmo genético
    investimento: vetor com número de circuitos em cada linha [n12, n13, n23]
    """
    # Garantir que investimento é inteiro (circuitos)
    n_circuitos = [int(max(1, round(x))) for x in investimento]
    
    # Clonar sistema
    sistema = copy.deepcopy(sistema_base)
    # Aplicar expansão de linhas
    sistema.atualizar_expansao_linhas(n_circuitos)
    
    # Calcular custo de investimento
    custo_investimento = 0
    for e in range(sistema.NLIN):
        # Custo = 5 * limite_original * (n_circuitos - 1)
        custo_investimento += sistema.custo_investimento[e] * (n_circuitos[e] - 1)
    
    # Calcular custo de operação para 24 horas
    custo_operacao, resultados = calcular_custo_operacao_24h(sistema, perfil_carga, perfil_eolica)
    
    if custo_operacao == float('inf'):
        return 1e10  # Penalidade alta para soluções inviáveis
    
    penalidade_quantidade = 1e6*sum(n_circuitos)

    # Custo total = investimento + operação
    custo_total = custo_investimento + custo_operacao + penalidade_quantidade
    
    return custo_total

class PlanejamentoTransmissao:
    def __init__(self, json_file_path):
        self.sistema_base = SistemaTransmissao(json_file_path)
        self.melhor_solucao = None
        self.melhor_custo = float('inf')
        self.historico = []
        
    def criar_perfis_horarios(self):
        """
        Cria perfis típicos de carga e geração eólica para 24 horas
        """
        # Perfil de carga (valores em pu)
        carga_base = self.sistema_base.PLOAD[self.sistema_base.idx_map[3]]
        load_shape = [
            0.7, 0.6, 0.5, 0.5, 0.6, 0.8,  # 00-05h
            1.0, 1.2, 1.3, 1.2, 1.1, 1.0,  # 06-11h
            0.9, 0.8, 0.8, 0.9, 1.2, 2.2,  # 12-17h
            5, 1.8, 1.3, 1.0, 0.8, 0.7   # 18-23h
        ]
        self.perfil_carga = [carga_base * f for f in load_shape]
        
        # Perfil eólica (valores em pu)
        eolica_base = 0.12  # 12 MW em pu
        eolica_shape = [
            1.9, 0.8, 0.7, 0.6, 0.5, 0.4,  # 00-05h: vento decrescente
            0.3, 5.2, 0.3, 0.4, 0.6, 0.8,  # 06-11h: vento aumentando
            0.9, 1.0, 2.9, 0.8, 0.7, 0.6,  # 12-17h: vento forte
            0.5, 0.4, 0.3, 0.2, 0.1, 0.1   # 18-23h: vento fraco
        ]
        self.perfil_eolica = [eolica_base * f for f in eolica_shape]
        
    def otimizar_investimento(self):
        """
        Otimiza investimento em linhas usando algoritmo genético
        """
        print("=" * 60)
        print("PLANEJAMENTO DE TRANSMISSÃO TÉRMICO-EÓLICO")
        print("=" * 60)
        
        # Criar perfis horários
        self.criar_perfis_horarios()
        
        # Configurar algoritmo genético
        varbounds = np.array([
            [1, 3],  # Linha 1-2: 1 a 3 circuitos
            [1, 3],  # Linha 1-3: 1 a 3 circuitos
            [1, 3]   # Linha 2-3: 1 a 3 circuitos
        ])
        
        vartype = np.array(['int', 'int', 'int'])
        
        algorithm_param = {
            'max_num_iteration': 5,
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 10
        }
        
        # Função de fitness wrapper
        def fitness_wrapper(X):
            custo = fitness(X, self.sistema_base, self.perfil_carga, self.perfil_eolica)
            self.historico.append({'solucao': X.copy(), 'custo': custo})
            return custo
        
        # Executar otimização
        model = ga(function=fitness_wrapper,
                  dimension=3,
                  variable_type_mixed=vartype,
                  variable_boundaries=varbounds,
                  algorithm_parameters=algorithm_param)
        
        model.run()
        
        # Armazenar melhor solução
        self.melhor_solucao = model.output_dict['variable']
        self.melhor_custo = model.output_dict['function']
        
        # Aplicar melhor solução ao sistema
        n_circuitos = [int(round(x)) for x in self.melhor_solucao]
        self.sistema_otimo = copy.deepcopy(self.sistema_base)
        self.sistema_otimo.atualizar_expansao_linhas(n_circuitos)
        
        return self.melhor_solucao, self.melhor_custo
    
    def analisar_solucao_otima(self):
        """
        Analisa a solução ótima encontrada
        """
        print("\n" + "=" * 60)
        print("ANÁLISE DA SOLUÇÃO ÓTIMA")
        print("=" * 60)
        
        n_circuitos = [int(round(x)) for x in self.melhor_solucao]
        
        print(f"\nConfiguração ótima de circuitos:")
        print(f"  Linha 1-2: {n_circuitos[0]} circuitos")
        print(f"  Linha 1-3: {n_circuitos[1]} circuitos")
        print(f"  Linha 2-3: {n_circuitos[2]} circuitos")
        
        # Calcular custos detalhados
        custo_investimento = 0
        for e in range(self.sistema_base.NLIN):
            custo_investimento += self.sistema_base.custo_investimento[e] * (n_circuitos[e] - 1)
        
        custo_operacao, resultados = calcular_custo_operacao_24h(
            self.sistema_otimo, self.perfil_carga, self.perfil_eolica
        )
        
        # # Relatório consolidado
        # df = pd.DataFrame(resultados)
        
        # print("\n" + "=" * 50)
        # print("RELATÓRIO CONSOLIDADO - 24 HORAS")
        # print("=" * 50)
        
        # print(f"Custo total: ${df['custo'].sum():.2f} USD")
        # print(f"Geração total: {df['geracao_total'].sum():.3f} pu")
        # print(f"Demanda total: {df['demanda_total'].sum():.3f} pu")
        # print(f"Curtailment total: {df['curtailment'].sum():.3f} pu")
        # print(f"Déficit total: {df['deficit'].sum():.3f} pu")
        # print(f"Operação baterias - Carga: {df['carga_bateria'].sum():.3f} pu")
        # print(f"Operação baterias - Descarga: {df['descarga_bateria'].sum():.3f} pu")



        # print(f"\nCustos:")
        # print(f"  Investimento em linhas: ${custo_investimento:.2f}")
        # print(f"  Operação (24h): ${custo_operacao:.2f}")
        # print(f"  Total: ${self.melhor_custo:.2f}")
        
        # Analisar operação da bateria
        print(f"\nOperação da Bateria (Barra 3):")
        
        # Simular operação para extrair dados da bateria
        bat_arm_inicial_original = self.sistema_otimo.BATarm_inicial.copy()
        operacao_bateria = []
        
        for hora in range(24):
            self.sistema_otimo.atualizar_perfis_horarios(self.perfil_carga, self.perfil_eolica, hora)
            resultado = SolveOPF(self.sistema_otimo, considerar_perdas=True)
            
            if resultado.sucesso:
                idx_barra3 = self.sistema_otimo.idx_map[3]
                bat_in = resultado.BATin[idx_barra3]
                bat_out = resultado.BATout[idx_barra3]
                operacao_bateria.append({
                    'hora': hora,
                    'PGB_in': bat_in,
                    'PGB_out': bat_out,
                    'SOC': resultado.BATarm[idx_barra3] / self.sistema_otimo.BATcapacidade[idx_barra3]
                })
                self.sistema_otimo.BATarm_inicial = resultado.BATarm.copy()
        
        # Restaurar estado inicial
        self.sistema_otimo.BATarm_inicial = bat_arm_inicial_original
        
        # Calcular totais
        total_in = sum([op['PGB_in'] for op in operacao_bateria])
        total_out = sum([op['PGB_out'] for op in operacao_bateria])
        
        print(f"  PGB_in total (24h): {total_in:.3f} pu ({total_in * self.sistema_otimo.SB:.2f} MW)")
        print(f"  PGB_out total (24h): {total_out:.3f} pu ({total_out * self.sistema_otimo.SB:.2f} MW)")
        print(f"  Balanço líquido: {total_in - total_out:.3f} pu")
        
        # Mostrar operação por hora
        print(f"\nDetalhamento por hora:")
        print(f"{'Hora':<6} {'PGB_in (pu)':<12} {'PGB_out (pu)':<12} {'SOC (%)':<10}")
        print("-" * 40)
        
        for op in operacao_bateria[:12]:  # Primeiras 12 horas
            print(f"{op['hora']:02d}:00 {op['PGB_in']:11.3f} {op['PGB_out']:11.3f} {op['SOC']*100:9.1f}")
        
        print("\n...")
        
        for op in operacao_bateria[12:]:  # Últimas 12 horas
            print(f"{op['hora']:02d}:00 {op['PGB_in']:11.3f} {op['PGB_out']:11.3f} {op['SOC']*100:9.1f}")
        
        return operacao_bateria
    
    def plotar_resultados(self, operacao_bateria):
        """
        Plota resultados da operação
        """
        horas = [op['hora'] for op in operacao_bateria]
        pgb_in = [op['PGB_in'] for op in operacao_bateria]
        pgb_out = [op['PGB_out'] for op in operacao_bateria]
        soc = [op['SOC'] * 100 for op in operacao_bateria]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico 1: Operação da bateria
        ax1.bar(horas, pgb_in, width=0.4, label='PGB_in (Carga)', alpha=0.7, color='blue')
        ax1.bar([h + 0.4 for h in horas], pgb_out, width=0.4, label='PGB_out (Descarga)', alpha=0.7, color='red')
        ax1.set_xlabel('Hora do Dia')
        ax1.set_ylabel('Potência (pu)')
        ax1.set_title('Operação da Bateria - PGB_in e PGB_out')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Gráfico 2: Estado de Carga
        ax2.plot(horas, soc, marker='o', linewidth=2, color='green')
        ax2.set_xlabel('Hora do Dia')
        ax2.set_ylabel('SOC (%)')
        ax2.set_title('Estado de Carga da Bateria')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig('resultados_planejamento.png', dpi=150)
        plt.show()
    
    def executar_planejamento(self):
        """
        Executa todo o processo de planejamento
        """
        # Etapa 1: Otimização do investimento
        print("\nETAPA 1: OTIMIZAÇÃO DO INVESTIMENTO EM LINHAS")
        solucao_otima, custo_total = self.otimizar_investimento()
        
        # Etapa 2: Análise da solução ótima
        print("\nETAPA 2: ANÁLISE DA OPERAÇÃO ÓTIMA")
        operacao_bateria = self.analisar_solucao_otima()
        
        # Etapa 3: Visualização
        print("\nETAPA 3: VISUALIZAÇÃO DOS RESULTADOS")
        self.plotar_resultados(operacao_bateria)
        
        return solucao_otima, custo_total, operacao_bateria

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    try:
        # Inicializar planejamento
        planejador = PlanejamentoTransmissao("DATA/input/3barras_PET.json")
        
        # Executar planejamento completo
        solucao_otima, custo_total, operacao_bateria = planejador.executar_planejamento()
        
        # Resumo final
        print("\n" + "=" * 60)
        print("RESUMO DO PLANEJAMENTO")
        print("=" * 60)
        print(f"\nSolução ótima encontrada:")
        print(f"  Circuitos linha 1-2: {int(round(solucao_otima[0]))}")
        print(f"  Circuitos linha 1-3: {int(round(solucao_otima[1]))}")
        print(f"  Circuitos linha 2-3: {int(round(solucao_otima[2]))}")
        print(f"\nCusto total: ${custo_total:.2f}")
        
        # Calcular PGB_in e PGB_out totais
        total_pgb_in = sum([op['PGB_in'] for op in operacao_bateria]) * planejador.sistema_base.SB
        total_pgb_out = sum([op['PGB_out'] for op in operacao_bateria]) * planejador.sistema_base.SB
        
        print(f"\nOperação da Bateria (em MW):")
        print(f"  PGB_in total: {total_pgb_in:.2f} MW")
        print(f"  PGB_out total: {total_pgb_out:.2f} MW")
        print(f"  Saldo líquido: {total_pgb_in - total_pgb_out:.2f} MW")
        
    except Exception as e:
        print(f"\nErro durante o planejamento: {e}")
        import traceback
        traceback.print_exc()