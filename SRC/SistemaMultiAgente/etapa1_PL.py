import pyomo.environ as pyo
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import copy
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
    perdas: float

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
        self.VB = self.data["V_base"]
        self.f_base = self.data["f_base"]
        self.ZB = (self.VB ** 2) / self.SB
        
    def MontaCenarioOPF(self):
        self.ProcessaPU()
        self.ProcessaDBAR()
        self.IdentificaBarraSlack()
        self.ProcessaDLIN()
        self.MontaBbus()
        self.ProcessaDGER()
        self.ProcessaLOAD()
        self.ProcessaDEF_GWD()
    
    def ProcessaPU(self):
        """Converte todos os valores para PU"""
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
            b["Q_carga_pu"] = b.get("Q_carga_MVAr", 0.0) / self.SB
        
        # Geradores
        for g in self.geradores_data:
            g["PGERmin_pu"] = g.get("PGERmin_MW", 0.0) / self.SB
            g["PGERmax_pu"] = g.get("PGERmax_MW", 0.0) / self.SB
            g["Qmin_pu"] = g.get("Qmin_MW", 0.0) / self.SB
            g["Qmax_pu"] = g.get("Qmax_MW", 0.0) / self.SB
            
            # Custos em USD/pu (custo USD/MW * S_base)
            g["custo_var_pu"] = g.get("custo_var_USD_MW", 0.0) * self.SB
            g["custo_curtailment_pu"] = g.get("custo_curtailment_USD_MW", 100.0) * self.SB
            
            # Rampas em pu/h
            if "ramp_up_MW_h" in g:
                g["ramp_up_pu_h"] = g["ramp_up_MW_h"] / self.SB
                g["ramp_down_pu_h"] = g["ramp_down_MW_h"] / self.SB
        
        # Demandas
        for d in self.demandas_data:
            d["PLOAD_pu"] = d.get("PLOAD", 0.0) / self.SB
            d["QLOAD_pu"] = d.get("QLOAD_MW", 0.0) / self.SB
        
        # Linhas - converter para PU
        for l in self.linhas:
            # Impedância já pode estar em PU
            if l.get("R_Unidade", "pu") != "pu":
                l["R"] = l["R"] / self.ZB
                l["X"] = l["X"] / self.ZB
            
            # Admitância shunt
            if "Bsh" in l and l.get("Bsh_Unidade", "pu") != "pu":
                l["Bsh"] = l["Bsh"] * self.ZB
            
            # Limite de fluxo em PU
            if l.get("LIM_Fluxo_Unidade", "MW") == "MW":
                l["Fmax_pu"] = l["LIM_Fluxo"] / self.SB
            else:
                l["Fmax_pu"] = l["LIM_Fluxo"]
            
            # Custo de investimento
            l["custo_investimento"] = 5 * (l["LIM_Fluxo"] if l.get("LIM_Fluxo_Unidade", "MW") == "MW" else l["LIM_Fluxo"] * self.SB)
        
        # Baterias
        for bat in self.baterias_data:
            if "Pmax_carga_MW" in bat:
                bat["Pmax_carga_pu"] = bat["Pmax_carga_MW"] / self.SB
                bat["Pmax_descarga_pu"] = bat.get("Pmax_descarga_MW", 0.0) / self.SB
            if "capacidade_armazenamento_MWh" in bat:
                bat["capacidade_pu"] = bat["capacidade_armazenamento_MWh"] / self.SB
        
        print("✓ Sistema convertido para PU")
    
    def ProcessaDBAR(self):
        self.bus_ids = [b["ID_Barra"] for b in self.barras]
        self.idx_map = {id: i for i, id in enumerate(self.bus_ids)}
        self.indice_para_barra = {i: id for i, id in enumerate(self.bus_ids)}  # Mapeamento inverso
        self.NBAR = len(self.bus_ids)
        self.NLIN = len(self.linhas)

    def IdentificaBarraSlack(self):
        slack_list = [b for b in self.barras if b["tipo"] == "Slack"]
        if len(slack_list) != 1:
            raise ValueError("Deve haver exatamente 1 barra Slack")
        slack_id = slack_list[0]["ID_Barra"]
        self.slack_idx = self.idx_map[slack_id]
            
    def ProcessaDLIN(self):
        """Processa dados das linhas em PU"""
        self.line_fr = []
        self.line_to = []
        self.x_line = np.zeros(self.NLIN)
        self.r_line = np.zeros(self.NLIN)
        self.b_shunt = np.zeros(self.NLIN)
        self.g_serie = np.zeros(self.NLIN)  # Condutância série
        self.b_serie = np.zeros(self.NLIN)  # Susceptância série
        self.FLIM = np.zeros(self.NLIN)
        self.custo_investimento = np.zeros(self.NLIN)
        
        for e, ln in enumerate(self.linhas):
            fr = ln["ID_Barra_Origem"]
            to = ln["ID_Barra_Destino"]
            self.line_fr.append(self.idx_map[fr])
            self.line_to.append(self.idx_map[to])
            
            # Impedância em PU
            x = ln["X"]
            r = ln.get("R", 0.0)
            b_sh = ln.get("Bsh", 0.0) / 2.0  # Metade em cada extremidade
            
            self.x_line[e] = x
            self.r_line[e] = r
            self.b_shunt[e] = b_sh
            
            # Calcular condutância e susceptância série
            if abs(x) > 0 or abs(r) > 0:
                denom = r**2 + x**2
                self.g_serie[e] = r / denom
                self.b_serie[e] = -x / denom
            
            self.FLIM[e] = ln["Fmax_pu"]
            self.custo_investimento[e] = ln["custo_investimento"]
    
    def MontaBbus(self):
        """Constrói matriz de susceptância (simplificada DC)"""
        self.Bbus = np.zeros((self.NBAR, self.NBAR))
        for e in range(self.NLIN):
            i = self.line_fr[e]
            j = self.line_to[e]
            # Para fluxo DC linearizado: susceptância = 1/x
            if abs(self.x_line[e]) > 0:
                b = 1.0 / self.x_line[e]
                self.Bbus[i, i] += b
                self.Bbus[j, j] += b
                self.Bbus[i, j] -= b
                self.Bbus[j, i] -= b
    
    def ProcessaDGER(self):
        """Processa dados dos geradores em PU"""
        self.NGER_ORIGINAL = len(self.geradores_data)
        self.BARPG_ORIGINAL = []
        self.BAR_GWD = []
        self.PGMIN_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMIN_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.CPG_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.GER_TIPOS = []
        
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            barra_idx = self.idx_map[id_barra]
            self.BARPG_ORIGINAL.append(barra_idx)
            self.GER_TIPOS.append(g.get("Tipo", "CONV"))
            if g.get("Tipo", "") == "GWD":
                self.BAR_GWD.append(i)  # Armazenar o índice do gerador, não da barra
        
        for i, g in enumerate(self.geradores_data):
            self.PGMIN_ORIGINAL[i] = g["PGERmin_pu"]
            self.PGMAX_ORIGINAL[i] = g["PGERmax_pu"]
            self.CPG_ORIGINAL[i] = g["custo_var_pu"]
            self.PGMAX_EFETIVO[i] = self.PGMAX_ORIGINAL[i]
            self.PGMIN_EFETIVO[i] = self.PGMIN_ORIGINAL[i]

    def ProcessaLOAD(self):
        """Processa cargas em PU"""
        self.PLOAD = np.zeros(self.NBAR)
        self.QLOAD = np.zeros(self.NBAR)
        
        # Carregar das barras
        for b in self.barras:
            idx = self.idx_map[b["ID_Barra"]]
            self.PLOAD[idx] += b.get("P_carga_pu", 0.0)
            self.QLOAD[idx] += b.get("Q_carga_pu", 0.0)
        
        # Adicionar das demandas
        for d in self.demandas_data:
            idx = self.idx_map[d["ID_Barra"]]
            self.PLOAD[idx] += d.get("PLOAD_pu", 0.0)
            self.QLOAD[idx] += d.get("QLOAD_pu", 0.0)

    def atualizar_perfis_horarios(self, perfil_carga, perfil_eolica, hora):
        """
        Atualiza carga e geração eólica para uma hora específica
        """
        # Salvar carga base se for a primeira hora
        if not hasattr(self, 'PLOAD_BASE'):
            self.PLOAD_BASE = self.PLOAD.copy()
            self.QLOAD_BASE = self.QLOAD.copy()
        
        # Atualizar carga em TODAS as barras proporcionalmente
        for i in range(self.NBAR):
            if self.PLOAD_BASE[i] > 0:
                self.PLOAD[i] = self.PLOAD_BASE[i] * perfil_carga[hora]
            if self.QLOAD_BASE[i] > 0:
                self.QLOAD[i] = self.QLOAD_BASE[i] * perfil_carga[hora]
        
        # Atualizar capacidade eólica
        for idx in self.BAR_GWD:  # Agora são índices dos geradores GWD
            if idx < len(self.PGMAX_EFETIVO):
                self.PGMAX_EFETIVO[idx] = self.PGMAX_ORIGINAL[idx] * perfil_eolica[hora]
        
        # Atualizar no PGMAX_CURTAILMENT (se existir)
        if hasattr(self, 'PGMAX_CURTAILMENT'):
            for i, gwd_idx in enumerate(self.BAR_GWD):
                if gwd_idx < len(self.PGMAX_EFETIVO):
                    self.PGMAX_CURTAILMENT[i] = self.PGMAX_EFETIVO[gwd_idx]

    def ProcessaDEF_GWD(self):
        """Processa deficit e curtailment em PU"""
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        barras_com_gerador = set(self.BARPG_ORIGINAL)
        self.barras_PQ_sem_gerador = [b for b in barras_PQ 
                                    if self.idx_map[b["ID_Barra"]] not in barras_com_gerador]
        
        # Curtailment (corte de vento) - todos os GWD
        self.NGER_CURTAILMENT = len(self.BAR_GWD)
        self.BARPG_CURTAILMENT = []
        self.PGMIN_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        self.PGMAX_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        self.CPG_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        
        for i, gwd_idx in enumerate(self.BAR_GWD):
            if gwd_idx < len(self.geradores_data):
                ger = self.geradores_data[gwd_idx]
                barra_idx = self.idx_map[ger["ID_Barra"]]
                self.BARPG_CURTAILMENT.append(barra_idx)
                self.PGMIN_CURTAILMENT[i] = 0.0
                self.PGMAX_CURTAILMENT[i] = self.PGMAX_EFETIVO[gwd_idx]
                # Custo de curtailment em USD/pu
                self.CPG_CURTAILMENT[i] = ger.get("custo_curtailment_pu", 1000.0)
        
        # Déficit (corte de carga) - barras PQ sem geradores
        self.NGER_DEFICIT = len(self.barras_PQ_sem_gerador)
        self.BARPG_DEFICIT = []
        self.PGMIN_DEFICIT = np.zeros(self.NGER_DEFICIT)
        self.PGMAX_DEFICIT = np.zeros(self.NGER_DEFICIT)
        self.CPG_DEFICIT = np.zeros(self.NGER_DEFICIT)
        
        for i, b in enumerate(self.barras_PQ_sem_gerador):
            idx = self.idx_map[b["ID_Barra"]]
            self.BARPG_DEFICIT.append(idx)
            self.PGMIN_DEFICIT[i] = 0.0
            # Déficit pode ser até 2x a carga da barra
            self.PGMAX_DEFICIT[i] = self.PLOAD[idx] * 2 if self.PLOAD[idx] > 0 else 0.1
            # Custo muito alto para déficit (penalidade)
            self.CPG_DEFICIT[i] = 5000.0 * self.SB  # USD/pu (alta penalidade)
        
        # Combinar todos os geradores
        self.NGER = self.NGER_ORIGINAL + self.NGER_CURTAILMENT + self.NGER_DEFICIT
        self.BARPG = self.BARPG_ORIGINAL + self.BARPG_CURTAILMENT + self.BARPG_DEFICIT
        self.PGMIN = np.concatenate([self.PGMIN_EFETIVO, self.PGMIN_CURTAILMENT, self.PGMIN_DEFICIT])
        self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, self.PGMAX_CURTAILMENT, self.PGMAX_DEFICIT])
        self.CPG = np.concatenate([self.CPG_ORIGINAL, self.CPG_CURTAILMENT, self.CPG_DEFICIT])
        
        # Mapeamento de tipos
        self.GER_TIPOS_COMBINADO = self.GER_TIPOS + ["CURTAILMENT"] * self.NGER_CURTAILMENT + ["DEFICIT"] * self.NGER_DEFICIT

def SolvePL(sistema, considerar_perdas=False, tol=1e-5, max_iter=20):
    """OPF DC com restrições de transmissão - COM CURTAILMENT CORRETO (PU)"""
    try:
        # Inicializar perdas
        perdas_barra = np.zeros(sistema.NBAR)
        perdas_anteriores = np.zeros(sistema.NBAR)
        
        for iteracao in range(max_iter):
            model = pyo.ConcreteModel()
            
            # Conjuntos
            model.GER = pyo.Set(initialize=range(sistema.NGER))
            model.BAR = pyo.Set(initialize=range(sistema.NBAR))
            model.LIN = pyo.Set(initialize=range(sistema.NLIN))
            
            # Geradores GWD para curtailment
            model.GWD = pyo.Set(initialize=sistema.BAR_GWD)  # Índices dos geradores GWD
            
            # Variáveis
            model.PG = pyo.Var(model.GER, within=pyo.NonNegativeReals)
            model.ANG = pyo.Var(model.BAR, within=pyo.Reals, bounds=(-3.14, 3.14))
            model.DEFICIT = pyo.Var(model.BAR, within=pyo.NonNegativeReals)
            
            # NOVA VARIÁVEL: Geração eólica utilizada (não curtailment)
            model.PG_WIND_USED = pyo.Var(model.GWD, within=pyo.NonNegativeReals)
            
            # Variável auxiliar para curtailment (derivada)
            model.CURTAILMENT = pyo.Var(model.GWD, within=pyo.NonNegativeReals)
            
            model.FLUXO = pyo.Var(model.LIN, within=pyo.Reals)
            
            # Fixar ângulo da barra slack
            model.ANG[sistema.slack_idx].fix(0.0)
            
            # LIMITES DE GERAÇÃO PARA TODOS OS GERADORES (em PU)
            def C_LimiteGER(m, g):
                if g in m.GWD:
                    # Para geradores eólicos, PG[g] = capacidade disponível
                    return m.PG[g] == sistema.PGMAX_EFETIVO[g]
                else:
                    # Para outros geradores, limites normais
                    return (sistema.PGMIN[g], m.PG[g], sistema.PGMAX[g])
            model.C_LimiteGER = pyo.Constraint(model.GER, rule=C_LimiteGER)
            
            # RELAÇÃO ENTRE GERAÇÃO EÓLICA DISPONÍVEL E UTILIZADA
            def C_WindBalance(m, g):
                # A geração eólica utilizada não pode exceder a disponível
                return m.PG_WIND_USED[g] <= m.PG[g]
            model.C_WindBalance = pyo.Constraint(model.GWD, rule=C_WindBalance)
            
            # DEFINIÇÃO DO CURTAILMENT
            def C_DefineCurtailment(m, g):
                # Curtailment = Geração disponível - Geração utilizada
                return m.CURTAILMENT[g] == m.PG[g] - m.PG_WIND_USED[g]
            model.C_DefineCurtailment = pyo.Constraint(model.GWD, rule=C_DefineCurtailment)
            
            # FLUXO NAS LINHAS (em PU)
            def C_DefinicaoFluxo(m, e):
                i = sistema.line_fr[e]
                j = sistema.line_to[e]
                return m.FLUXO[e] == (m.ANG[i] - m.ANG[j]) / sistema.x_line[e]
            model.C_DefinicaoFluxo = pyo.Constraint(model.LIN, rule=C_DefinicaoFluxo)
            
            # LIMITES DE FLUXO (em PU)
            def C_LimiteFluxoPos(m, e):
                return m.FLUXO[e] <= sistema.FLIM[e]
            model.C_LimiteFluxoPos = pyo.Constraint(model.LIN, rule=C_LimiteFluxoPos)
            
            def C_LimiteFluxoNeg(m, e):
                return m.FLUXO[e] >= -sistema.FLIM[e]
            model.C_LimiteFluxoNeg = pyo.Constraint(model.LIN, rule=C_LimiteFluxoNeg)
            
            # BALANÇO DE POTÊNCIA - VERSÃO CORRIGIDA (em PU)
            def C_BalancoPotencia(m, i):
                geracao_total = 0.0
                
                # Adicionar geração de todos os geradores
                for g in m.GER:
                    if sistema.BARPG[g] == i:
                        if g in m.GWD:
                            # Usar a geração eólica UTILIZADA (não a disponível)
                            geracao_total += m.PG_WIND_USED[g]
                        else:
                            geracao_total += m.PG[g]
                
                deficit_barra = m.DEFICIT[i]
                
                # Fluxos 
                fluxo_liquido = 0.0
                for e in m.LIN:
                    if sistema.line_fr[e] == i:
                        fluxo_liquido += m.FLUXO[e]
                    elif sistema.line_to[e] == i:
                        fluxo_liquido -= m.FLUXO[e]
                
                # Perdas nas linhas (em PU)
                if considerar_perdas:
                    perdas = perdas_barra[i]
                else:
                    perdas = 0.0
                
                return geracao_total + deficit_barra - fluxo_liquido - perdas == sistema.PLOAD[i]

            model.C_BalancoPotencia = pyo.Constraint(model.BAR, rule=C_BalancoPotencia)
            
            # FUNÇÃO OBJETIVO - SIMPLIFICADA (em USD)
            def FOB(m):
                # Custo de geração (exceto eólica que tem custo zero)
                custo_geracao = 0.0
                for g in m.GER:
                    if g not in m.GWD:  # Apenas geradores não-eólicos
                        custo_geracao += sistema.CPG[g] * m.PG[g]
                
                # Custo de curtailment (penalidade alta para evitar corte)
                custo_curtailment = 0.0
                for g in m.GWD:
                    # Usar custo fixo alto para penalizar curtailment
                    custo_curtailment += 1000.0 * m.CURTAILMENT[g]
                
                # Custo de déficit (penalidade muito alta)
                custo_deficit = 0.0
                for b in m.BAR:
                    # Custo do déficit (5000 USD/pu)
                    custo_deficit += 5000.0 * m.DEFICIT[b]
                
                return custo_geracao + custo_curtailment + custo_deficit 
            
            model.FOB = pyo.Objective(rule=FOB, sense=pyo.minimize)
            
            # RESOLVER
            solver = pyo.SolverFactory('glpk')
            results = solver.solve(model, tee=False)

            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                print(f"OPF não convergiu na iteração {iteracao}: {results.solver.termination_condition}")
                return ResultadoOPF(False, 
                                   [0.0] * sistema.NGER, 
                                   [0.0] * sistema.NBAR, 
                                   0.0,
                                   0.0,
                                   [0.0] * sistema.NLIN,
                                   0.0, 0.0, 0.0)
            
            # Calcular novas perdas se considerar_perdas=True
            if considerar_perdas:
                # Extrair fluxos
                fluxos_val = [pyo.value(model.FLUXO[e]) for e in range(sistema.NLIN)]
                
                # Calcular perdas por barra (em PU)
                novas_perdas_barra = np.zeros(sistema.NBAR)
                for e in range(sistema.NLIN):
                    i = sistema.line_fr[e]
                    j = sistema.line_to[e]
                    # Perdas = R * I² = R * (Fluxo/1.0)² (simplificado)
                    perdas_linha = sistema.r_line[e] * (fluxos_val[e] ** 2)
                    # Distribuir perdas igualmente entre as barras
                    novas_perdas_barra[i] += perdas_linha / 2
                    novas_perdas_barra[j] += perdas_linha / 2
                
                # Verificar convergência
                diferenca = np.max(np.abs(novas_perdas_barra - perdas_anteriores))
                if diferenca < tol:
                    print(f"Convergência atingida na iteração {iteracao} com diferença {diferenca}")
                    perdas_barra = novas_perdas_barra
                    break
                
                perdas_anteriores = perdas_barra.copy()
                perdas_barra = novas_perdas_barra
            
            else:
                # Não considerar perdas, apenas uma iteração
                break
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Extrair resultados
            PG_val = [pyo.value(model.PG[g]) for g in model.GER]
            ANG_val = [pyo.value(model.ANG[b]) for b in model.BAR]
            DEFICIT_val = [pyo.value(model.DEFICIT[b]) for b in model.BAR]
            
            # Curtailment
            curtailment_total = 0.0
            for g in model.GWD:
                curtailment_total += pyo.value(model.CURTAILMENT[g])
            
            deficit_total = sum(DEFICIT_val)
            
            # Custo
            custo_total = pyo.value(model.FOB)
            
            # Fluxos
            fluxos_val = [pyo.value(model.FLUXO[e]) for e in range(sistema.NLIN)]
            
            # Calcular CMO (custo marginal de operação) - dual da restrição de balanço
            cmo_total = 0.0
            if hasattr(model, 'dual'):
                cmo_total = model.dual[model.C_BalancoPotencia[sistema.slack_idx]] if considerar_perdas else 0.0
            
            # Calcular perdas totais (em PU)
            perdas_totais = np.sum(perdas_barra) if considerar_perdas else 0.0
            
            return ResultadoOPF(True, PG_val, ANG_val, custo_total, cmo_total,
                              fluxos_val, deficit_total, curtailment_total,
                              perdas_totais)
        else:
            print(f"OPF não convergiu: {results.solver.termination_condition}")
            return ResultadoOPF(False, 
                               [0.0] * sistema.NGER, 
                               [0.0] * sistema.NBAR, 
                               0.0,
                               0.0,
                               [0.0] * sistema.NLIN,
                               0.0, 0.0, 0.0)
    
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
                           0.0, 0.0, 0.0)

import sqlite3
import json
from datetime import datetime

def calcular_custo_operacao_24h(sistema, perfil_carga, perfil_eolica):
    """
    Calcula custo de operação para 24 horas - VERSÃO PU
    """
    custo_total = 0
    resultados_horarios = []
    
    # Criar uma cópia do sistema para não modificar o original
    sistema_hora = copy.deepcopy(sistema)
    
    # Processar DEF_GWD na cópia (IMPORTANTE!)
    sistema_hora.ProcessaDEF_GWD()
    
    print(f"\n{'='*60}")
    print("SIMULAÇÃO 24 HORAS (PU)")
    print(f"{'='*60}")
    
    # Criar conexão SQLite
    conn = sqlite3.connect('DATA/SMA/resultados_PL.db')
    cursor = conn.cursor()
    
    # Criar tabela melhorada
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resultados_PL (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        hora INTEGER,
        sucesso INTEGER,
        custo REAL,
        curtailment REAL,
        deficit REAL,
        perdas REAL,
        carga_total REAL,
        eolica_disponivel REAL,
        eolica_utilizada REAL,
        custo_geracao REAL,
        custo_curtailment REAL,
        custo_deficit REAL,
        pg_json TEXT,
        ang_json TEXT,
        fluxos_json TEXT,
        dados_barras_json TEXT
    )
    ''')
    
    # Data da execução
    data_exec = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for hora in range(24):
        print(f"\nHORA {hora:02d}:00")
        
        # Atualizar perfis para a hora
        sistema_hora.atualizar_perfis_horarios(perfil_carga, perfil_eolica, hora)
        
        # Mostrar dados atualizados
        carga_total = np.sum(sistema_hora.PLOAD)
        print(f"  Carga total do sistema: {carga_total:.3f} pu ({carga_total * sistema_hora.SB:.1f} MW)")
        
        # Encontrar capacidade eólica atualizada
        capacidade_eolica = 0
        for idx in sistema_hora.BAR_GWD:
            if idx < len(sistema_hora.PGMAX_EFETIVO):
                capacidade_eolica += sistema_hora.PGMAX_EFETIVO[idx]
        print(f"  Capacidade eólica disponível: {capacidade_eolica:.3f} pu ({capacidade_eolica * sistema_hora.SB:.1f} MW)")
        
        # Resolver OPF
        print(f"  Resolvendo OPF DC...")
        resultado = SolvePL(sistema_hora, considerar_perdas=True)
        
        if resultado.sucesso:
            custo_total += resultado.custo_total
            
            # Calcular eólica utilizada
            eolica_utilizada = 0
            for g_idx in range(sistema_hora.NGER_ORIGINAL):
                if sistema_hora.GER_TIPOS[g_idx] == "GWD":
                    eolica_utilizada += resultado.PG[g_idx]
            
            # Salvar dados detalhados das barras
            dados_barras = []
            for i in range(sistema_hora.NBAR):
                geracao_barra = 0
                for g_idx, barra_idx in enumerate(sistema_hora.BARPG):
                    if barra_idx == i and g_idx < sistema_hora.NGER_ORIGINAL:
                        geracao_barra += resultado.PG[g_idx]
                
                dados_barras.append({
                    'barra': sistema_hora.indice_para_barra[i],
                    'carga_P': sistema_hora.PLOAD[i],
                    'carga_Q': sistema_hora.QLOAD[i],
                    'geracao_P': geracao_barra,
                    'deficit': resultado.deficit_total if i == sistema_hora.slack_idx else 0,  # Simplificado
                    'tensao_ang': resultado.ANG[i] * 180 / np.pi if i < len(resultado.ANG) else 0
                })
            
            # Salvar no SQLite
            cursor.execute('''
            INSERT INTO resultados_PL 
            (timestamp, hora, sucesso, custo, curtailment, deficit, perdas, 
             carga_total, eolica_disponivel, eolica_utilizada,
             custo_geracao, custo_curtailment, custo_deficit,
             pg_json, ang_json, fluxos_json, dados_barras_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_exec,
                hora,
                1,
                float(resultado.custo_total),
                float(resultado.curtailment_total),
                float(resultado.deficit_total),
                float(resultado.perdas),
                float(carga_total),
                float(capacidade_eolica),
                float(eolica_utilizada),
                float(resultado.custo_total),  # Simplificado
                float(resultado.curtailment_total * 100),  # Penalidade
                float(resultado.deficit_total * 5000),  # Penalidade
                json.dumps([float(x) for x in resultado.PG]),
                json.dumps([float(x) for x in resultado.ANG]),
                json.dumps([float(x) for x in resultado.fluxos]),
                json.dumps(dados_barras)
            ))
            
            resultados_horarios.append({
                'hora': hora,
                'custo': resultado.custo_total,
                'curtailment': resultado.curtailment_total,
                'deficit': resultado.deficit_total,
                'perdas': resultado.perdas,
                'carga_total': carga_total,
                'eolica_disponivel': capacidade_eolica,
                'eolica_utilizada': eolica_utilizada
            })
            
            print(f"    Custo: ${resultado.custo_total:.2f}")
            print(f"    Déficit: {resultado.deficit_total:.3f} pu ({resultado.deficit_total * sistema_hora.SB:.1f} MW)")
            print(f"    Curtailment: {resultado.curtailment_total:.3f} pu ({resultado.curtailment_total * sistema_hora.SB:.1f} MW)")
            print(f"    Perdas: {resultado.perdas:.3f} pu ({resultado.perdas * sistema_hora.SB:.1f} MW)")
        else:
            # Salvar falha no SQLite
            cursor.execute('''
            INSERT INTO resultados_PL (timestamp, hora, sucesso)
            VALUES (?, ?, ?)
            ''', (data_exec, hora, 0))
            
            print(f"    FALHA no OPF!")
            # Retornar custo muito alto se falhar
            conn.commit()
            conn.close()
            return float('inf'), []
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"RESUMO 24 HORAS")
    print(f"{'='*60}")
    print(f"Custo total 24h: ${custo_total:.2f}")
    print(f"Custo médio por hora: ${custo_total/24:.2f}")
    print(f"✅ Dados salvos em 'resultados_PL.db' na tabela 'resultados_PL'")
    
    # Calcular totais
    if resultados_horarios:
        total_deficit = sum(r['deficit'] for r in resultados_horarios)
        total_curtailment = sum(r['curtailment'] for r in resultados_horarios)
        total_perdas = sum(r['perdas'] for r in resultados_horarios)
        
        print(f"Déficit total: {total_deficit:.3f} pu ({total_deficit * sistema_hora.SB:.1f} MW)")
        print(f"Curtailment total: {total_curtailment:.3f} pu ({total_curtailment * sistema_hora.SB:.1f} MW)")
        print(f"Perdas totais: {total_perdas:.3f} pu ({total_perdas * sistema_hora.SB:.1f} MW)")
    
    return custo_total, resultados_horarios

class PlanejamentoTransmissao:
    def __init__(self, json_file_path):
        self.sistema_base = SistemaTransmissao(json_file_path)
        self.historico = []
        
    def criar_perfis_horarios(self):
        """
        Cria perfis típicos de carga e geração eólica para 24 horas
        """
        # Perfil de carga (normalizado)
        self.perfil_carga = np.random.uniform(0.5, 1, 24) * [
            1.0, 0.6, 0.5, 0.5, 0.6, 0.8,  # 00-05h
            1.0, 1.2, 1.3, 1.2, 1.5, 1.2,  # 06-11h
            0.9, 0.8, 0.8, 0.9, 1.5, 1.6,  # 12-17h
            1.3, 0.8, 0.7, 0.9, 0.8, 0.7   # 18-23h
        ]
        
        # Perfil eólica (normalizado)
        self.perfil_eolica = np.random.uniform(0.5, 1, 24)*[
            0.5, 0.3, 0.7, 0.6, 0.5, 0.4,  # 00-05h
            0.3, 0.2, 0.3, 1.4, 0.6, 0.8,  # 06-11h
            0.9, 1.0, 0.9, 0.8, 0.7, 0.6,  # 12-17h
            0.5, 0.4, 0.3, 0.2, 0.1, 0.1   # 18-23h
        ]
        
        print(f"Perfil de carga criado (normalizado)")
        print(f"Perfil eólica criado (normalizado)")
    
    def plotar_resultados(self, resultados):
        """
        Plota resultados da operação
        """
        if not resultados:
            return
            
        horas = [res['hora'] for res in resultados]
        custos = [res['custo'] for res in resultados]
        curtailments = [res['curtailment'] * self.sistema_base.SB for res in resultados]
        deficits = [res['deficit'] * self.sistema_base.SB for res in resultados]
        perdas = [res['perdas'] * self.sistema_base.SB for res in resultados]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Custo por hora
        ax1.bar(horas, custos, alpha=0.7)
        ax1.set_xlabel('Hora do Dia')
        ax1.set_ylabel('Custo (USD)')
        ax1.set_title('Custo de Operação por Hora')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Gráfico 2: Curtailment
        ax2.bar(horas, curtailments, alpha=0.7, color='orange')
        ax2.set_xlabel('Hora do Dia')
        ax2.set_ylabel('Curtailment (MW)')
        ax2.set_title('Curtailment de Geração Eólica por Hora')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        
        # Gráfico 3: Déficit
        ax3.bar(horas, deficits, alpha=0.7, color='red')
        ax3.set_xlabel('Hora do Dia')
        ax3.set_ylabel('Déficit (MW)')
        ax3.set_title('Déficit de Carga por Hora')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))
        
        # Gráfico 4: Perdas
        ax4.bar(horas, perdas, alpha=0.7, color='green')
        ax4.set_xlabel('Hora do Dia')
        ax4.set_ylabel('Perdas (MW)')
        ax4.set_title('Perdas nas Linhas por Hora')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 2))
        
        #plt.tight_layout()
        #plt.savefig('DATA/SMA/resultados_planejamento_PL.png', dpi=150)
        #plt.show()
    
    def executar_planejamento(self):
        """
        Executa todo o processo de planejamento
        """
        print(f"\n{'='*60}")
        print("PLANEJAMENTO DE TRANSMISSÃO (PU)")
        print(f"{'='*60}")
        
        # Etapa 1: Criar perfis horários
        self.criar_perfis_horarios()
        
        # Etapa 2: Calcular custo de operação para 24h
        custo_operacao, resultados = calcular_custo_operacao_24h(
            self.sistema_base,
            self.perfil_carga,
            self.perfil_eolica
        )
        
        # Etapa 3: Plotar resultados
        print("\n3. Plotando resultados...")
        self.plotar_resultados(resultados)
        
        return custo_operacao, resultados



# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    try:
        # Inicializar planejamento
        print("Inicializando sistema...")
        planejador = PlanejamentoTransmissao("DATA/input/B6L8_BASE.json")
        
        # Mostrar informações do sistema
        print(f"\nSistema carregado:")
        print(f"  Potência base (S_base): {planejador.sistema_base.SB} MVA")
        print(f"  Número de barras: {planejador.sistema_base.NBAR}")
        print(f"  Número de linhas: {planejador.sistema_base.NLIN}")
        print(f"  Número de geradores originais: {planejador.sistema_base.NGER_ORIGINAL}")
        print(f"  Geradores eólicos (GWD): {planejador.sistema_base.BAR_GWD}")
        print(f"  Carga total inicial: {np.sum(planejador.sistema_base.PLOAD):.3f} pu ({np.sum(planejador.sistema_base.PLOAD) * planejador.sistema_base.SB:.1f} MW)")
        
        # Executar planejamento completo
        custo_total, resultados = planejador.executar_planejamento()
        
        # Resumo final
        print(f"\n{'='*60}")
        print("RESUMO FINAL")
        print(f"{'='*60}")
        
        if custo_total != float('inf'):
            print(f"\nCusto total 24h: ${custo_total:.2f}")
            print(f"Custo médio por hora: ${custo_total/24:.2f}")
            
            if resultados:
                total_deficit = sum(r['deficit'] for r in resultados) * planejador.sistema_base.SB
                total_curtailment = sum(r['curtailment'] for r in resultados) * planejador.sistema_base.SB
                total_perdas = sum(r['perdas'] for r in resultados) * planejador.sistema_base.SB
                
                print(f"\nTotais em MW (24 horas):")
                print(f"  Déficit total: {total_deficit:.1f} MW")
                print(f"  Curtailment total: {total_curtailment:.1f} MW")
                print(f"  Perdas totais: {total_perdas:.1f} MW")
                
                # Análise por período do dia
                print(f"\nAnálise por período do dia:")
                manha = sum(r['custo'] for r in resultados if 6 <= r['hora'] < 12)
                tarde = sum(r['custo'] for r in resultados if 12 <= r['hora'] < 18)
                noite = sum(r['custo'] for r in resultados if 18 <= r['hora'] < 24)
                madrugada = sum(r['custo'] for r in resultados if 0 <= r['hora'] < 6)
                
                print(f"  Madrugada (00-06h): ${madrugada:.2f}")
                print(f"  Manhã (06-12h): ${manha:.2f}")
                print(f"  Tarde (12-18h): ${tarde:.2f}")
                print(f"  Noite (18-24h): ${noite:.2f}")
        else:
            print("\n A simulação falhou em alguma hora!")
        
    except Exception as e:
        print(f"\n Erro durante o planejamento: {e}")
        import traceback
        traceback.print_exc()