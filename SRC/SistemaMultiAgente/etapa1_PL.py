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
        Atualiza carga e geração eólica para uma hora específica - VERSÃO OTIMIZADA
        """
        # Salvar carga base se for a primeira hora
        if not hasattr(self, 'PLOAD_BASE'):
            self.PLOAD_BASE = self.PLOAD.copy()
        
        # Atualizar carga em TODAS as barras proporcionalmente
        for i in range(self.NBAR):
            if self.PLOAD_BASE[i] > 0:
                self.PLOAD[i] = self.PLOAD_BASE[i] * perfil_carga[hora]
        
        # Criar mapeamento barra->índice para GWD nos dados originais
        gwd_original_indices = {}
        for i, g in enumerate(self.geradores_data):
            if g["Tipo"] == "GWD":
                barra_idx = self.idx_map[g["ID_Barra"]]
                gwd_original_indices[barra_idx] = i
        
        # Criar mapeamento barra->índice para curt
        gwd_curt_indices = {}
        if hasattr(self, 'BARPG_CURTAILMENT'):
            for i, barra_idx in enumerate(self.BARPG_CURTAILMENT):
                gwd_curt_indices[barra_idx] = i
        
        # Atualizar TODOS os GWD
        for barra_idx in self.BAR_GWD:
            # Atualizar no PGMAX_EFETIVO (dados originais)
            if barra_idx in gwd_original_indices:
                idx_original = gwd_original_indices[barra_idx]
                self.PGMAX_EFETIVO[idx_original] = self.PGMAX_ORIGINAL[idx_original] * perfil_eolica[hora]
            
            # Atualizar no PGMAX_CURTAILMENT (se existir)
            if barra_idx in gwd_curt_indices:
                idx_curt = gwd_curt_indices[barra_idx]
                self.PGMAX_CURTAILMENT[idx_curt] = self.PGMAX_EFETIVO[idx_original] if 'idx_original' in locals() else 0
        
        # Reconstruir PGMAX combinado
        if hasattr(self, 'PGMAX'):
            self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, 
                                        self.PGMAX_CURTAILMENT, 
                                        self.PGMAX_DEFICIT])
    
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
                self.CPG_CURTAILMENT[i] = 100
        
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
            self.CPG_DEFICIT[i] = 5000  
        
        # Combinar todos os geradores
        self.NGER = self.NGER_ORIGINAL + self.NGER_CURTAILMENT + self.NGER_DEFICIT
        self.BARPG = self.BARPG_ORIGINAL + self.BARPG_CURTAILMENT + self.BARPG_DEFICIT
        self.PGMIN = np.concatenate([self.PGMIN_EFETIVO, self.PGMIN_CURTAILMENT, self.PGMIN_DEFICIT])
        self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, self.PGMAX_CURTAILMENT, self.PGMAX_DEFICIT])
        self.CPG = np.concatenate([self.CPG_ORIGINAL, self.CPG_CURTAILMENT, self.CPG_DEFICIT])

def SolveOPF(sistema, considerar_perdas=False, tol=1e-5, max_iter=20):
    """OPF DC com restrições de transmissão - COM CURTAILMENT CORRETO"""
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
            gwd_indices = [g for g in range(sistema.NGER_ORIGINAL) 
                        if sistema.geradores_data[g]["Tipo"] == "GWD"]
            model.GWD = pyo.Set(initialize=gwd_indices)
            
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
            
            # LIMITES DE GERAÇÃO PARA TODOS OS GERADORES
            def C_LimiteGER(m, g):
                if g in m.GWD:
                    # Para geradores eólicos, PG[g] é a geração DISPONÍVEL (não controlável)
                    # Devemos fixá-la no máximo disponível
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
            
            # BALANÇO DE POTÊNCIA - VERSÃO CORRIGIDA
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
                
                # Perdas nas linhas
                if considerar_perdas:
                    perdas = perdas_barra[i]
                else:
                    perdas = 0.0
                
                return geracao_total + deficit_barra - fluxo_liquido - perdas == sistema.PLOAD[i]

            model.C_BalancoPotencia = pyo.Constraint(model.BAR, rule=C_BalancoPotencia)
            
            # FUNÇÃO OBJETIVO - PRIORIZAR USO DA EÓLICA
            def FOB(m):
                # Custo de geração (exceto eólica que tem custo zero)
                custo_geracao = 0.0
                for g in m.GER:
                    if g not in m.GWD:  # Apenas geradores não-eólicos
                        custo_geracao += sistema.CPG[g] * m.PG[g]
                
                # Custo de curtailment (penalidade alta para evitar corte)
                # MAS: curtailment pode ser necessário se a rede não suportar
                custo_curtailment = 1000 * sum(m.CURTAILMENT[g] for g in m.GWD)
                
                # Custo de déficit (penalidade muito alta)
                custo_deficit = 5000 * sum(m.DEFICIT[b] for b in m.BAR)
                
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
                
                # Calcular perdas por barra (distribuir igualmente entre as barras da linha)
                novas_perdas_barra = np.zeros(sistema.NBAR)
                for e in range(sistema.NLIN):
                    i = sistema.line_fr[e]
                    j = sistema.line_to[e]
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
            
            # Calcular perdas totais
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
    Calcula custo de operação para 24 horas - VERSÃO CORRIGIDA E FUNCIONAL
    """
    custo_total = 0
    resultados_horarios = []
    
    # Criar uma cópia do sistema para não modificar o original
    sistema_hora = copy.deepcopy(sistema)
    
    # Processar DEF_GWD na cópia (IMPORTANTE!)
    sistema_hora.ProcessaDEF_GWD()
    
    print(f"\n{'='*60}")
    print("SIMULAÇÃO 24 HORAS")
    print(f"{'='*60}")
    
    # Criar conexão SQLite
    conn = sqlite3.connect('resultados_PL.db')
    cursor = conn.cursor()
    
    # Criar tabela simples
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
        pg_json TEXT,
        ang_json TEXT,
        fluxos_json TEXT
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
        print(f"  Carga total do sistema: {carga_total:.3f} pu")
        
        # Encontrar capacidade eólica atualizada
        capacidade_eolica = 0
        for i, g in enumerate(sistema_hora.geradores_data):
            if g["Tipo"] == "GWD":
                capacidade_eolica = sistema_hora.PGMAX_EFETIVO[i]
                break
        print(f"  Capacidade eólica disponível: {capacidade_eolica:.3f} pu")
        
        # Resolver OPF
        print(f"  Resolvendo OPF...")
        resultado = SolveOPF(sistema_hora, considerar_perdas=True)
        
        if resultado.sucesso:
            custo_total += resultado.custo_total
            
            # Salvar no SQLite
            cursor.execute('''
            INSERT INTO resultados_PL 
            (timestamp, hora, sucesso, custo, curtailment, deficit, perdas, 
             carga_total, eolica_disponivel, pg_json, ang_json, fluxos_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps([float(x) for x in resultado.PG]),
                json.dumps([float(x) for x in resultado.ANG]),
                json.dumps([float(x) for x in resultado.fluxos])
            ))
            
            resultados_horarios.append({
                'hora': hora,
                'custo': resultado.custo_total,
                'curtailment': resultado.curtailment_total,
                'deficit': resultado.deficit_total,
                'perdas': resultado.perdas
            })
            
            print(f"    Custo: ${resultado.custo_total:.2f}")
            print(f"    Déficit: {resultado.deficit_total:.3f} pu")
            print(f"    Curtailment: {resultado.curtailment_total:.3f} pu")
            print(f"    Perdas: {resultado.perdas:.3f} pu")
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
        
        print(f"Déficit total: {total_deficit:.3f} pu")
        print(f"Curtailment total: {total_curtailment:.3f} pu")
        print(f"Perdas totais: {total_perdas:.3f} pu")
    
    return custo_total, resultados_horarios

class PlanejamentoTransmissao:
    def __init__(self, json_file_path):
        self.sistema_base = SistemaTransmissao(json_file_path)
        self.historico = []
        
    def criar_perfis_horarios(self):
        """
        Cria perfis típicos de carga e geração eólica para 24 horas
        """
        # Perfil de carga 
        self.perfil_carga = [
            0.7, 0.6, 0.5, 0.5, 0.6, 0.8,  # 00-05h
            1.0, 1.2, 1.3, 1.2, 1.5, 1.2,  # 06-11h
            0.9, 0.8, 0.8, 0.9, 1.5, 1.6,  # 12-17h
            1.3, 0.8, 0.7, 0.9, 0.8, 0.7   # 18-23h
        ]
        
        # Perfil eólica 
        self.perfil_eolica = [
            0.6, 1.3, 0.7, 1.6, 0.5, 0.4,  # 00-05h
            0.3, 0.2, 0.3, 1.4, 0.6, 0.8,  # 06-11h
            0.9, 1.0, 0.9, 0.8, 0.7, 0.6,  # 12-17h
            0.5, 0.4, 0.3, 0.2, 0.1, 0.1   # 18-23h
        ]
        
        print(f"Perfil de carga criado: {self.perfil_carga}")
        print(f"Perfil eólica criado: {self.perfil_eolica}")
    
    def plotar_resultados(self, resultados):
        """
        Plota resultados da operação
        """
        if not resultados:
            return
            
        horas = [res['hora'] for res in resultados]
        custos = [res['custo'] for res in resultados]
        curtailments = [res['curtailment'] for res in resultados]
        deficits = [round(res['deficit']) for res in resultados]
        perdas = [res['perdas'] for res in resultados]
        
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
        ax2.set_ylabel('Curtailment (pu)')
        ax2.set_title('Curtailment de Geração Eólica por Hora')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        
        # Gráfico 3: Déficit
        ax3.bar(horas, deficits, alpha=0.7, color='red')
        ax3.set_xlabel('Hora do Dia')
        ax3.set_ylabel('Déficit (pu)')
        ax3.set_title('Déficit de Carga por Hora')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))
        
        # Gráfico 4: Perdas
        ax4.bar(horas, perdas, alpha=0.7, color='green')
        ax4.set_xlabel('Hora do Dia')
        ax4.set_ylabel('Perdas (pu)')
        ax4.set_title('Perdas nas Linhas por Hora')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig('resultados_planejamento.png', dpi=150)
        plt.show()
    
    def executar_planejamento(self):
        """
        Executa todo o processo de planejamento
        """
        print(f"\n{'='*60}")
        print("PLANEJAMENTO DE TRANSMISSÃO")
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
        planejador = PlanejamentoTransmissao("DATA/input/3barras_BASE.json")
        
        # Mostrar informações do sistema
        print(f"\nSistema carregado:")
        print(f"  Número de barras: {planejador.sistema_base.NBAR}")
        print(f"  Número de linhas: {planejador.sistema_base.NLIN}")
        print(f"  Número de geradores: {planejador.sistema_base.NGER}")
        print(f"  Carga total inicial: {np.sum(planejador.sistema_base.PLOAD):.3f} pu")
        
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
                print(f"  Déficit total: {total_deficit:.2f} MW")
                print(f"  Curtailment total: {total_curtailment:.2f} MW")
                print(f"  Perdas totais: {total_perdas:.2f} MW")
                
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