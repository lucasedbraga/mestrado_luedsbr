import json
import sqlite3
from datetime import datetime
import random
import string
import numpy as np
import warnings
from pyomo.environ import *
warnings.filterwarnings('ignore')

# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class SistemaEletrico:
    def __init__(self, barras, geradores, demandas, linhas, contingencias,
                 idx_map, NBAR, NLIN, NGER, slack_idx,
                 line_fr, line_to, y_line, FLIM,
                 BARPG, PGMIN, PGMAX, CPG,
                 PLOAD, SB, ID_EXECUCAO, Bbus_base):
        self.barras = barras
        self.geradores = geradores
        self.demandas = demandas
        self.linhas = linhas
        self.contingencias = contingencias
        self.idx_map = idx_map
        self.NBAR = NBAR
        self.NLIN = NLIN
        self.NGER = NGER
        self.slack_idx = slack_idx
        self.line_fr = line_fr
        self.line_to = line_to
        self.y_line = y_line
        self.FLIM = FLIM
        self.BARPG = BARPG
        self.PGMIN = PGMIN
        self.PGMAX = PGMAX
        self.CPG = CPG
        self.PLOAD = PLOAD
        self.SB = SB
        self.ID_EXECUCAO = ID_EXECUCAO
        self.Bbus_base = Bbus_base

class ResultadoOPF:
    def __init__(self, convergiu, PG, ANG, lambda_, custo, fluxos, deficit_total=0.0, 
                 curtailment_total=0.0, curtailment_por_gerador=None, dual_power_balance=None):
        self.convergiu = convergiu
        self.PG = PG
        self.ANG = ANG
        self.lambda_ = lambda_
        self.custo = custo
        self.fluxos = fluxos
        self.deficit_total = deficit_total
        self.curtailment_total = curtailment_total
        self.curtailment_por_gerador = curtailment_por_gerador if curtailment_por_gerador is not None else []
        self.dual_power_balance = dual_power_balance if dual_power_balance is not None else []

# ==============================================================================
# BANCO DE DADOS - TABELAS COMPLETAMENTE REFEITAS
# ==============================================================================

def criar_tabelas():
    conn = sqlite3.connect("resultados_opf_contingencias.db")
    cursor = conn.cursor()
    
    # Tabela de execuções (mantida)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS execucoes (
            id_execucao TEXT PRIMARY KEY,
            sistema TEXT,
            data_execucao DATETIME,
            status TEXT,
            contador_cenario INTEGER
        )
    """)
    
    # TABELAS PARA REMOÇÃO DE LINHA
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS remocao_custos_detalhados (
            id_custo INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            linha_removida TEXT,
            custo_operacao REAL,
            custo_marginal_total REAL,
            custo_marginal_operacao REAL,
            custo_curtailment REAL,
            custo_deficit REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS remocao_linha_contingencias (
            id_cenario INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            linha_removida TEXT,
            total_geracao_pu REAL,
            total_carga_pu REAL,
            total_curtailment_pu REAL,
            total_deficit_pu REAL,
            custo_total_usd_h REAL,
            lambda_slack REAL,
            convergiu BOOLEAN,
            descricao TEXT,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS remocao_linha_geradores (
            id_geracao INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_gerador TEXT,
            tipo TEXT,
            geracao_pu REAL,
            curtailment_pu REAL,
            custo_marginal_usd_mwh REAL,
            barra INTEGER,
            dual_geracao REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS remocao_linha_barras (
            id_barra_cenario INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_barra INTEGER,
            tipo TEXT,
            deficit_pu REAL,
            lambda_nodal REAL,
            dual_power_balance REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS remocao_linha_linhas (
            id_linha_cenario INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_linha TEXT,
            fluxo_pu REAL,
            limite_pu REAL,
            utilizacao_percentual REAL,
            id_barra_origem INTEGER,
            id_barra_destino INTEGER,
            dual_fluxo_pos REAL,
            dual_fluxo_neg REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    # TABELAS PARA DUPLICAÇÃO DE LINHA
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicacao_custos_detalhados (
            id_custo INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            linha_original TEXT,
            custo_operacao REAL,
            custo_marginal_total REAL,
            custo_marginal_operacao REAL,
            custo_curtailment REAL,
            custo_deficit REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicacao_linha_contingencias (
            id_cenario INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            linha_original TEXT,
            total_geracao_pu REAL,
            total_carga_pu REAL,
            total_curtailment_pu REAL,
            total_deficit_pu REAL,
            custo_total_usd_h REAL,
            lambda_slack REAL,
            convergiu BOOLEAN,
            descricao TEXT,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicacao_linha_geradores (
            id_geracao INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_gerador TEXT,
            tipo TEXT,
            geracao_pu REAL,
            curtailment_pu REAL,
            custo_marginal_usd_mwh REAL,
            barra INTEGER,
            dual_geracao REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicacao_linha_barras (
            id_barra_cenario INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_barra INTEGER,
            tipo TEXT,
            deficit_pu REAL,
            lambda_nodal REAL,
            dual_power_balance REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicacao_linha_linhas (
            id_linha_cenario INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_linha TEXT,
            fluxo_pu REAL,
            limite_pu REAL,
            utilizacao_percentual REAL,
            id_barra_origem INTEGER,
            id_barra_destino INTEGER,
            dual_fluxo_pos REAL,
            dual_fluxo_neg REAL,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    conn.commit()
    conn.close()
    #print("✅ Todas as tabelas do banco de dados criadas/atualizadas com sucesso!")

# ==============================================================================
# FUNÇÃO EXPORTAR_RESULTADOS - COMPLETAMENTE REFEITA
# ==============================================================================

def exportar_resultados(conn, sistema, cenario, linha_afetada, tipo_operacao, resultado, descricao):
    cursor = conn.cursor()
    
    # Calcular totais
    total_pg = sum(resultado.PG)
    total_pl = sum(sistema.PLOAD)
    total_deficit = resultado.deficit_total
    total_curtailment = resultado.curtailment_total
    
    # Calcular custos detalhados
    custo_operacao = sum(sistema.CPG[g] * resultado.PG[g] for g in range(sistema.NGER))
    custo_marginal_total = resultado.custo  # Inclui penalidades
    custo_marginal_operacao = custo_operacao  # Apenas custo real de operação
    custo_curtailment = 1000.0 * total_curtailment  # Penalidade de curtailment
    custo_deficit = 10000.0 * total_deficit  # Penalidade de déficit
    
    # Determinar tabelas baseadas no tipo de operação
    if tipo_operacao in ["REMOCAO", "REMOCAO_ARTIFICIAL", "CONTINGENCIA_PROGRAMADA"]:
        tabela_principal = "remocao_linha_contingencias"
        tabela_custos = "remocao_custos_detalhados"
        tabela_geradores = "remocao_linha_geradores"
        tabela_barras = "remocao_linha_barras"
        tabela_linhas = "remocao_linha_linhas"
        coluna_linha = "linha_removida"
        ctg_id = f"REM-{linha_afetada}"
    elif tipo_operacao in ["DUPLICACAO", "DUPLICACAO_LINHA"]:
        tabela_principal = "duplicacao_linha_contingencias"
        tabela_custos = "duplicacao_custos_detalhados"
        tabela_geradores = "duplicacao_linha_geradores"
        tabela_barras = "duplicacao_linha_barras"
        tabela_linhas = "duplicacao_linha_linhas"
        coluna_linha = "linha_original"
        ctg_id = f"DUP-{linha_afetada}"
    else:  # Caso base
        tabela_principal = "remocao_linha_contingencias"
        tabela_custos = "remocao_custos_detalhados"
        tabela_geradores = "remocao_linha_geradores"
        tabela_barras = "remocao_linha_barras"
        tabela_linhas = "remocao_linha_linhas"
        coluna_linha = "linha_removida"
        ctg_id = "BASE"
        linha_afetada = "CASO_BASE"
    
    # 1. INSERIR NA TABELA DE CUSTOS DETALHADOS (NOVA)
    cursor.execute(f"""
        INSERT INTO {tabela_custos} 
        (id_execucao, id_contingencia, {coluna_linha}, 
         custo_operacao, custo_marginal_total, custo_marginal_operacao,
         custo_curtailment, custo_deficit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [sistema.ID_EXECUCAO, ctg_id, linha_afetada,
          custo_operacao, custo_marginal_total, custo_marginal_operacao,
          custo_curtailment, custo_deficit])
    
    # 2. INSERIR NA TABELA PRINCIPAL DE CONTINGÊNCIAS
    cursor.execute(f"""
        INSERT INTO {tabela_principal} 
        (id_execucao, id_contingencia, {coluna_linha}, total_geracao_pu, total_carga_pu, 
         total_curtailment_pu, total_deficit_pu, custo_total_usd_h, lambda_slack, 
         convergiu, descricao)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [sistema.ID_EXECUCAO, ctg_id, linha_afetada, total_pg, total_pl,
          total_curtailment, total_deficit, resultado.custo, resultado.lambda_[sistema.slack_idx],
          resultado.convergiu, descricao])
    
    # 3. EXPORTAR GERADORES
    for g in range(sistema.NGER):
        barra_idx = sistema.BARPG[g]
        tipo_ger = sistema.geradores[g]["Tipo"]
        id_gerador = sistema.geradores[g]["ID_Gerador"]
        
        pg_val = max(0.0, resultado.PG[g])
        curtailment_val = resultado.curtailment_por_gerador[g] if g < len(resultado.curtailment_por_gerador) else 0.0
        
        # Calcular variável dual aproximada para geração
        dual_geracao = 0.0
        if pg_val > sistema.PGMIN[g] + 0.001 and pg_val < sistema.PGMAX[g] - 0.001:
            dual_geracao = sistema.CPG[g]
        
        cursor.execute(f"""
            INSERT INTO {tabela_geradores} 
            (id_execucao, id_contingencia, id_gerador, tipo, 
             geracao_pu, curtailment_pu, custo_marginal_usd_mwh, barra, dual_geracao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, id_gerador, tipo_ger,
              pg_val, curtailment_val, sistema.CPG[g], barra_idx, dual_geracao])
    
    # 4. EXPORTAR DÉFICIT COMO GERADOR VIRTUAL
    if total_deficit > 0.001:
        cursor.execute(f"""
            INSERT INTO {tabela_geradores} 
            (id_execucao, id_contingencia, id_gerador, tipo, 
             geracao_pu, curtailment_pu, custo_marginal_usd_mwh, barra, dual_geracao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, "DEFICIT", "DEF",
              total_deficit, 0.0, 100000.0, -1, 100000.0])
    
    # 5. EXPORTAR CURTAILMENT TOTAL COMO GERADOR VIRTUAL
    if total_curtailment > 0.001:
        cursor.execute(f"""
            INSERT INTO {tabela_geradores} 
            (id_execucao, id_contingencia, id_gerador, tipo, 
             geracao_pu, curtailment_pu, custo_marginal_usd_mwh, barra, dual_geracao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, "CURTAILMENT_TOTAL", "CURTAIL",
              -total_curtailment, total_curtailment, 1000.0, -1, 1000.0])
    
    # 6. EXPORTAR BARRAS
    for i in range(sistema.NBAR):
        barra = sistema.barras[i]
        deficit = 0.0
        
        if total_deficit > 0.001:
            deficit = total_deficit * (sistema.PLOAD[i] / max(total_pl, 0.001))
        
        dual_power_balance = resultado.dual_power_balance[i] if i < len(resultado.dual_power_balance) else resultado.lambda_[i]
        
        cursor.execute(f"""
            INSERT INTO {tabela_barras} 
            (id_execucao, id_contingencia, id_barra, tipo, deficit_pu, lambda_nodal, dual_power_balance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, i+1, barra["tipo"], deficit, resultado.lambda_[i], dual_power_balance])
    
    # 7. EXPORTAR LINHAS
    for e in range(sistema.NLIN):
        linha = sistema.linhas[e]
        i = sistema.line_fr[e]
        j = sistema.line_to[e]
        fluxo = resultado.fluxos[e]
        utilizacao = abs(fluxo) / sistema.FLIM[e] * 100 if sistema.FLIM[e] > 0 else 0.0
        
        # Calcular variáveis duais aproximadas para limites de fluxo
        dual_fluxo_pos = 0.0
        dual_fluxo_neg = 0.0
        
        if abs(fluxo - sistema.FLIM[e]) < 0.001:
            dual_fluxo_pos = 10.0
        if abs(fluxo + sistema.FLIM[e]) < 0.001:
            dual_fluxo_neg = 10.0
        
        cursor.execute(f"""
            INSERT INTO {tabela_linhas} 
            (id_execucao, id_contingencia, id_linha, fluxo_pu, 
             limite_pu, utilizacao_percentual, id_barra_origem, id_barra_destino,
             dual_fluxo_pos, dual_fluxo_neg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, linha["ID_linha"], fluxo, 
              sistema.FLIM[e], utilizacao, sistema.line_fr[e]+1, sistema.line_to[e]+1,
              dual_fluxo_pos, dual_fluxo_neg])
    
    # 8. ATUALIZAR EXECUÇÃO COM CONTADOR DE CENÁRIO
    cursor.execute("""
        INSERT OR REPLACE INTO execucoes 
        (id_execucao, sistema, data_execucao, status, contador_cenario)
        VALUES (?, ?, datetime('now'), ?, 
               COALESCE((SELECT MAX(contador_cenario) FROM execucoes WHERE id_execucao = ?), 0) + 1)
    """, [sistema.ID_EXECUCAO, "3barras_BASE.json", "CONCLUIDO", sistema.ID_EXECUCAO])
    
    conn.commit()
    # print(f"📊 Dados exportados para: {linha_afetada} ({tipo_operacao})")
    # print(f"   💰 Custos detalhados:")
    # print(f"      Operação: ${custo_operacao:.2f}/h")
    # print(f"      Marginal Total: ${custo_marginal_total:.2f}/h") 
    # print(f"      Marginal Operação: ${custo_marginal_operacao:.2f}/h")
    # print(f"      Curtailment: ${custo_curtailment:.2f}/h")
    # print(f"      Déficit: ${custo_deficit:.2f}/h")
    # print(f"   📈 Curtailment total: {total_curtailment:.4f} pu")
    # print(f"   📉 Déficit total: {total_deficit:.4f} pu")
    
# ==============================================================================
# CÁLCULOS DE BBUS
# ==============================================================================

def calcular_Bbus_base(sistema):
    """Calcula a matriz de susceptância BASE corretamente"""
    Bbus = np.zeros((sistema.NBAR, sistema.NBAR))
    for e in range(sistema.NLIN):
        i, j = sistema.line_fr[e], sistema.line_to[e]
        y = sistema.y_line[e]
        Bbus[i,i] += y
        Bbus[j,j] += y
        Bbus[i,j] -= y
        Bbus[j,i] -= y
    return Bbus

def calcular_Bbus_linhas_ativas(sistema, linhas_ativas):
    """Calcula Bbus considerando apenas linhas ativas"""
    Bbus = np.zeros((sistema.NBAR, sistema.NBAR))
    for e in range(sistema.NLIN):
        if linhas_ativas[e]:
            i, j = sistema.line_fr[e], sistema.line_to[e]
            y = sistema.y_line[e]
            Bbus[i,i] += y
            Bbus[j,j] += y
            Bbus[i,j] -= y
            Bbus[j,i] -= y
    return Bbus

def calcular_Bbus_duplicar_linha(sistema, linha_id):
    """Calcula Bbus duplicando uma linha específica"""
    Bbus = sistema.Bbus_base.copy()
    
    # Encontrar a linha a ser duplicada
    linha_idx = next((i for i, ln in enumerate(sistema.linhas) if ln["ID_linha"] == linha_id), None)
    if linha_idx is None:
        raise ValueError(f"Linha '{linha_id}' não encontrada!")
    
    i, j = sistema.line_fr[linha_idx], sistema.line_to[linha_idx]
    y_original = sistema.y_line[linha_idx]
    
    # Duplicar a linha: adicionar susceptância equivalente
    Bbus[i,i] += y_original
    Bbus[j,j] += y_original
    Bbus[i,j] -= y_original
    Bbus[j,i] -= y_original
    
    return Bbus

# ==============================================================================
# CÁLCULO DE PREÇOS NODAIS E VARIÁVEIS DUAIS
# ==============================================================================

def calcular_precos_nodais(sistema, PG_opt, Bbus):
    """Calcula preços nodais de forma simplificada"""
    try:
        custo_minimo = min([sistema.CPG[g] for g in range(sistema.NGER) 
                           if PG_opt[g] < sistema.PGMAX[g] - 0.001])
        return np.ones(sistema.NBAR) * custo_minimo
    except:
        return np.ones(sistema.NBAR) * np.mean(sistema.CPG)

def extrair_variaveis_duais(model, sistema):
    """Extrai variáveis duais das restrições"""
    dual_power_balance = [0.0] * sistema.NBAR
    
    try:
        # Extrair dual do balanço de potência
        for i in range(sistema.NBAR):
            constraint = model.power_balance[i]
            if constraint in model.power_balance:
                # Em Pyomo, precisamos acessar os duals através do solver
                # Esta é uma aproximação - em implementação real usaríamos results
                dual_power_balance[i] = sistema.CPG[0]  # Aproximação
    except:
        pass
    
    return dual_power_balance

# ==============================================================================
# OPF PRINCIPAL - MODELO COMPLETO COM CURTAILMENT
# ==============================================================================

def resolver_opf(sistema, Bbus, linhas_ativas=None, nome_modelo="opf"):
    """OPF DC completo com curtailment, déficit e variáveis duais"""
    try:
        model = ConcreteModel()
        
        # Conjuntos
        model.G = Set(initialize=range(sistema.NGER))
        model.B = Set(initialize=range(sistema.NBAR))
        
        # CONJUNTO DE LINHAS ATIVAS - CORREÇÃO CRÍTICA!
        if linhas_ativas is None:
            indices_linhas_ativas = list(range(sistema.NLIN))
        else:
            indices_linhas_ativas = [i for i, ativa in enumerate(linhas_ativas) if ativa]
        
        model.L = Set(initialize=indices_linhas_ativas)
        
        # Identificar geradores GWD
        gwd_indices = [g for g in range(sistema.NGER) 
                      if sistema.geradores[g]["Tipo"] == "GWD"]
        model.GWD = Set(initialize=gwd_indices)
        
        # Variáveis
        model.PG = Var(model.G, within=NonNegativeReals)
        model.ANG = Var(model.B, within=Reals, bounds=(-3.14, 3.14))
        model.DEFICIT = Var(model.B, within=NonNegativeReals)
        model.CURTAILMENT = Var(model.GWD, within=NonNegativeReals)
        
        # Fixar ângulo da barra slack
        model.ANG[sistema.slack_idx].fix(0.0)
        
        # LIMITES DE GERAÇÃO
        def pg_limits_rule(m, g):
            return (sistema.PGMIN[g], m.PG[g], sistema.PGMAX[g])
        model.pg_limits = Constraint(model.G, rule=pg_limits_rule)
        
        # LIMITES DE CURTAILMENT
        def curtailment_limits_rule(m, g):
            return m.CURTAILMENT[g] <= sistema.PGMAX[g]
        model.curtailment_limits = Constraint(model.GWD, rule=curtailment_limits_rule)
        
        # PENALIDADES
        custo_maximo = max(sistema.CPG) if sistema.CPG else 1000.0
        custo_curtailment_gwd = 10 * custo_maximo
        PENALIDADE_DEFICIT = 100 * custo_maximo
        
        # RESTRIÇÃO DE BALANÇO DE POTÊNCIA
        def power_balance_rule(m, i):
            geracao_total = 0.0
            for g in m.G:
                if sistema.BARPG[g] == i:
                    if g in m.GWD:
                        geracao_total += m.PG[g] - m.CURTAILMENT[g]
                    else:
                        geracao_total += m.PG[g]
            
            deficit_barra = m.DEFICIT[i]
            fluxo_saindo = 0.0
            for j in m.B:
                fluxo_saindo += Bbus[i, j] * m.ANG[j]
            
            return geracao_total + deficit_barra - fluxo_saindo == sistema.PLOAD[i]

        model.power_balance = Constraint(model.B, rule=power_balance_rule)
        
        # RESTRIÇÕES DE FLUXO APENAS PARA LINHAS ATIVAS
        def line_flow_limits_rule(m, e):
            i = sistema.line_fr[e]
            j = sistema.line_to[e]
            fluxo = sistema.y_line[e] * (m.ANG[i] - m.ANG[j])
            return (-sistema.FLIM[e], fluxo, sistema.FLIM[e])
        
        model.line_flow_limits = Constraint(model.L, rule=line_flow_limits_rule)
        
        # FUNÇÃO OBJETIVO
        def objective_rule(m):
            custo_geracao = sum(sistema.CPG[g] * m.PG[g] for g in m.G)
            custo_curtailment = custo_curtailment_gwd * sum(m.CURTAILMENT[g] for g in m.GWD)
            custo_deficit = PENALIDADE_DEFICIT * sum(m.DEFICIT[b] for b in m.B)
            
            return custo_geracao + custo_curtailment + custo_deficit
        
        model.objective = Objective(rule=objective_rule, sense=minimize)
        
        # RESOLVER
        #model.pprint()
        solver = SolverFactory('glpk')
        results = solver.solve(model, tee=False)
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            # Extrair resultados principais
            PG_val = [value(model.PG[g]) for g in model.G]
            ANG_val = [value(model.ANG[b]) for b in model.B]
            DEFICIT_val = [value(model.DEFICIT[b]) for b in model.B]
            
            # Extrair curtailment por gerador
            curtailment_por_gerador = [0.0] * sistema.NGER
            curtailment_total = 0.0
            for g in model.GWD:
                curtailment_val = value(model.CURTAILMENT[g])
                curtailment_por_gerador[g] = curtailment_val
                curtailment_total += curtailment_val
            
            deficit_total = sum(DEFICIT_val)
            
            # Calcular custo REAL
            custo_geracao_real = sum(sistema.CPG[g] * PG_val[g] for g in range(sistema.NGER))
            custo_total = custo_geracao_real
            
            # Calcular fluxos APENAS para linhas ativas
            fluxos_val = [0.0] * sistema.NLIN
            for e in indices_linhas_ativas:
                i, j = sistema.line_fr[e], sistema.line_to[e]
                fluxos_val[e] = sistema.y_line[e] * (ANG_val[i] - ANG_val[j])
            
            # Calcular preços nodais e variáveis duais
            lambda_val = calcular_precos_nodais(sistema, PG_val, Bbus)
            dual_power_balance = extrair_variaveis_duais(model, sistema)
            
            # print(f"✅ OPF convergiu - Custo: {custo_total:.2f} USD/h")
            # print(f"   Geração: {sum(PG_val):.4f} pu, Demanda: {sum(sistema.PLOAD):.4f} pu")
            # print(f"   Curtailment: {curtailment_total:.4f} pu, Déficit: {deficit_total:.4f} pu")
            
            # # DEBUG: Mostrar geração detalhada
            # print("   📊 Geração detalhada:")
            for g in range(sistema.NGER):
                tipo_ger = sistema.geradores[g]["Tipo"]
                curtailment_info = f", Curtailment: {curtailment_por_gerador[g]:.4f} pu" if tipo_ger == "GWD" else ""
                #print(f"      {sistema.geradores[g]['ID_Gerador']} ({tipo_ger}): {PG_val[g]:.4f} pu{curtailment_info}")
            
            return ResultadoOPF(True, PG_val, ANG_val, lambda_val, custo_total, 
                              fluxos_val, deficit_total, curtailment_total,
                              curtailment_por_gerador, dual_power_balance)
        else:
            #print(f"❌ OPF não convergiu: {results.solver.termination_condition}")
            return ResultadoOPF(False, 
                               [100000.0] * sistema.NGER, 
                               [100000.0] * sistema.NBAR, 
                               [100000.0] * sistema.NBAR, 
                               100000.0, 
                               [100000.0] * sistema.NLIN)
    
    except Exception as e:
        #print(f"❌ Erro no OPF: {e}")
        import traceback
        traceback.print_exc()
        return ResultadoOPF(False, 
                           [100000.0] * sistema.NGER, 
                           [100000.0] * sistema.NBAR, 
                           [100000.0] * sistema.NBAR, 
                           100000.0, 
                           [100000.0] * sistema.NLIN)

# ==============================================================================
# CONSTRUÇÃO DO SISTEMA
# ==============================================================================

def criar_sistema(data):
    # Configurações
    SB = data["S_base"]
    VB = data["V_base"]
    ZB = (VB**2) / SB

    # Dados
    barras = data["BARRAS"]
    geradores = data["GERADORES"]
    demandas = data["DEMANDAS"]
    linhas = data["LINHAS"]
    contingencias = data.get("CONTINGENCIAS", [])
    
    # Mapeamento
    idx_map = {b["ID_Barra"]: i for i, b in enumerate(barras)}
    NBAR = len(barras)
    NLIN = len(linhas)
    NGER = len(geradores)
    
    # Slack
    slack_idx = next((i for i, b in enumerate(barras) if b["tipo"] == "Slack"), 0)
    
    # Conversão para PU das linhas
    for l in linhas:
        l["X_pu"] = l["X"] / ZB
        l["Fmax_pu"] = l["LIM_Fluxo"] / SB
    
    # Parâmetros das linhas
    line_fr = [idx_map[l["ID_Barra_Origem"]] for l in linhas]
    line_to = [idx_map[l["ID_Barra_Destino"]] for l in linhas]
    y_line = [1.0 / l["X_pu"] for l in linhas]
    FLIM = [l["Fmax_pu"] for l in linhas]
    
    # Parâmetros geradores
    BARPG = [idx_map[g["ID_Barra"]] for g in geradores]
    PGMIN = [g["PGERmin_MW"] / SB for g in geradores]
    PGMAX = [g["PGERmax_MW"] / SB for g in geradores]
    
    # CUSTOS REAIS
    CPG = []
    for g in geradores:
        custo = g["custo_var_USD_MW"]
        CPG.append(custo)
    
    # Demanda
    PLOAD = np.zeros(NBAR)
    for d in demandas:
        idx = idx_map[d["ID_Barra"]]
        PLOAD[idx] = d["PLOAD"] / SB
    
    # Criar sistema temporário para calcular Bbus_base
    sistema_temp = SistemaEletrico(
        barras, geradores, demandas, linhas, contingencias,
        idx_map, NBAR, NLIN, NGER, slack_idx,
        line_fr, line_to, y_line, FLIM,
        BARPG, PGMIN, PGMAX, CPG,
        PLOAD, SB, "", np.zeros((NBAR, NBAR))
    )
    
    # Bbus base
    Bbus_base = calcular_Bbus_base(sistema_temp)
    
    # ID execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    ID_EXECUCAO = f"{timestamp}_{random_suffix}"
    
    # print(f"✅ Sistema criado: {NBAR} barras, {NLIN} linhas, {NGER} geradores")
    # print(f"📊 Demanda total: {sum(PLOAD):.4f} pu")
    # print(f"📊 Capacidade total: {sum(PGMAX):.4f} pu")
    # print(f"💰 Custos dos geradores: {CPG}")
    
    return SistemaEletrico(
        barras, geradores, demandas, linhas, contingencias,
        idx_map, NBAR, NLIN, NGER, slack_idx,
        line_fr, line_to, y_line, FLIM,
        BARPG, PGMIN, PGMAX, CPG,
        PLOAD, SB, ID_EXECUCAO, Bbus_base
    )

# ==============================================================================
# CENÁRIOS
# ==============================================================================

def executar_caso_base(sistema, conn):
    # print("\n" + "="*60)
    # print("CASO BASE")
    # print("="*60)
    
    resultado = resolver_opf(sistema, sistema.Bbus_base, None, "caso_base")
    
    if resultado.convergiu:
        #print(f"✅ Custo base: {resultado.custo:.2f} USD/h")
        exportar_resultados(conn, sistema, "CASO_BASE", "CASO_BASE", "BASE", 
                           resultado, "Caso base com topologia original")
    else:
        print("❌ Caso base não convergiu")
    
    return resultado

def executar_remocao_artificial(sistema, conn, linha_id):
    print(f"\n── REMOÇÃO ARTIFICIAL: {linha_id}")
    
    # Encontrar índice da linha
    linha_idx = None
    for i, ln in enumerate(sistema.linhas):
        if ln["ID_linha"] == linha_id:
            linha_idx = i
            break
    
    if linha_idx is None:
        print(f"❌ Linha '{linha_id}' não encontrada")
        return None
    
    # Criar vetor de linhas ativas (todas ativas exceto a removida)
    linhas_ativas = [True] * sistema.NLIN
    linhas_ativas[linha_idx] = False
    
    Bbus_remocao = calcular_Bbus_linhas_ativas(sistema, linhas_ativas)
    resultado = resolver_opf(sistema, Bbus_remocao, linhas_ativas, f"remocao_{linha_id}")
    
    if resultado.convergiu:
        exportar_resultados(conn, sistema, "REMOCAO_ARTIFICIAL", linha_id, "REMOCAO", 
                           resultado, f"Remoção artificial da linha {linha_id}")
        #print(f"✅ Custo com remoção: {resultado.custo:.2f} USD/h")
        
        # # DEBUG ESPECIAL para remoção da linha 2-3
        # if linha_id == "2-3":
        #     print("\n🔍 DEBUG REMOÇÃO LINHA 2-3:")
        #     print(f"   Geração esperada: G1=0.9 pu, G3=0.1 pu")
        #     print(f"   Geração obtida: G1={resultado.PG[0]:.4f} pu, G3={resultado.PG[2]:.4f} pu")
        #     print(f"   Curtailment G3: {resultado.curtailment_por_gerador[2]:.4f} pu")
            
    else:
        print("❌ Remoção não convergiu")
    
    return resultado

def executar_duplicacao_linha(sistema, conn, linha_id):
    print(f"\n── DUPLICAÇÃO: {linha_id}")
    
    Bbus_dup = calcular_Bbus_duplicar_linha(sistema, linha_id)
    resultado = resolver_opf(sistema, Bbus_dup, None, f"duplicacao_{linha_id}")
    
    if resultado.convergiu:
        exportar_resultados(conn, sistema, "DUPLICACAO_LINHA", linha_id, "DUPLICACAO", 
                           resultado, f"Duplicação da linha {linha_id}")
        #print(f"✅ Custo com duplicação: {resultado.custo:.2f} USD/h")
    else:
        print("❌ Duplicação não convergiu")
    
    return resultado

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    print("🚀 ANÁLISE DE CONTINGÊNCIAS - VERSÃO COMPLETA COM CURTAILMENT E DUALS")
    
    try:
        criar_tabelas()
        conn = sqlite3.connect("resultados_opf_contingencias.db")
        
        # Carregar dados
        with open("DATA/input/3barras_BASE.json", "r") as f:
            data = json.load(f)
        
        sistema = criar_sistema(data)
        
        # 1. Caso base
        print("\n ETAPA 1: CASO BASE")
        base_result = executar_caso_base(sistema, conn)
        
        # 2. Remoções individuais
        print("\n ETAPA 2: REMOÇÕES INDIVIDUAIS")
        for linha in sistema.linhas:
            rem_result = executar_remocao_artificial(sistema, conn, linha["ID_linha"])
            
            # Comparação
            if base_result and base_result.convergiu and rem_result and rem_result.convergiu:
                diferenca = rem_result.custo - base_result.custo
                percentual = (diferenca / base_result.custo) * 100 if base_result.custo > 0 else 0
                #print(f"   📊 Variação vs base: {diferenca:.2f} USD/h ({percentual:.1f}%)")
        
        # 3. Duplicações
        print("\n ETAPA 3: DUPLICAÇÕES")
        for linha in sistema.linhas:
            dup_result = executar_duplicacao_linha(sistema, conn, linha["ID_linha"])
            
            # Comparação
            if base_result and base_result.convergiu and dup_result and dup_result.convergiu:
                diferenca = dup_result.custo - base_result.custo
                percentual = (diferenca / base_result.custo) * 100 if base_result.custo > 0 else 0
                #print(f"   📊 Variação vs base: {diferenca:.2f} USD/h ({percentual:.1f}%)")
        
        conn.close()
        
    except Exception as e:
        print(f" ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()