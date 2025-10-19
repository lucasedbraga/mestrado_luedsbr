using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP
using Clp
using LinearAlgebra
using JSON
using Distributions
using SQLite
using DataFrames
using Dates
using Random
using DBInterface

# ==============================================================================
# CONFIGURAÇÕES INICIAIS E PARÂMETROS
# ==============================================================================

NITER_MAX = 10  # Número máximo de iterações
TOL = 1e-6   # Tolerância para convergência

println("=== OPF DC ITERATIVO COM PERDAS, CURTAILMENT E CONTINGÊNCIAS ===")

# ==============================================================================
# FUNÇÃO PARA GERAR CENÁRIO ALEATÓRIO
# ==============================================================================

function gerar_cenario_aleatorio!(geradores_data, demandas_data, SB)
    println("🎲 Gerando cenário aleatório...")
    
    # 1. Variar geração eólica (GWD) - entre 20% e 100% da capacidade
    for g in geradores_data
        if g["Tipo"] == "GWD"
            capacidade_original = get(g, "PGERmax_MW_ORIGINAL", g["PGERmax_MW"])
            fator_geracao = rand(Uniform(0.2, 1.0))
            g["PGERmax_MW"] = capacidade_original * fator_geracao
            println("🌬️  Geração eólica ajustada: $(round(g["PGERmax_MW"], digits=2)) MW ($(round(fator_geracao*100, digits=1))% da capacidade)")
        end
    end
    
    # 2. Variar demanda - entre 80% e 120% da demanda base  
    for d in demandas_data
        if haskey(d, "PLOAD")
            demanda_original = get(d, "PLOAD_ORIGINAL", d["PLOAD"])
            fator_demanda = rand(Uniform(0.8, 1.2))
            d["PLOAD"] = demanda_original * fator_demanda
            println("💡 Demanda ajustada: $(round(d["PLOAD"], digits=2)) MW ($(round(fator_demanda*100, digits=1))% da base)")
        end
    end
    
    # 3. Salvar parâmetros do cenário para referência
    total_geracao_eolica = sum(g["PGERmax_MW"] for g in geradores_data if g["Tipo"] == "GWD")
    total_demanda = sum(d["PLOAD"] for d in demandas_data if haskey(d, "PLOAD"))
    
    println("📊 Resumo do cenário aleatório:")
    println("   - Geração eólica total: $(round(total_geracao_eolica, digits=2)) MW")
    println("   - Demanda total: $(round(total_demanda, digits=2)) MW")
end

# ==============================================================================
# FUNÇÃO PARA EXPORTAR CONTINGÊNCIA PARA SQLite
# ==============================================================================

function exportar_contingencia_para_sqlite(ctg_id, ctg_descricao, linhas_removidas,
                                          Pg_original, Pg_curtailment, Pg_deficit, Pg_total,
                                          ANGLE, fij, fji, perdas, final_PG,
                                          ID_EXECUCAO, barras, NGER_ORIGINAL, geradores_data,
                                          BARPG, PGMIN, PGMAX, CPG, SB, BAR_GWD, NGER_CURTAILMENT,
                                          CPG_ORIGINAL, CPG_CURTAILMENT, CPG_DEFICIT, PLOAD,
                                          linhas, FLIM, contingencias_data, MVu, MVd)
    
    arquivo_db = "resultados_opf_contingencias.db"
    
    db = SQLite.DB(arquivo_db)
    
    # ==========================================================================
    # TABELA: EXECUCOES (Metadados da execução completa)
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS execucoes (
            id_execucao TEXT PRIMARY KEY,
            data_execucao DATETIME,
            n_barras INTEGER,
            n_geradores INTEGER,
            n_linhas INTEGER,
            n_contingencias INTEGER,
            sistema TEXT,
            descricao TEXT,
            timestamp_criacao DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    NBAR = length(barras)
    NGER = length(BARPG)
    NLIN = length(linhas)
    
    # Inserir/atualizar execução
    SQLite.execute(db, """
        INSERT OR REPLACE INTO execucoes 
        (id_execucao, data_execucao, n_barras, n_geradores, n_linhas, n_contingencias, sistema, descricao)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [ID_EXECUCAO, Dates.now(), NBAR, NGER, NLIN, length(contingencias_data), 
          "Sistema $(NBAR) barras", "Análise de contingências"])
    
    # ==========================================================================
    # TABELA: CONTINGENCIAS
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS contingencias (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            descricao TEXT,
            linhas_removidas TEXT,
            total_geracao_pu REAL,
            total_carga_pu REAL,
            total_perdas_pu REAL,
            total_curtailment_pu REAL,
            total_deficit_pu REAL,
            custo_total_usd_h REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao)
        )
    """)
    
    # Calcular custos para esta contingência
    custo_original = sum(CPG_ORIGINAL[g] * final_PG[g] for g in 1:NGER_ORIGINAL)
    custo_curtailment_calc = sum(CPG_CURTAILMENT[g] * final_PG[NGER_ORIGINAL+g] for g in 1:NGER_CURTAILMENT)
    custo_deficit_calc = sum(CPG_DEFICIT[g] * final_PG[NGER_ORIGINAL+NGER_CURTAILMENT+g] for g in 1:length(CPG_DEFICIT) if NGER_ORIGINAL+NGER_CURTAILMENT+g <= length(final_PG))
    custo_total = custo_original + custo_curtailment_calc + custo_deficit_calc
    
    total_pg = sum(Pg_total)
    total_pl = sum(PLOAD)
    total_perdas_val = sum(perdas)
    total_pg_curtailment = sum(Pg_curtailment)
    total_pg_deficit = sum(Pg_deficit)
    
    SQLite.execute(db, """
        INSERT INTO contingencias 
        (id_execucao, id_contingencia, descricao, linhas_removidas, total_geracao_pu, total_carga_pu, 
         total_perdas_pu, total_curtailment_pu, total_deficit_pu, custo_total_usd_h, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [ID_EXECUCAO, ctg_id, ctg_descricao, join(linhas_removidas, ","),
          total_pg, total_pl, total_perdas_val, total_pg_curtailment, total_pg_deficit,
          custo_total, total_pg_deficit > 0.01 ? "COM_DEFICIT" : "NORMAL"])
    
    # ==========================================================================
    # TABELA: BARRAS_POR_CONTINGENCIA
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS barras_contingencia (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_barra TEXT,
            tipo TEXT,
            tensao_pu REAL,
            angulo_graus REAL,
            carga_pu REAL,
            geracao_pu REAL,
            curtailment_pu REAL,
            deficit_pu REAL,
            preco_nodal_usd_mwh REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao),
            FOREIGN KEY (id_contingencia) REFERENCES contingencias (id_contingencia)
        )
    """)
    
    for i in 1:NBAR
        barra = barras[i]
        SQLite.execute(db, """
            INSERT INTO barras_contingencia 
            (id_execucao, id_contingencia, id_barra, tipo, tensao_pu, angulo_graus, 
             carga_pu, geracao_pu, curtailment_pu, deficit_pu, preco_nodal_usd_mwh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_EXECUCAO, ctg_id, barra["ID_Barra"], barra["tipo"], 
              1.0, round(rad2deg(ANGLE[i]), digits=6),  # DC OPF - tensão fixa em 1.0
              PLOAD[i], Pg_original[i], Pg_curtailment[i], Pg_deficit[i],
              0.0])  # Preço nodal não calculado no DC
    end
    
    # ==========================================================================
    # TABELA: GERADORES_POR_CONTINGENCIA
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS geradores_contingencia (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_gerador TEXT,
            id_barra TEXT,
            tipo TEXT,
            geracao_pu REAL,
            pmin_pu REAL,
            pmax_pu REAL,
            custo_marginal_usd_mwh REAL,
            ramp_up_pu REAL,
            ramp_down_pu REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao),
            FOREIGN KEY (id_contingencia) REFERENCES contingencias (id_contingencia)
        )
    """)
    
    for g in 1:NGER
        barra_idx = BARPG[g]
        id_barra = barras[barra_idx]["ID_Barra"]
        
        if g <= length(geradores_data)
            gerador = geradores_data[g]
            tipo = gerador["Tipo"]
            id_gerador = gerador["ID_Gerador"]
            ramp_up = get(gerador, "ramp_up_MW_h", 0.0) / SB
            ramp_down = get(gerador, "ramp_down_MW_h", 0.0) / SB
        elseif g <= length(geradores_data) + length(BAR_GWD)
            tipo = "CUR"
            id_gerador = "CUR_$id_barra"
            ramp_up = 0.0
            ramp_down = 0.0
        else
            tipo = "DEF"
            id_gerador = "DEF_$id_barra"
            ramp_up = 0.0
            ramp_down = 0.0
        end
        
        SQLite.execute(db, """
            INSERT INTO geradores_contingencia 
            (id_execucao, id_contingencia, id_gerador, id_barra, tipo, geracao_pu, 
             pmin_pu, pmax_pu, custo_marginal_usd_mwh, ramp_up_pu, ramp_down_pu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_EXECUCAO, ctg_id, id_gerador, id_barra, tipo,
              round(final_PG[g], digits=6), round(PGMIN[g], digits=6),
              round(PGMAX[g], digits=6), round(CPG[g], digits=6),
              ramp_up, ramp_down])
    end
    
    # ==========================================================================
    # TABELA: LINHAS_POR_CONTINGENCIA
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS linhas_contingencia (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_linha TEXT,
            id_barra_origem TEXT,
            id_barra_destino TEXT,
            fluxo_max_pu REAL,
            limite_pu REAL,
            fluxo_origem_destino_pu REAL,
            fluxo_destino_origem_pu REAL,
            perdas_pu REAL,
            utilizacao_percentual REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao),
            FOREIGN KEY (id_contingencia) REFERENCES contingencias (id_contingencia)
        )
    """)
    
    for e in 1:NLIN
        linha = linhas[e]
        
        fluxo_max = max(abs(fij[e]), abs(fji[e]))
        utilizacao_percentual = FLIM[e] > 0 ? round(fluxo_max / FLIM[e] * 100, digits=2) : 0.0
        status = utilizacao_percentual > 95 ? "CRITICO" : "NORMAL"
        
        SQLite.execute(db, """
            INSERT INTO linhas_contingencia 
            (id_execucao, id_contingencia, id_linha, id_barra_origem, id_barra_destino,
             fluxo_max_pu, limite_pu, fluxo_origem_destino_pu, fluxo_destino_origem_pu,
             perdas_pu, utilizacao_percentual, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_EXECUCAO, ctg_id, linha["ID_linha"], linha["ID_Barra_Origem"], linha["ID_Barra_Destino"],
              round(fluxo_max, digits=6), round(FLIM[e], digits=6),
              round(fij[e], digits=6), round(fji[e], digits=6),
              round(perdas[e], digits=6), utilizacao_percentual, status])
    end
    
    # ==========================================================================
    # TABELA: MVU_MVD_POR_CONTINGENCIA
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS mvu_mvd_contingencia (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_execucao TEXT,
            id_contingencia TEXT,
            id_gerador TEXT,
            mvu_pu REAL,
            mvd_pu REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao),
            FOREIGN KEY (id_contingencia) REFERENCES contingencias (id_contingencia)
        )
    """)
    
    ctg_idx = findfirst(ctg -> ctg["ID_Contingencia"] == ctg_id, contingencias_data)
    if ctg_idx !== nothing && ctg_idx > 1
        for g in 1:NGER_ORIGINAL
            if g <= length(geradores_data)
                gerador = geradores_data[g]
                SQLite.execute(db, """
                    INSERT INTO mvu_mvd_contingencia 
                    (id_execucao, id_contingencia, id_gerador, mvu_pu, mvd_pu)
                    VALUES (?, ?, ?, ?, ?)
                """, [ID_EXECUCAO, ctg_id, gerador["ID_Gerador"],
                      round(MVu[ctg_idx, g], digits=6), round(MVd[ctg_idx, g], digits=6)])
            end
        end
    end
    
    SQLite.close(db)
    
    println("  📊 Dados da contingência $ctg_id exportados para SQLite")
end

# ==============================================================================
# CARREGAMENTO E PROCESSAMENTO DOS DADOS DO SISTEMA
# ==============================================================================

# Carrega dados da rede elétrica do arquivo JSON
data = JSON.parsefile("../DATA/input/3barras_BASE.json")
#data = JSON.parsefile("../DATA/input/B6L8_BASE.json")
#data = JSON.parsefile("../DATA/input/ieee14_BASE.json")
#data = JSON.parsefile("DATA/input/ieee118_BASE.json")

# ==============================================================================
# DADOS DE BASE DO SISTEMA
# ==============================================================================
# Extrai informações das listas separadas
barras = data["BARRAS"]
geradores_data = data["GERADORES"]
demandas_data = data["DEMANDAS"]
linhas = data["LINHAS"]
contingencias_data = get(data, "CONTINGENCIAS", [])

# Salvar valores originais para referência
for g in geradores_data
    if g["Tipo"] == "GWD"
        g["PGERmax_MW_ORIGINAL"] = g["PGERmax_MW"]
    end
end

for d in demandas_data
    if haskey(d, "PLOAD")
        d["PLOAD_ORIGINAL"] = d["PLOAD"]
    end
end

# Aplicar cenário aleatório ANTES das conversões para PU
gerar_cenario_aleatorio!(geradores_data, demandas_data, data["S_base"])

# Verifica se há contingências definidas
if isempty(contingencias_data)
    println("⚠️  Nenhuma contingência definida no arquivo. Criando contingência base.")
    contingencias_data = [
        Dict(
            "ID_Contingencia" => "CTG-BASE",
            "Descricao" => "Caso Base - Sem contingência",
            "Linhas_Removidas" => []
        )
    ]
end

println("Número de contingências a serem analisadas: $(length(contingencias_data))")

# Potência base (MVA)
SB = data["S_base"]
PB = data["P_base"]
# Tensão base (kV)
VB = data["V_base"]
# Frequência base (Hz)
FB = data["f_base"]
# Impedância base (Ω)
ZB = (VB^2) / PB
# Admitância base (S)
YB = 1 / ZB
# Potência reativa base (MVAr)
QB = data["Q_base"]

println("===== BASES DO SISTEMA =====")
println("Potência base (PB): ", PB, " MVA")
println("Tensão base (VB): ", VB, " kV")
println("Frequência base (FB): ", FB, " Hz")
println("Impedância base (ZB): ", round(ZB, digits=4), " Ω")
println("Admitância base (YB): ", round(YB, digits=6), " S")
println("============================")

# ------------------------------------------------------------------------------
# Conversão dos dados da rede para PU
# ------------------------------------------------------------------------------

# --- Conversão das barras (cargas e tensões)
for b in barras
    # Se os dados já estão em MW/MVAr, converter para pu
    if haskey(b, "P_carga_MW")
        b["P_carga_pu"] = b["P_carga_MW"] / SB
    else
        b["P_carga_pu"] = 0.0
    end
    
    if haskey(b, "Q_carga_MVAr")
        b["Q_carga_pu"] = b["Q_carga_MVAr"] / SB
    else
        b["Q_carga_pu"] = 0.0
    end
end

# --- Conversão dos geradores
for g in geradores_data
    g["Pmax_pu"] = g["PGERmax_MW"] / SB
    g["Pmin_pu"] = g["PGERmin_MW"] / SB
    
    # Para geradores que não têm Qmax/Qmin definidos, usar valores padrão
    if haskey(g, "Qmax_MVAr")
        g["Qmax_pu"] = g["Qmax_MVAr"] / SB
    else
        g["Qmax_pu"] = 2.0  # 200 MVAr em pu
    end
    
    if haskey(g, "Qmin_MVAr")
        g["Qmin_pu"] = g["Qmin_MVAr"] / SB
    else
        g["Qmin_pu"] = -2.0  # -200 MVAr em pu
    end
    
    # Valores de referência (se existirem)
    if haskey(g, "Pg_ref_MW")
        g["Pg_ref_pu"] = g["Pg_ref_MW"] / SB
    else
        g["Pg_ref_pu"] = 0.0
    end
    
    if haskey(g, "Qg_ref_MVAr")
        g["Qg_ref_pu"] = g["Qg_ref_MVAr"] / SB
    else
        g["Qg_ref_pu"] = 0.0
    end
end

# --- Conversão das linhas (impedâncias e limites)
for l in linhas
    # Converter resistência e reatância de ohms para pu
    l["R_pu"] = l["R"] / ZB
    l["X_pu"] = l["X"] / ZB
    
    # Se existir susceptância shunt, converter de siemens para pu
    if haskey(l, "Bsh")
        l["B_pu"] = l["Bsh"] * ZB
    else
        l["B_pu"] = 0.0
    end
    
    # Converter limite de fluxo para pu
    l["Fmax_pu"] = l["LIM_Fluxo"] / SB
end

# Mapeamento de IDs das barras para índices numéricos
bus_ids = [b["ID_Barra"] for b in barras]
idx_map = Dict(id => i for (i,id) in enumerate(bus_ids))
NBAR = length(bus_ids)
NLIN = length(linhas)

# ==============================================================================
# IDENTIFICAÇÃO DA BARRA SLACK
# ==============================================================================

slack_list = filter(b->b["tipo"]=="Slack", barras)
if length(slack_list) != 1
    error("Deve haver exatamente 1 barra Slack no JSON")
end
slack_id = slack_list[1]["ID_Barra"]
slack_idx = idx_map[slack_id]

# ==============================================================================
# PARÂMETROS DAS LINHAS DE TRANSMISSÃO
# ==============================================================================

line_fr = Vector{Int}(undef, NLIN)
line_to = Vector{Int}(undef, NLIN)  
r_line = zeros(NLIN)
x_line = zeros(NLIN)
y_line = zeros(NLIN)
g_line = zeros(NLIN)
FLIM = zeros(NLIN)

for (e, ln) in enumerate(linhas)
    fr = ln["ID_Barra_Origem"]
    to = ln["ID_Barra_Destino"]
    line_fr[e] = idx_map[fr]
    line_to[e] = idx_map[to]
    
    r = ln["R_pu"]
    x = ln["X_pu"]
    
    r_line[e] = r
    x_line[e] = x
    
    denom = r^2 + x^2
    g_line[e] = denom > 0 ? r/denom : 0.0
    y_line[e] = abs(x) > 0 ? 1.0/x : 0.0
    
    FLIM[e] = ln["Fmax_pu"]
end

# ==============================================================================
# MONTAGEM DA MATRIZ DE SUSCEPTÂNCIA B_bus 
# ==============================================================================

Bbus = zeros(NBAR, NBAR)

for e in 1:NLIN
    i = line_fr[e]
    j = line_to[e]
    y = y_line[e]
    
    Bbus[i,i] += y
    Bbus[j,j] += y
    Bbus[i,j] -= y
    Bbus[j,i] -= y
end

# ==============================================================================
# DADOS DOS GERADORES
# ==============================================================================

# Processa geradores da lista separada
NGER_ORIGINAL = length(geradores_data)
BARPG_ORIGINAL = Vector{Int}(undef, NGER_ORIGINAL)
BAR_GWD = Int[]
PGMIN_ORIGINAL = zeros(NGER_ORIGINAL)
PGMAX_ORIGINAL = zeros(NGER_ORIGINAL) 
PGMIN_EFETIVO = zeros(NGER_ORIGINAL)
PGMAX_EFETIVO = zeros(NGER_ORIGINAL)   
CPG_ORIGINAL = zeros(NGER_ORIGINAL)

# Identificar geradores eólicos primeiro
for (i, g) in enumerate(geradores_data)
    id_barra = g["ID_Barra"]
    BARPG_ORIGINAL[i] = idx_map[id_barra]
    tipo_ger = g["Tipo"]
    
    if tipo_ger == "GWD"
        println("Identificado gerador eólico GWD na barra $id_barra")
        push!(BAR_GWD, idx_map[id_barra])
    end
end

# Processar todos os geradores
for (i, g) in enumerate(geradores_data)
    id_barra = g["ID_Barra"]
    BARPG_ORIGINAL[i] = idx_map[id_barra]
    tipo_ger = g["Tipo"]
    
    # Usar valores já convertidos para pu
    PGMIN_ORIGINAL[i] = g["Pmin_pu"]
    PGMAX_ORIGINAL[i] = g["Pmax_pu"]
    
    if tipo_ger == "UTE"
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * SB  # Custo em USD/h para 1 pu

    elseif tipo_ger == "UTH"
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * SB

    elseif tipo_ger == "GWD"
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * SB

        potencia_normalizada = rand(Normal(1,0.8))
        potencia_instantanea = potencia_normalizada * PGMAX_ORIGINAL[i]
        
        PGMAX_EFETIVO[i] = min(max(potencia_instantanea, 0.1*PGMAX_ORIGINAL[i]), PGMAX_ORIGINAL[i])        
        PGMIN_EFETIVO[i] = PGMAX_EFETIVO[i]
    else
        CPG_ORIGINAL[i] = 0.0
    end
    
    # Para geradores não-eólicos, usar limites originais
    if tipo_ger != "GWD"
        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        PGMIN_EFETIVO[i] = PGMIN_ORIGINAL[i]
    end
end

# ==============================================================================
# DADOS DAS DEMANDAS
# ==============================================================================

PLOAD = zeros(NBAR)
for d in demandas_data
    id_barra = d["ID_Barra"]
    idx = idx_map[id_barra]
    # Potência ativa
    potencia_demanda = get(d, "PLOAD", 0.0) / SB
    # Adicionar incerteza (3% de desvio padrão)
    σ = 0.03 * potencia_demanda  
    dist = Normal(potencia_demanda, σ)
    potencia_instantanea_incerta = max(0, rand(dist))
    PLOAD[idx] += potencia_instantanea_incerta
end

# ==============================================================================
# IDENTIFICAÇÃO DAS BARRAS PQ E ADIÇÃO DE GERADORES DE DÉFICIT E CURTAILMENT
# ==============================================================================

barras_PQ = filter(b -> b["tipo"] == "PQ", barras)

# Identifica barras que já têm geradores
barras_com_gerador = Set(BARPG_ORIGINAL)
barras_PQ_sem_gerador = [b for b in barras_PQ if idx_map[b["ID_Barra"]] ∉ barras_com_gerador]

NGER_CURTAILMENT = length(BAR_GWD)
NGER_DEFICIT = length(barras_PQ)

custo_maximo_existente = isempty(CPG_ORIGINAL) ? 1000.0 : maximum(CPG_ORIGINAL)

CUSTO_CURTAILMENT = 10.0 * custo_maximo_existente
CUSTO_DEFICIT = 100.0 * custo_maximo_existente

# Atribuição de Valor para Curtailment
BARPG_CURTAILMENT = Vector{Int}(undef, NGER_CURTAILMENT)
PGMIN_CURTAILMENT = zeros(NGER_CURTAILMENT)
PGMAX_CURTAILMENT = zeros(NGER_CURTAILMENT)
CPG_CURTAILMENT = zeros(NGER_CURTAILMENT)

# Encontrar os geradores GWD correspondentes
for (i, barra_idx) in enumerate(BAR_GWD)
    gwd_idx = findfirst(g -> idx_map[g["ID_Barra"]] == barra_idx && get(g, "Tipo", "") == "GWD", geradores_data)
    
    if gwd_idx !== nothing
        BARPG_CURTAILMENT[i] = barra_idx
        PGMIN_CURTAILMENT[i] = 0.0
        PGMAX_CURTAILMENT[i] = PGMAX_EFETIVO[gwd_idx]
        CPG_CURTAILMENT[i] = get(geradores_data[gwd_idx], "custo_curtailment_USD_MWh", CUSTO_CURTAILMENT) * PB
    end
end

# Atribuição de Valor para Déficit
BARPG_DEFICIT = Vector{Int}(undef, NGER_DEFICIT)
PGMIN_DEFICIT = zeros(NGER_DEFICIT)
PGMAX_DEFICIT = zeros(NGER_DEFICIT)
CPG_DEFICIT = zeros(NGER_DEFICIT)

for (i, b) in enumerate(barras_PQ)
    id = b["ID_Barra"]
    idx = idx_map[id]
    BARPG_DEFICIT[i] = idx
    PGMIN_DEFICIT[i] = 0.0
    PGMAX_DEFICIT[i] = PLOAD[idx] > 0 ? PLOAD[idx] * 100 : 1.0
    CPG_DEFICIT[i] = CUSTO_DEFICIT
end

# ==============================================================================
# COMBINA GERADORES ORIGINAIS, CURTAILMENT E DÉFICIT
# ==============================================================================

NGER = NGER_ORIGINAL + NGER_CURTAILMENT + NGER_DEFICIT
println("Total de geradores: $NGER_ORIGINAL originais + $NGER_CURTAILMENT curtailment + $NGER_DEFICIT de déficit = $NGER")

BARPG = vcat(BARPG_ORIGINAL, BARPG_CURTAILMENT, BARPG_DEFICIT)
PGMIN = vcat(PGMIN_EFETIVO, PGMIN_CURTAILMENT, PGMIN_DEFICIT)
PGMAX = vcat(PGMAX_EFETIVO, PGMAX_CURTAILMENT, PGMAX_DEFICIT)
CPG = vcat(CPG_ORIGINAL, CPG_CURTAILMENT, CPG_DEFICIT)

# ==============================================================================
# INICIALIZAÇÃO DAS VARIÁVEIS PARA ANÁLISE DE CONTINGÊNCIAS
# ==============================================================================

# Variáveis para armazenar resultados de todas as contingências
resultados_contingencias = Dict{String, Dict}()
global Pg_anterior = zeros(NGER_ORIGINAL)

# Matrizes para armazenar MVu e MVd
MVu = zeros(length(contingencias_data), NGER_ORIGINAL)
MVd = zeros(length(contingencias_data), NGER_ORIGINAL)

# ID único para esta execução
ID_EXECUCAO = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_" * randstring(6)

# ==============================================================================
# LOOP PRINCIPAL DE CONTINGÊNCIAS
# ==============================================================================

for (ctg_idx, contingencia) in enumerate(contingencias_data)
    global Pg_anterior
    
    ctg_id = contingencia["ID_Contingencia"]
    ctg_descricao = contingencia["Descricao"]
    linhas_removidas = get(contingencia, "Linhas_Removidas", [])
    
    println("\n" * "="^80)
    println("CONTINGÊNCIA $ctg_idx/$(length(contingencias_data)): $ctg_id")
    println("Descrição: $ctg_descricao")
    println("Linhas removidas: $(isempty(linhas_removidas) ? "Nenhuma" : join(linhas_removidas, ", "))")
    println("="^80)
    
    # ==========================================================================
    # PREPARAÇÃO DAS LINHAS PARA CONTINGÊNCIA - USANDO VARIÁVEIS AUXILIARES
    # ==========================================================================
    
    # Variáveis auxiliares para armazenar valores originais
    r_line_aux = zeros(length(linhas_removidas))
    x_line_aux = zeros(length(linhas_removidas))
    y_line_aux = zeros(length(linhas_removidas))
    g_line_aux = zeros(length(linhas_removidas))
    FLIM_aux = zeros(length(linhas_removidas))
    
    # "Anular" as linhas da contingência
    for (idx_rem, linha_id) in enumerate(linhas_removidas)
        linha_idx = findfirst(ln -> ln["ID_linha"] == linha_id, linhas)
        if linha_idx !== nothing
            # Armazenar valores originais
            r_line_aux[idx_rem] = r_line[linha_idx]
            x_line_aux[idx_rem] = x_line[linha_idx]
            y_line_aux[idx_rem] = y_line[linha_idx]
            g_line_aux[idx_rem] = g_line[linha_idx]
            FLIM_aux[idx_rem] = FLIM[linha_idx]
            
            # Anular a linha (valores muito pequenos para evitar singularidade)
            r_line[linha_idx] = 0.0
            x_line[linha_idx] = 1e-5
            y_line[linha_idx] = 1e5
            g_line[linha_idx] = 0.0
            FLIM[linha_idx] = 1e-5
            
            println("  - Anulando linha: $linha_id")
        end
    end
    
    # ==========================================================================
    # RECALCULAR MATRIZ Bbus COM LINHAS ANULADAS
    # ==========================================================================
    
    Bbus_ctg = zeros(NBAR, NBAR)
    for e in 1:NLIN
        i = line_fr[e]
        j = line_to[e]
        y = y_line[e]
        
        Bbus_ctg[i,i] += y
        Bbus_ctg[j,j] += y
        Bbus_ctg[i,j] -= y
        Bbus_ctg[j,i] -= y
    end
    
    # ==========================================================================
    # VARIÁVEIS DE ESTADO PARA ESTA CONTINGÊNCIA
    # ==========================================================================
    
    ANGLE = zeros(NBAR)
    PINJ = zeros(NBAR)
    perdas = zeros(NLIN)
    fij = zeros(NLIN)
    fji = zeros(NLIN)
    prev_total_perdas = 0.0
    
    # ==========================================================================
    # LOOP ITERATIVO DO OPF PARA ESTA CONTINGÊNCIA
    # ==========================================================================
    
    convergiu = false
    final_PG = zeros(NGER)
    final_ANGLE = zeros(NBAR)
    
    for iter in 1:NITER_MAX
        println("\n--- Contingência $ctg_id - Iteração $iter ---")
        
        # ======================================================================
        # FORMULAÇÃO DO PROBLEMA DE OTIMIZAÇÃO
        # ======================================================================
        
        model = Model(Clp.Optimizer)
        set_silent(model)
        
        # Variáveis de decisão
        @variable(model, v_PG[i=1:NGER] >= 0)
        @variable(model, v_ANG[i=1:NBAR])
        
        # Aplicação dos limites físicos
        for i in 1:NGER
            set_upper_bound(v_PG[i], PGMAX[i])
            set_lower_bound(v_PG[i], PGMIN[i])
        end
        
        for i in 1:NBAR
            set_lower_bound(v_ANG[i], -pi)
            set_upper_bound(v_ANG[i], pi)
        end
        
        # RESTRIÇÃO DA BARRA SLACK
        @constraint(model, v_ANG[slack_idx] == 0.0)
        
        # ======================================================================
        # RESTRIÇÕES DE RAMPA (MVu/MVd) - APENAS PARA CENÁRIOS C+1
        # ======================================================================
        
        if ctg_idx > 1
            for g in 1:NGER_ORIGINAL
                if g <= length(geradores_data)
                    tipo_ger = geradores_data[g]["Tipo"]
                    if tipo_ger in ["UTE", "UTH"]
                        ramp_up = geradores_data[g]["ramp_up_MW_h"] / SB
                        ramp_down = geradores_data[g]["ramp_down_MW_h"] / SB
                        
                        @constraint(model, v_PG[g] - Pg_anterior[g] <= ramp_up)
                        @constraint(model, Pg_anterior[g] - v_PG[g] <= ramp_down)
                    end
                end
            end
            println("  Aplicadas restrições de rampa MVu/MVd")
        end
        
        # ======================================================================
        # RESTRIÇÕES DE BALANÇO DE POTÊNCIA 
        # ======================================================================

        balance_constraints = @constraint(model, balance[i=1:NBAR],
            sum(v_PG[g] for g in 1:NGER_ORIGINAL if BARPG[g] == i && geradores_data[g]["Tipo"] != "GWD") +
            sum(v_PG[g] for g in 1:NGER_ORIGINAL if BARPG[g] == i && geradores_data[g]["Tipo"] == "GWD") -
            sum(v_PG[g] for g in NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT if BARPG[g] == i) +
            sum(v_PG[g] for g in NGER_ORIGINAL+NGER_CURTAILMENT+1:NGER if BARPG[g] == i) -
            sum(Bbus_ctg[i,j] * v_ANG[j] for j in 1:NBAR) == PLOAD[i] + PINJ[i]
        )
        
        # ======================================================================
        # RESTRIÇÃO DE CURTAILMENT 
        # ======================================================================
        
        for g in geradores_data
            if g["Tipo"] == "GWD"
                posicao_barra = idx_map[g["ID_Barra"]]
                posicao_var_gerador = findfirst(i -> BARPG_ORIGINAL[i] == posicao_barra, 1:NGER_ORIGINAL)
                posicao_var_curtailment = findfirst(i -> BARPG_CURTAILMENT[i] == posicao_barra, 1:NGER_CURTAILMENT)            
                
                if posicao_var_gerador !== nothing && posicao_var_curtailment !== nothing
                    @constraint(model, 
                        v_PG[posicao_var_gerador] + v_PG[NGER_ORIGINAL + posicao_var_curtailment] == PGMAX_EFETIVO[posicao_var_gerador]
                    )
                end
            end
        end
        
        # ======================================================================
        # RESTRIÇÕES DE LIMITES DE FLUXO NAS LINHAS
        # ======================================================================
        
        for e in 1:NLIN
            i = line_fr[e]
            j = line_to[e]

            @constraint(model, y_line[e] * (v_ANG[i] - v_ANG[j]) <= FLIM[e])
            @constraint(model, y_line[e] * (v_ANG[i] - v_ANG[j]) >= -FLIM[e])
        end
        
        # ======================================================================
        # FUNÇÃO OBJETIVO
        # ======================================================================

        @objective(model, Min, 
            sum(CPG[g] * v_PG[g] for g in 1:NGER_ORIGINAL) +
            sum(CPG[g] * v_PG[g] for g in NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT) +
            sum(CPG[g] * v_PG[g] for g in NGER_ORIGINAL+NGER_CURTAILMENT+1:NGER)
        )
        
        # ======================================================================
        # RESOLUÇÃO DO PROBLEMA
        # ======================================================================

        optimize!(model)
        status = termination_status(model)
        println("Status da solução: $status")
        
        # ======================================================================
        # PROCESSAMENTO DA SOLUÇÃO
        # ======================================================================
        
        if has_values(model)
            PG_new = value.(v_PG)
            ANGLE_NEW = value.(v_ANG)
            
            DIFMAX = maximum(abs.(ANGLE_NEW - ANGLE))
            
            # Atualização das perdas e fluxos
            total_perdas = 0.0
            fill!(PINJ, 0.0)
            
            for i in 1:NLIN
                fr = line_fr[i]
                to = line_to[i]
                delta_theta = ANGLE_NEW[fr] - ANGLE_NEW[to]
                g = g_line[i]
                y = y_line[i]
                
                perdas[i] = g * delta_theta^2
                total_perdas += perdas[i]
                
                fij[i] = y * delta_theta + (g * delta_theta^2) / 2
                fji[i] = -y * delta_theta + (g * delta_theta^2) / 2
                
                PINJ[fr] += perdas[i] / 2
                PINJ[to] += perdas[i] / 2
            end
            
            loss_diff = abs(total_perdas - prev_total_perdas)
            
            println("Iteração $iter:")
            println("  delta_theta_max = $(round(DIFMAX, digits=6))")
            println("  delta_Perdas = $(round(loss_diff, digits=6))") 
            println("  Perdas = $(round(total_perdas, digits=6))")
            
            ANGLE = copy(ANGLE_NEW)
            prev_total_perdas = total_perdas
            final_PG = copy(PG_new)
            final_ANGLE = copy(ANGLE_NEW)
            
            if DIFMAX < TOL && loss_diff < TOL
                println("Convergência atingida na iteração $iter")
                convergiu = true
                break
            end
        else
            println("Nenhuma solução disponível na iteração $iter")
            break
        end
    end
    
    if !convergiu && NITER_MAX > 1
        println("⚠️  Convergência não atingida após $NITER_MAX iterações para contingência $ctg_id")
    end
    
    # ==========================================================================
    # CÁLCULO DE MVu E MVd - APENAS PARA CENÁRIOS C+1
    # ==========================================================================
    
    if ctg_idx > 1
        for g in 1:NGER_ORIGINAL
            if g <= length(geradores_data)
                diferenca = final_PG[g] - Pg_anterior[g]
                
                if diferenca < 0
                    diferenca_abs = -diferenca
                    if diferenca_abs > MVd[ctg_idx, g]
                        MVd[ctg_idx, g] = diferenca_abs
                    end
                elseif diferenca > 0
                    if diferenca > MVu[ctg_idx, g]
                        MVu[ctg_idx, g] = diferenca
                    end
                end
            end
        end
        println("  MVu/MVd calculados para esta contingência")
    end
    
    # ==========================================================================
    # RESTAURAR VALORES ORIGINAIS DAS LINHAS ANULADAS
    # ==========================================================================
    
    for (idx_rem, linha_id) in enumerate(linhas_removidas)
        linha_idx = findfirst(ln -> ln["ID_linha"] == linha_id, linhas)
        if linha_idx !== nothing
            r_line[linha_idx] = r_line_aux[idx_rem]
            x_line[linha_idx] = x_line_aux[idx_rem]
            y_line[linha_idx] = y_line_aux[idx_rem]
            g_line[linha_idx] = g_line_aux[idx_rem]
            FLIM[linha_idx] = FLIM_aux[idx_rem]
        end
    end
    
    # ==========================================================================
    # ARMAZENAMENTO DOS RESULTADOS E PREPARAÇÃO PARA PRÓXIMA CONTINGÊNCIA
    # ==========================================================================
    
    Pg_original = zeros(NBAR)
    Pg_curtailment = zeros(NBAR)
    Pg_deficit = zeros(NBAR)
    Pg_total = zeros(NBAR)
    
    for g in 1:NGER
        bar_idx = BARPG[g]
        if g <= NGER_ORIGINAL
            Pg_original[bar_idx] += final_PG[g]
        elseif g <= NGER_ORIGINAL + NGER_CURTAILMENT
            Pg_curtailment[bar_idx] += final_PG[g]
        else
            Pg_deficit[bar_idx] += final_PG[g]
        end
        Pg_total[bar_idx] += final_PG[g]
    end
    
    for bar_idx in BAR_GWD
        Pg_total[bar_idx] = Pg_original[bar_idx] - Pg_curtailment[bar_idx] + Pg_deficit[bar_idx]
    end
    
    total_pg = sum(Pg_total)
    total_pl = sum(PLOAD)
    total_perdas_val = sum(perdas)
    total_pg_curtailment = sum(Pg_curtailment)
    total_pg_deficit = sum(Pg_deficit)
    
    # Armazenar Pg atual para próxima contingência
    Pg_anterior = copy(final_PG[1:NGER_ORIGINAL])
    
    # Armazenar resultados
    resultados_contingencias[ctg_id] = Dict(
        "Pg_original" => copy(Pg_original),
        "Pg_curtailment" => copy(Pg_curtailment),
        "Pg_deficit" => copy(Pg_deficit),
        "Pg_total" => copy(Pg_total),
        "total_pg" => total_pg,
        "total_pl" => total_pl,
        "total_perdas" => total_perdas_val,
        "total_curtailment" => total_pg_curtailment,
        "total_deficit" => total_pg_deficit,
        "ANGLE" => copy(final_ANGLE),
        "FLUXOS_FIJ" => copy(fij),
        "FLUXOS_FJI" => copy(fji),
        "PERDAS" => copy(perdas),
        "PG_FINAL" => copy(final_PG)
    )
    
    println("\n✓ Contingência $ctg_id finalizada:")
    println("  - Geração total: $(round(total_pg, digits=4)) pu")
    println("  - Curtailment: $(round(total_pg_curtailment, digits=4)) pu")
    println("  - Déficit: $(round(total_pg_deficit, digits=4)) pu")
    
    # Exportar para SQLite após cada contingência
    exportar_contingencia_para_sqlite(ctg_id, ctg_descricao, linhas_removidas, 
                                     Pg_original, Pg_curtailment, Pg_deficit, Pg_total,
                                     final_ANGLE, fij, fji, perdas, final_PG,
                                     ID_EXECUCAO, barras, NGER_ORIGINAL, geradores_data,
                                     BARPG, PGMIN, PGMAX, CPG, SB, BAR_GWD, NGER_CURTAILMENT,
                                     CPG_ORIGINAL, CPG_CURTAILMENT, CPG_DEFICIT, PLOAD,
                                     linhas, FLIM, contingencias_data, MVu, MVd)
end

# ==============================================================================
# ANÁLISE FINAL DOS RESULTADOS DAS CONTINGÊNCIAS
# ==============================================================================

println("\n" * "="^100)
println("ANÁLISE FINAL DAS CONTINGÊNCIAS")
println("="^100)

# 1. BARRAS COM MAIOR CORTE DE CARGA MÉDIO
println("\n1. BARRAS COM MAIOR CORTE DE CARGA MÉDIO:")
corte_carga_medio = zeros(NBAR)
for barra in 1:NBAR
    cortes = []
    for ctg_id in keys(resultados_contingencias)
        resultado = resultados_contingencias[ctg_id]
        push!(cortes, resultado["Pg_deficit"][barra])
    end
    corte_carga_medio[barra] = mean(cortes)
end

barras_ordenadas = sortperm(corte_carga_medio, rev=true)
println("Barra | Corte de Carga Médio (pu)")
println("-"^40)
for i in 1:min(5, NBAR)
    barra_idx = barras_ordenadas[i]
    if corte_carga_medio[barra_idx] > 0
        println("$(lpad(barra_idx, 5)) | $(round(corte_carga_medio[barra_idx], digits=6))")
    end
end

# 2. CURTAILMENT MÉDIO POR CONTINGÊNCIA
println("\n2. CURTAILMENT MÉDIO POR CONTINGÊNCIA:")
println("Contingência | Curtailment Médio (pu)")
println("-"^45)
for ctg_id in keys(resultados_contingencias)
    resultado = resultados_contingencias[ctg_id]
    curt_medio = resultado["total_curtailment"]
    println("$(lpad(ctg_id, 11)) | $(round(curt_medio, digits=6))")
end

# 3. MVu E MVd POR GERADOR
println("\n3. MVu E MVd POR GERADOR:")
println("Gerador | MVu Máximo (pu) | MVd Máximo (pu)")
println("-"^45)
for g in 1:NGER_ORIGINAL
    if g <= length(geradores_data)
        mv_u_max = maximum(MVu[:, g])
        mv_d_max = maximum(MVd[:, g])
        if mv_u_max > 0 || mv_d_max > 0
            gerador_id = geradores_data[g]["ID_Gerador"]
            println("$(lpad(gerador_id, 7)) | $(lpad(round(mv_u_max, digits=6), 14)) | $(lpad(round(mv_d_max, digits=6), 14))")
        end
    end
end

# 4. LINHA MAIS IMPACTANTE PARA CTG
println("\n4. LINHA MAIS IMPACTANTE PARA CTG:")
impacto_linhas = Dict{String, Float64}()
for (ctg_idx, contingencia) in enumerate(contingencias_data)
    if ctg_idx > 1
        linhas_removidas = get(contingencia, "Linhas_Removidas", [])
        for linha_id in linhas_removidas
            impacto_total = sum(MVu[ctg_idx, :]) + sum(MVd[ctg_idx, :])
            if haskey(impacto_linhas, linha_id)
                impacto_linhas[linha_id] += impacto_total
            else
                impacto_linhas[linha_id] = impacto_total
            end
        end
    end
end

if !isempty(impacto_linhas)
    linhas_ordenadas = sort(collect(impacto_linhas), by=x->x[2], rev=true)
    println("Linha | Impacto Total (MVu + MVd)")
    println("-"^35)
    for (linha_id, impacto) in linhas_ordenadas
        println("$(lpad(linha_id, 5)) | $(round(impacto, digits=6))")
    end
else
    println("Nenhuma linha removida nas contingências analisadas.")
end

println("\n" * "="^100)
println("🎯 ANÁLISE DE CONTINGÊNCIAS CONCLUÍDA!")
println("📊 Dados exportados para: resultados_opf_contingencias.db")
println("📈 ID da execução: $ID_EXECUCAO")
println("="^100)

# ==============================================================================
# FUNÇÃO PARA CONSULTA DOS DADOS (para gráficos posteriormente)
# ==============================================================================

function consultar_dados_para_graficos()
    arquivo_db = "resultados_opf_contingencias.db"
    db = SQLite.DB(arquivo_db)
    
    println("\n📈 DADOS DISPONÍVEIS PARA GRÁFICOS:")
    
    # Consultar contingências
    contingencias_df = DBInterface.execute(db, """
        SELECT id_contingencia, descricao, total_curtailment_pu, total_deficit_pu, custo_total_usd_h
        FROM contingencias 
        WHERE id_execucao = ?
        ORDER BY id_contingencia
    """, [ID_EXECUCAO]) |> DataFrame
    
    println("\nContingências disponíveis:")
    show(contingencias_df)
    println()
    
    # Consultar dados de barras para gráficos de corte de carga
    barras_df = DBInterface.execute(db, """
        SELECT id_contingencia, id_barra, deficit_pu
        FROM barras_contingencia 
        WHERE id_execucao = ? AND deficit_pu > 0
        ORDER BY id_contingencia, deficit_pu DESC
    """, [ID_EXECUCAO]) |> DataFrame
    
    println("\nCortes de carga por barra (para gráficos):")
    show(barras_df)
    println()
    
    # Consultar MVu/MVd para gráficos
    mvu_mvd_df = DBInterface.execute(db, """
        SELECT id_contingencia, id_gerador, mvu_pu, mvd_pu
        FROM mvu_mvd_contingencia 
        WHERE id_execucao = ?
        ORDER BY id_contingencia, id_gerador
    """, [ID_EXECUCAO]) |> DataFrame
    
    println("\nMVu/MVd por gerador (para gráficos):")
    show(mvu_mvd_df)
    println()
    
    SQLite.close(db)
    
    return contingencias_df, barras_df, mvu_mvd_df
end

# Executar consulta para verificar dados disponíveis
consultar_dados_para_graficos()

println("\n=== EXECUÇÃO CONCLUÍDA ===")