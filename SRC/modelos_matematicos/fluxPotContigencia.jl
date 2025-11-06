using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP, Clp, JSON, SQLite, Dates, Random, LinearAlgebra

# ==============================================================================
# MÓDULO PRINCIPAL - VERSÃO CORRIGIDA
# ==============================================================================

module AnaliseContingencias

using JuMP, Clp, JSON, SQLite, Dates, Random, LinearAlgebra

export main

# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

struct SistemaEletrico
    barras::Vector{Dict}
    geradores::Vector{Dict}
    demandas::Vector{Dict}
    linhas::Vector{Dict}
    contingencias::Vector{Dict}
    
    idx_map::Dict{String, Int}
    NBAR::Int
    NLIN::Int
    NGER::Int
    slack_idx::Int
    
    line_fr::Vector{Int}
    line_to::Vector{Int}
    y_line::Vector{Float64}
    FLIM::Vector{Float64}
    
    BARPG::Vector{Int}
    PGMIN::Vector{Float64}
    PGMAX::Vector{Float64}
    CPG::Vector{Float64}
    
    PLOAD::Vector{Float64}
    SB::Float64
    ID_EXECUCAO::String
    Bbus_base::Matrix{Float64}
end

struct ResultadoOPF
    convergiu::Bool
    PG::Vector{Float64}
    ANG::Vector{Float64}
    lambda::Vector{Float64}
    custo::Float64
    fluxos::Vector{Float64}
end

# ==============================================================================
# BANCO DE DADOS - TABELAS COMPATÍVEIS COM SEU SCRIPT PYTHON
# ==============================================================================

function criar_tabelas()
    db = SQLite.DB("resultados_opf_contingencias.db")
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS execucoes (
            id_execucao TEXT PRIMARY KEY,
            sistema TEXT,
            data_execucao DATETIME,
            status TEXT
        )
    """)
    
    # ==========================================================================
    # TABELAS PARA REMOÇÃO DE LINHA (COMPATÍVEIS COM SEU SCRIPT)
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS remocao_linha_contingencias (
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
            descricao TEXT
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS remocao_linha_geradores (
            id_execucao TEXT,
            id_contingencia TEXT,
            id_gerador TEXT,
            tipo TEXT,
            geracao_pu REAL,
            custo_marginal_usd_mwh REAL,
            barra INTEGER
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS remocao_linha_barras (
            id_execucao TEXT,
            id_contingencia TEXT,
            id_barra INTEGER,
            tipo TEXT,
            deficit_pu REAL,
            lambda_nodal REAL
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS remocao_linha_linhas (
            id_execucao TEXT,
            id_contingencia TEXT,
            id_linha TEXT,
            fluxo_pu REAL,
            limite_pu REAL,
            utilizacao_percentual REAL,
            id_barra_origem INTEGER,
            id_barra_destino INTEGER
        )
    """)
    
    # ==========================================================================
    # TABELAS PARA DUPLICAÇÃO DE LINHA (COMPATÍVEIS COM SEU SCRIPT)
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS duplicacao_linha_contingencias (
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
            descricao TEXT
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS duplicacao_linha_geradores (
            id_execucao TEXT,
            id_contingencia TEXT,
            id_gerador TEXT,
            tipo TEXT,
            geracao_pu REAL,
            custo_marginal_usd_mwh REAL,
            barra INTEGER
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS duplicacao_linha_barras (
            id_execucao TEXT,
            id_contingencia TEXT,
            id_barra INTEGER,
            tipo TEXT,
            deficit_pu REAL,
            lambda_nodal REAL
        )
    """)
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS duplicacao_linha_linhas (
            id_execucao TEXT,
            id_contingencia TEXT,
            id_linha TEXT,
            fluxo_pu REAL,
            limite_pu REAL,
            utilizacao_percentual REAL,
            id_barra_origem INTEGER,
            id_barra_destino INTEGER
        )
    """)
    
    SQLite.close(db)
    println("✅ Tabelas do banco de dados criadas com sucesso!")
end

# ==============================================================================
# FUNÇÃO EXPORTAR_RESULTADOS MODIFICADA - COMPATÍVEL COM SEU SCRIPT PYTHON
# ==============================================================================

function exportar_resultados(db, sistema, cenario, linha_afetada, tipo_operacao, 
                            resultado::ResultadoOPF, descricao)
    
    total_pg = sum(resultado.PG)
    total_pl = sum(sistema.PLOAD)
    
    # Determinar qual tabela usar baseado no tipo de operação
    if tipo_operacao == "REMOCAO" || cenario == "REMOCAO_ARTIFICIAL" || cenario == "CONTINGENCIA_PROGRAMADA"
        # USAR TABELAS DE REMOÇÃO
        tabela_principal = "remocao_linha_contingencias"
        tabela_geradores = "remocao_linha_geradores"
        tabela_barras = "remocao_linha_barras"
        tabela_linhas = "remocao_linha_linhas"
        coluna_linha = "linha_removida"
        ctg_id = "REM-$linha_afetada"
    elseif tipo_operacao == "DUPLICACAO" || cenario == "DUPLICACAO_LINHA"
        # USAR TABELAS DE DUPLICAÇÃO
        tabela_principal = "duplicacao_linha_contingencias"
        tabela_geradores = "duplicacao_linha_geradores"
        tabela_barras = "duplicacao_linha_barras"
        tabela_linhas = "duplicacao_linha_linhas"
        coluna_linha = "linha_original"
        ctg_id = "DUP-$linha_afetada"
    else
        # CASO BASE - usar tabelas de remoção
        tabela_principal = "remocao_linha_contingencias"
        tabela_geradores = "remocao_linha_geradores"
        tabela_barras = "remocao_linha_barras"
        tabela_linhas = "remocao_linha_linhas"
        coluna_linha = "linha_removida"
        ctg_id = "BASE"
    end
    
    # Calcular valores para curtailment e déficit (simplificado)
    total_curtailment = 0.0
    total_deficit = max(0, total_pl - total_pg)
    
    # Inserir na tabela principal
    SQLite.execute(db, """
        INSERT INTO $tabela_principal 
        (id_execucao, id_contingencia, $coluna_linha, total_geracao_pu, total_carga_pu, 
         total_curtailment_pu, total_deficit_pu, custo_total_usd_h, lambda_slack, 
         convergiu, descricao)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [sistema.ID_EXECUCAO, ctg_id, linha_afetada, total_pg, total_pl,
          total_curtailment, total_deficit, resultado.custo, resultado.lambda[sistema.slack_idx],
          resultado.convergiu, descricao])
    
    # Inserir detalhes de geração
    for g in 1:sistema.NGER
        barra_idx = sistema.BARPG[g]
        # Determinar tipo do gerador (simplificado - você pode ajustar conforme seus dados)
        tipo_ger = "GWD"  # Ajuste conforme necessário
        
        SQLite.execute(db, """
            INSERT INTO $tabela_geradores 
            (id_execucao, id_contingencia, id_gerador, tipo, 
             geracao_pu, custo_marginal_usd_mwh, barra)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, "G$(g)", tipo_ger,
              resultado.PG[g], sistema.CPG[g], barra_idx])
    end
    
    # Inserir dados das barras
    for i in 1:sistema.NBAR
        barra = sistema.barras[i]
        deficit = 0.0
        
        # Calcular déficit na barra
        if barra["tipo"] == "PQ"
            geracao_barra = sum(resultado.PG[g] for g in 1:sistema.NGER if sistema.BARPG[g] == i)
            demanda_barra = sistema.PLOAD[i]
            deficit = max(0, demanda_barra - geracao_barra)
        end
        
        SQLite.execute(db, """
            INSERT INTO $tabela_barras 
            (id_execucao, id_contingencia, id_barra, tipo, deficit_pu, lambda_nodal)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, i, barra["tipo"], deficit, resultado.lambda[i]])
    end
    
    # Inserir fluxos nas linhas
    for e in 1:sistema.NLIN
        linha = sistema.linhas[e]
        i = sistema.line_fr[e]
        j = sistema.line_to[e]
        delta_theta = resultado.ANG[i] - resultado.ANG[j]
        fluxo = sistema.y_line[e] * delta_theta
        utilizacao = sistema.FLIM[e] > 0 ? abs(fluxo) / sistema.FLIM[e] * 100 : 0.0
        
        SQLite.execute(db, """
            INSERT INTO $tabela_linhas 
            (id_execucao, id_contingencia, id_linha, fluxo_pu, 
             limite_pu, utilizacao_percentual, id_barra_origem, id_barra_destino)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [sistema.ID_EXECUCAO, ctg_id, linha["ID_linha"], fluxo, 
              sistema.FLIM[e], utilizacao, sistema.line_fr[e], sistema.line_to[e]])
    end
    
    # Registrar execução (mantido igual)
    SQLite.execute(db, """
        INSERT OR REPLACE INTO execucoes 
        (id_execucao, sistema, data_execucao, status)
        VALUES (?, ?, datetime('now'), ?)
    """, [sistema.ID_EXECUCAO, "3barras_BASE.json", "CONCLUIDO"])
    
    println("📊 Dados exportados para: $linha_afetada ($tipo_operacao)")
end

# ==============================================================================
# CÁLCULOS DE BBUS - VERSÃO CORRIGIDA
# ==============================================================================

function calcular_Bbus_base(sistema::SistemaEletrico)
    Bbus = zeros(sistema.NBAR, sistema.NBAR)
    for e in 1:sistema.NLIN
        i, j = sistema.line_fr[e], sistema.line_to[e]
        y = sistema.y_line[e]
        Bbus[i,i] += y
        Bbus[j,j] += y
        Bbus[i,j] -= y
        Bbus[j,i] -= y
    end
    return Bbus
end

# CORREÇÃO: Aceitar tanto Vector{Bool} quanto BitVector
function calcular_Bbus_linhas_ativas(sistema::SistemaEletrico, linhas_ativas)
    Bbus = zeros(sistema.NBAR, sistema.NBAR)
    for e in 1:sistema.NLIN
        if linhas_ativas[e]
            i, j = sistema.line_fr[e], sistema.line_to[e]
            y = sistema.y_line[e]
            Bbus[i,i] += y
            Bbus[j,j] += y
            Bbus[i,j] -= y
            Bbus[j,i] -= y
        end
    end
    return Bbus
end

function calcular_Bbus_duplicar_linha(sistema::SistemaEletrico, linha_id::String)
    Bbus = copy(sistema.Bbus_base)
    
    linha_idx = findfirst(ln -> ln["ID_linha"] == linha_id, sistema.linhas)
    linha_idx === nothing && error("Linha '$linha_id' não encontrada!")
    
    i, j = sistema.line_fr[linha_idx], sistema.line_to[linha_idx]
    y_original = sistema.y_line[linha_idx]
    
    # Duplicar = adicionar mesma susceptância
    Bbus[i,i] += y_original
    Bbus[j,j] += y_original
    Bbus[i,j] -= y_original
    Bbus[j,i] -= y_original
    
    return Bbus
end

# ==============================================================================
# OPF PRINCIPAL
# ==============================================================================

function resolver_opf(sistema::SistemaEletrico, Bbus::Matrix{Float64})::ResultadoOPF
    model = Model(Clp.Optimizer)
    set_silent(model)
    
    @variable(model, PG[1:sistema.NGER] >= 0)
    @variable(model, ANG[1:sistema.NBAR])
    
    # Limites
    for g in 1:sistema.NGER
        set_upper_bound(PG[g], sistema.PGMAX[g])
        set_lower_bound(PG[g], sistema.PGMIN[g])
    end
    
    # Slack
    @constraint(model, ANG[sistema.slack_idx] == 0.0)
    
    # Balanço de potência
    balance_constraints = @constraint(model, [i=1:sistema.NBAR],
        sum(PG[g] for g in 1:sistema.NGER if sistema.BARPG[g] == i) - 
        sum(Bbus[i,j] * ANG[j] for j in 1:sistema.NBAR) == sistema.PLOAD[i]
    )
    
    # Limites de fluxo
    for e in 1:sistema.NLIN
        i, j = sistema.line_fr[e], sistema.line_to[e]
        fluxo = sistema.y_line[e] * (ANG[i] - ANG[j])
        @constraint(model, fluxo <= sistema.FLIM[e])
        @constraint(model, fluxo >= -sistema.FLIM[e])
    end
    
    # Objetivo
    @objective(model, Min, sum(sistema.CPG[g] * PG[g] for g in 1:sistema.NGER))
    
    # Resolver
    optimize!(model)
    
    if termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        PG_val = value.(PG)
        ANG_val = value.(ANG)
        
        # Multiplicadores
        lambda = [dual(balance_constraints[i]) for i in 1:sistema.NBAR]
        
        # Fluxos
        fluxos = [sistema.y_line[e] * (ANG_val[sistema.line_fr[e]] - ANG_val[sistema.line_to[e]]) 
                 for e in 1:sistema.NLIN]
        
        custo = objective_value(model)
        
        return ResultadoOPF(true, PG_val, ANG_val, lambda, custo, fluxos)
    else
        return ResultadoOPF(false, zeros(sistema.NGER), zeros(sistema.NBAR), 
                          zeros(sistema.NBAR), 0.0, zeros(sistema.NLIN))
    end
end

# ==============================================================================
# CENÁRIOS - VERSÃO CORRIGIDA
# ==============================================================================

function executar_caso_base(sistema::SistemaEletrico, db)
    println("\n" * "="^60)
    println("CASO BASE")
    println("="^60)
    
    resultado = resolver_opf(sistema, sistema.Bbus_base)
    
    if resultado.convergiu
        println("✅ Custo: $(round(resultado.custo, digits=2)) USD/h")
        exportar_resultados(db, sistema, "CASO_BASE", "CASO_BASE", "BASE", 
                           resultado, "Caso base com topologia original")
    else
        println("❌ Caso base não convergiu")
    end
    
    return resultado
end

function executar_contingencia_programada(sistema::SistemaEletrico, db, ctg)
    println("\n" * "="^60)
    println("CONTINGÊNCIA: $(ctg["ID_Contingencia"])")
    println("DESCRIÇÃO: $(ctg["Descricao"])")
    println("="^60)
    
    # CORREÇÃO: Usar Vector{Bool} explicitamente
    linhas_ativas = fill(true, sistema.NLIN)  # Em vez de trues()
    
    for linha_id in ctg["Linhas_Removidas"]
        idx = findfirst(ln -> ln["ID_linha"] == linha_id, sistema.linhas)
        if idx !== nothing
            linhas_ativas[idx] = false
            println("   - Removendo linha: $linha_id")
        end
    end
    
    Bbus_ctg = calcular_Bbus_linhas_ativas(sistema, linhas_ativas)
    resultado = resolver_opf(sistema, Bbus_ctg)
    
    if resultado.convergiu
        linhas_str = join(ctg["Linhas_Removidas"], ", ")
        exportar_resultados(db, sistema, "CONTINGENCIA_PROGRAMADA", linhas_str, "REMOCAO", 
                           resultado, "Contingência: $(ctg["Descricao"])")
        println("✅ Custo contingência: $(round(resultado.custo, digits=2)) USD/h")
    else
        println("❌ Contingência não convergiu")
    end
    
    return resultado
end

function executar_remocao_artificial(sistema::SistemaEletrico, db, linha_id::String)
    println("\n── REMOÇÃO ARTIFICIAL: $linha_id")
    
    # CORREÇÃO: Usar Vector{Bool} explicitamente
    linhas_ativas = fill(true, sistema.NLIN)
    
    idx = findfirst(ln -> ln["ID_linha"] == linha_id, sistema.linhas)
    if idx === nothing
        println("❌ Linha '$linha_id' não encontrada")
        return ResultadoOPF(false, zeros(sistema.NGER), zeros(sistema.NBAR), 
                          zeros(sistema.NBAR), 0.0, zeros(sistema.NLIN))
    end
    
    linhas_ativas[idx] = false
    Bbus_remocao = calcular_Bbus_linhas_ativas(sistema, linhas_ativas)
    resultado = resolver_opf(sistema, Bbus_remocao)
    
    if resultado.convergiu
        exportar_resultados(db, sistema, "REMOCAO_ARTIFICIAL", linha_id, "REMOCAO", 
                           resultado, "Remoção artificial da linha $linha_id")
        println("✅ Custo: $(round(resultado.custo, digits=2)) USD/h")
    else
        println("❌ Remoção não convergiu")
    end
    
    return resultado
end

function executar_duplicacao_linha(sistema::SistemaEletrico, db, linha_id::String)
    println("\n── DUPLICAÇÃO: $linha_id")
    
    Bbus_dup = calcular_Bbus_duplicar_linha(sistema, linha_id)
    resultado = resolver_opf(sistema, Bbus_dup)
    
    if resultado.convergiu
        exportar_resultados(db, sistema, "DUPLICACAO_LINHA", linha_id, "DUPLICACAO", 
                           resultado, "Duplicação da linha $linha_id")
        println("✅ Custo: $(round(resultado.custo, digits=2)) USD/h")
    else
        println("❌ Duplicação não convergiu")
    end
    
    return resultado
end

# ==============================================================================
# CONSTRUÇÃO DO SISTEMA
# ==============================================================================

function criar_sistema(data::Dict)::SistemaEletrico
    # Configurações
    SB = data["S_base"]
    VB = data["V_base"]
    ZB = (VB^2) / SB
    
    # Dados
    barras = data["BARRAS"]
    geradores = data["GERADORES"]
    demandas = data["DEMANDAS"]
    linhas = data["LINHAS"]
    contingencias = get(data, "CONTINGENCIAS", [])
    
    # Mapeamento
    idx_map = Dict(b["ID_Barra"] => i for (i,b) in enumerate(barras))
    NBAR = length(barras)
    NLIN = length(linhas)
    NGER = length(geradores)
    
    # Slack
    slack_idx = findfirst(b -> b["tipo"] == "Slack", barras)
    slack_idx === nothing && error("Barra slack não encontrada")
    
    # Conversão para PU das linhas
    for l in linhas
        l["X_pu"] = get(l, "X_pu", l["X"] / ZB)
        l["Fmax_pu"] = get(l, "Fmax_pu", l["LIM_Fluxo"] / SB)
    end
    
    # Parâmetros das linhas
    line_fr = [idx_map[l["ID_Barra_Origem"]] for l in linhas]
    line_to = [idx_map[l["ID_Barra_Destino"]] for l in linhas]
    y_line = [1.0/l["X_pu"] for l in linhas]
    FLIM = [l["Fmax_pu"] for l in linhas]
    
    # Conversão geradores
    for g in geradores
        g["Pmax_pu"] = get(g, "Pmax_pu", g["PGERmax_MW"] / SB)
        g["Pmin_pu"] = get(g, "Pmin_pu", g["PGERmin_MW"] / SB)
    end
    
    # Parâmetros geradores
    BARPG = [idx_map[g["ID_Barra"]] for g in geradores]
    PGMIN = [g["Pmin_pu"] for g in geradores]
    PGMAX = [g["Pmax_pu"] for g in geradores]
    CPG = [get(g, "custo_var_USD_MWh", 50.0) * SB for g in geradores]
    
    # Demanda
    PLOAD = zeros(NBAR)
    for d in demandas
        idx = idx_map[d["ID_Barra"]]
        PLOAD[idx] = get(d, "PLOAD_pu", d["PLOAD"] / SB)
    end
    
    # Bbus base
    Bbus_base = calcular_Bbus_base(SistemaEletrico(
        barras, geradores, demandas, linhas, contingencias,
        idx_map, NBAR, NLIN, NGER, slack_idx,
        line_fr, line_to, y_line, FLIM,
        BARPG, PGMIN, PGMAX, CPG,
        PLOAD, SB, "", zeros(NBAR, NBAR)  # ID e Bbus temporários
    ))
    
    # ID execução
    ID_EXECUCAO = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_" * randstring(6)
    
    println("✅ Sistema criado: $NBAR barras, $NLIN linhas, $NGER geradores")
    println("📊 Demanda total: $(round(sum(PLOAD), digits=4)) pu")
    
    return SistemaEletrico(
        barras, geradores, demandas, linhas, contingencias,
        idx_map, NBAR, NLIN, NGER, slack_idx,
        line_fr, line_to, y_line, FLIM,
        BARPG, PGMIN, PGMAX, CPG,
        PLOAD, SB, ID_EXECUCAO, Bbus_base
    )
end

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

function main()
    println("🚀 ANÁLISE DE CONTINGÊNCIAS - VERSÃO CORRIGIDA")
    
    try
        criar_tabelas()
        db = SQLite.DB("resultados_opf_contingencias.db")
        
        data = JSON.parsefile("DATA/input/3barras_BASE.json")
        sistema = criar_sistema(data)
        
        # 1. Caso base
        println("\n🎯 ETAPA 1: CASO BASE")
        base_result = executar_caso_base(sistema, db)
        
        # # 2. Contingências programadas
        # println("\n🎯 ETAPA 2: CONTINGÊNCIAS PROGRAMADAS")
        # for ctg in sistema.contingencias
        #     ctg_result = executar_contingencia_programada(sistema, db, ctg)
            
        #     # Comparação com caso base
        #     if base_result.convergiu && ctg_result.convergiu
        #         diferenca = ctg_result.custo - base_result.custo
        #         percentual = (diferenca / base_result.custo) * 100
        #         println("   📊 Variação vs base: $(round(diferenca, digits=2)) USD/h ($(round(percentual, digits=1))%)")
        #     end
        # end
        
        # 3. Remoções individuais
        println("\n🎯 ETAPA 3: REMOÇÕES INDIVIDUAIS")
        for linha in sistema.linhas
            rem_result = executar_remocao_artificial(sistema, db, linha["ID_linha"])
            
            # Comparação
            if base_result.convergiu && rem_result.convergiu
                diferenca = rem_result.custo - base_result.custo
                percentual = (diferenca / base_result.custo) * 100
                println("   📊 Variação vs base: $(round(diferenca, digits=2)) USD/h ($(round(percentual, digits=1))%)")
            end
        end
        
        # 4. Duplicações
        println("\n🎯 ETAPA 4: DUPLICAÇÕES")
        for linha in sistema.linhas
            dup_result = executar_duplicacao_linha(sistema, db, linha["ID_linha"])
            
            # Comparação
            if base_result.convergiu && dup_result.convergiu
                diferenca = dup_result.custo - base_result.custo
                percentual = (diferenca / base_result.custo) * 100
                println("   📊 Variação vs base: $(round(diferenca, digits=2)) USD/h ($(round(percentual, digits=1))%)")
            end
        end
        
        SQLite.close(db)
        println("\n🎉 ANÁLISE CONCLUÍDA! Resultados salvos no banco de dados.")
        
    catch e
        println("❌ ERRO: $e")
        showerror(stdout, e, catch_backtrace())
    end
end

end # module

# Executar
AnaliseContingencias.main()
Any

data = JSON.parsefile("DATA/input/3barras_BASE.json")
