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

NITER_MAX = 10
TOL = 1e-6

println("=== OPF DC ITERATIVO REVISADO - CORRIGIDO ===")

# ==============================================================================
# FUNÇÃO PARA GERAR CENÁRIO ALEATÓRIO
# ==============================================================================

function gerar_cenario_aleatorio!(geradores_data, demandas_data, SB)
    println("🎲 Gerando cenário aleatório...")
    
    for g in geradores_data
        if g["Tipo"] == "GWD"
            capacidade_original = get(g, "PGERmax_MW_ORIGINAL", g["PGERmax_MW"])
            fator_geracao = rand(Uniform(0.2, 1.0))
            g["PGERmax_MW"] = capacidade_original * fator_geracao
            println("🌬️  Geração eólica ajustada: $(round(g["PGERmax_MW"], digits=2)) MW ($(round(fator_geracao*100, digits=1))% da capacidade)")
        end
    end
    
    for d in demandas_data
        if haskey(d, "PLOAD")
            demanda_original = get(d, "PLOAD_ORIGINAL", d["PLOAD"])
            fator_demanda = rand(Uniform(0.8, 1.2))
            d["PLOAD"] = demanda_original * fator_demanda
            println("💡 Demanda ajustada: $(round(d["PLOAD"], digits=2)) MW ($(round(fator_demanda*100, digits=1))% da base)")
        end
    end
    
    total_geracao_eolica = sum(g["PGERmax_MW"] for g in geradores_data if g["Tipo"] == "GWD")
    total_demanda = sum(d["PLOAD"] for d in demandas_data if haskey(d, "PLOAD"))
    
    println("📊 Resumo do cenário aleatório:")
    println("   - Geração eólica total: $(round(total_geracao_eolica, digits=2)) MW")
    println("   - Demanda total: $(round(total_demanda, digits=2)) MW")
end

# ==============================================================================
# FUNÇÃO PARA EXPORTAR CONTINGÊNCIA PARA SQLite - MODIFICADA
# ==============================================================================

function exportar_contingencia_para_sqlite(ctg_id, ctg_descricao, linhas_removidas,
                                          Pg_original, Pg_curtailment, Pg_deficit, Pg_total,
                                          ANGLE, perdas, final_PG,
                                          ID_EXECUCAO, barras, NGER_ORIGINAL, geradores_data,
                                          BARPG, PGMIN, PGMAX, CPG, SB, BAR_GWD, NGER_CURTAILMENT,
                                          CPG_ORIGINAL, CPG_CURTAILMENT, CPG_DEFICIT, PLOAD,
                                          linhas, FLIM, contingencias_data, MVu, MVd, y_line,
                                          lambda_balance, lambda_flow)  # NOVOS PARÂMETROS
    
    arquivo_db = "resultados_opf_contingencias.db"
    
    db = SQLite.DB(arquivo_db)
    
    # ==========================================================================
    # TABELA: EXECUCOES
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
    # TABELA: BARRAS_POR_CONTINGENCIA - ATUALIZADA COM lambda_balance
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
            lambda_balance REAL,  -- NOVA COLUNA: variável dual do balanço
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
             carga_pu, geracao_pu, curtailment_pu, deficit_pu, preco_nodal_usd_mwh, lambda_balance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_EXECUCAO, ctg_id, barra["ID_Barra"], barra["tipo"], 
              1.0, round(rad2deg(ANGLE[i]), digits=6),
              PLOAD[i], Pg_original[i], Pg_curtailment[i], Pg_deficit[i],
              0.0, round(lambda_balance[i], digits=6)])  # NOVO VALOR
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
    # TABELA: LINHAS_POR_CONTINGENCIA - ATUALIZADA COM lambda_flow
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
            perdas_pu REAL,
            utilizacao_percentual REAL,
            status TEXT,
            lambda_flow_pos REAL,  -- NOVA COLUNA: lambda para restrição de fluxo positivo
            lambda_flow_neg REAL,  -- NOVA COLUNA: lambda para restrição de fluxo negativo
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_execucao) REFERENCES execucoes (id_execucao),
            FOREIGN KEY (id_contingencia) REFERENCES contingencias (id_contingencia)
        )
    """)
    
    for e in 1:length(linhas)
        linha = linhas[e]
        
        # Calcular fluxo aproximado para DC OPF
        i = findfirst(b -> b["ID_Barra"] == linha["ID_Barra_Origem"], barras)
        j = findfirst(b -> b["ID_Barra"] == linha["ID_Barra_Destino"], barras)
        fluxo_aproximado = 0.0
        if i !== nothing && j !== nothing
            fluxo_aproximado = y_line[e] * (ANGLE[i] - ANGLE[j])
        end
        
        fluxo_max = abs(fluxo_aproximado)
        utilizacao_percentual = FLIM[e] > 0 ? round(fluxo_max / FLIM[e] * 100, digits=2) : 0.0
        status = utilizacao_percentual > 95 ? "CRITICO" : "NORMAL"
        
        # Índices para lambda_flow (2 restrições por linha)
        idx_pos = 2*(e-1) + 1  # Restrição de limite superior
        idx_neg = 2*(e-1) + 2  # Restrição de limite inferior
        
        SQLite.execute(db, """
            INSERT INTO linhas_contingencia 
            (id_execucao, id_contingencia, id_linha, id_barra_origem, id_barra_destino,
             fluxo_max_pu, limite_pu, perdas_pu, utilizacao_percentual, status,
             lambda_flow_pos, lambda_flow_neg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_EXECUCAO, ctg_id, linha["ID_linha"], linha["ID_Barra_Origem"], linha["ID_Barra_Destino"],
              round(fluxo_max, digits=6), round(FLIM[e], digits=6),
              round(perdas[e], digits=6), utilizacao_percentual, status,
              round(lambda_flow[idx_pos], digits=6),  # NOVO VALOR
              round(lambda_flow[idx_neg], digits=6)]) # NOVO VALOR
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
# FUNÇÃO PRINCIPAL REVISADA DO OPF - CORRIGIDA E MODIFICADA
# ==============================================================================

function solve_FPO_DC(NBAR, NGER, BARPG, PGMIN, PGMAX, CPG, PLOAD, Bbus, slack_idx, 
                                 line_fr, line_to, y_line, g_line, FLIM, NGER_ORIGINAL, geradores_data,
                                 NGER_CURTAILMENT, NGER_DEFICIT, Pg_anterior, ctg_idx, SB,
                                 idx_map, BARPG_ORIGINAL, BARPG_CURTAILMENT, PGMAX_EFETIVO)

    # Inicializar variáveis locais
    convergiu = false
    final_PG = zeros(NGER)
    final_ANGLE = zeros(NBAR)
    PINJ = zeros(NBAR)
    perdas = zeros(length(line_fr))
    
    # Variáveis para iteração
    ANGLE = zeros(NBAR)
    prev_total_perdas = 0.0
    fij = zeros(length(line_fr))
    fji = zeros(length(line_fr))
    lambda_balance = zeros(NBAR)
    lambda_flow = zeros(2 * length(line_fr))
    
    for iter in 1:NITER_MAX
        println("\n--- Iteração $iter ---")
        
        # ==========================================================================
        # FORMULAÇÃO DO PROBLEMA DE OTIMIZAÇÃO
        # ==========================================================================
        
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
        # RESTRIÇÕES DE RAMPA
        # ======================================================================
        
        if ctg_idx > 1
            for g in 1:NGER_ORIGINAL
                if g <= length(geradores_data)
                    tipo_ger = geradores_data[g]["Tipo"]
                    if tipo_ger in ["UTE"]
                        ramp_up = get(geradores_data[g], "ramp_up_MW_h", 1000.0) / SB
                        ramp_down = get(geradores_data[g], "ramp_down_MW_h", 1000.0) / SB
                        
                        @constraint(model, v_PG[g] - Pg_anterior[g] <= ramp_up)
                        @constraint(model, Pg_anterior[g] - v_PG[g] <= min(ramp_down, Pg_anterior[g]))
                    end
                end
            end
        end
        
        # ======================================================================
        # RESTRIÇÃO DE BALANÇO DE POTÊNCIA
        # ======================================================================
        
        balance_constraints = @constraint(model, balance[i=1:NBAR],
        
            # GERAÇÃO CONVENCIONAL (UTE, UTH)
            sum(v_PG[g] for g in 1:NGER_ORIGINAL if BARPG[g] == i && geradores_data[g]["Tipo"] != "GWD") 
            
            # GERAÇÃO EÓLICA LÍQUIDA (GWD - Curtailment)
            + sum(v_PG[g] for g in 1:NGER_ORIGINAL if BARPG[g] == i && geradores_data[g]["Tipo"] == "GWD") 
            - sum(v_PG[g] for g in NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT if BARPG[g] == i) 
            
            # DÉFICIT
            + sum(v_PG[g] for g in NGER_ORIGINAL+NGER_CURTAILMENT+1:NGER if BARPG[g] == i)
            
            # Fluxo que sai das linhas
            - sum(Bbus[i,j] * v_ANG[j] for j in 1:NBAR) 
            
            ==
            
            # Cargas + Perdas
            PLOAD[i] + PINJ[i]
        )
        
        # ==========================================================================
        # RESTRIÇÃO DE CURTAILMENT - CORRIGIDA
        # ==========================================================================
        
        # Para cada gerador eólico: Geração Real + Curtailment = Geração Disponível
        for g in geradores_data
            if g["Tipo"] == "GWD"
                posicao_barra = idx_map[g["ID_Barra"]]
                posicao_var_gerador = findfirst(i -> BARPG_ORIGINAL[i] == posicao_barra, 1:NGER_ORIGINAL)
                
                # Encontrar a posição correspondente no curtailment
                posicao_var_curtailment = nothing
                for i in 1:NGER_CURTAILMENT
                    if BARPG_CURTAILMENT[i] == posicao_barra
                        posicao_var_curtailment = i
                        break
                    end
                end
                
                if posicao_var_gerador !== nothing && posicao_var_curtailment !== nothing
                    @constraint(model, 
                        v_PG[posicao_var_gerador] + v_PG[NGER_ORIGINAL + posicao_var_curtailment] 
                        == 
                        PGMAX_EFETIVO[posicao_var_gerador]
                    )
                end
            end
        end
        
        # ==========================================================================
        # RESTRIÇÕES DE LIMITES DE FLUXO NAS LINHAS
        # ==========================================================================
        
        flow_constraints = []
        for e in 1:length(line_fr)
            i = line_fr[e]
            j = line_to[e]

            # Restrições de limite de fluxo (ambos os sentidos)
            c1 = @constraint(model,  y_line[e] * (v_ANG[i] - v_ANG[j]) <=  FLIM[e])
            c2 = @constraint(model,  y_line[e] * (v_ANG[i] - v_ANG[j]) >= -FLIM[e])

            push!(flow_constraints, c1)
            push!(flow_constraints, c2)
        end
        
        # ==========================================================================
        # FUNÇÃO OBJETIVO COM CUSTO DE CURTAILMENT
        # ==========================================================================

        @objective(model, Min, 
            sum(CPG[g] * v_PG[g] for g in 1:NGER_ORIGINAL) +  # Custo geração convencional e eólica
            sum(CPG[g] * v_PG[g] for g in NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT) +  # Custo curtailment
            sum(CPG[g] * v_PG[g] for g in NGER_ORIGINAL+NGER_CURTAILMENT+1:NGER)  # Custo déficit
        )
        
        # ==========================================================================
        # RESOLUÇÃO DO PROBLEMA
        # ==========================================================================

        optimize!(model)
        status = termination_status(model)
        println("Status da solução: $status")
        
        if status == MOI.OPTIMAL
            PG_new = value.(v_PG)
            ANGLE_NEW = value.(v_ANG)
            DIFMAX = maximum(abs.(ANGLE_NEW - ANGLE))            
          
            # Atualizar perdas
            total_perdas = 0.0
            fill!(PINJ, 0.0)
            
            for i in 1:length(line_fr)
                fr = line_fr[i]
                to = line_to[i]
                delta_theta = ANGLE_NEW[fr] - ANGLE_NEW[to]
                g = g_line[i]
                y = y_line[i]
                
                # Cálculo de perdas
                perdas[i] = g * delta_theta^2
                total_perdas += perdas[i]
                
                # Fluxos com perdas
                fij[i] = y * delta_theta + (g * delta_theta^2) / 2
                fji[i] = -y * delta_theta + (g * delta_theta^2) / 2
                
                # Distribuição de perdas (50% em cada extremidade)
                PINJ[fr] += perdas[i] / 2
                PINJ[to] += perdas[i] / 2
            end

            loss_diff = abs(total_perdas - prev_total_perdas)
            
            println("Iteração $iter:")
            println("  delta_theta_max = $(round(DIFMAX, digits=6))")
            println("  delta_Perdas = $(round(loss_diff, digits=6))") 
            println("  Perdas = $(round(total_perdas, digits=6))")
            
            # EXTRAIR MULTIPLICADORES DE LAGRANGE - CORRIGIDO
            try
                # Multiplicadores das restrições de balanço
                for i in 1:NBAR
                    lambda_balance[i] = dual(balance_constraints[i])
                end
                
                # Multiplicadores das restrições de fluxo
                for i in 1:length(flow_constraints)
                    lambda_flow[i] = dual(flow_constraints[i])
                end
            catch e
                println("⚠️  Aviso: Erro ao extrair multiplicadores de Lagrange: $e")
                # Manter valores zeros em caso de erro
            end
            
            # Atualização para próxima iteração
            ANGLE = copy(ANGLE_NEW)
            prev_total_perdas = total_perdas
            final_PG = copy(PG_new)
            final_ANGLE = copy(ANGLE_NEW)
            
            # Critério de convergência
            if DIFMAX < TOL && loss_diff < TOL
                println("Convergência atingida na iteração $iter")
                convergiu = true
                break
            end
        else
            println("Nenhuma solução disponível na iteração $iter")
            println("Status: $status")
            break
        end
        
        if iter == NITER_MAX
            println("Número máximo de iterações atingido")
        end
    end
    
    # Retornar também as variáveis duais
    return convergiu, final_PG, final_ANGLE, perdas, PINJ, lambda_balance, lambda_flow
end

# ==============================================================================
# CARREGAMENTO E PROCESSAMENTO DOS DADOS
# ==============================================================================

#data = JSON.parsefile("../DATA/input/ieee118_BASE.json")
#data = JSON.parsefile("../DATA/input/B6L8_BASE.json")
data = JSON.parsefile("../DATA/input/3barras_BASE.json")

barras = data["BARRAS"]
geradores_data = data["GERADORES"] 
demandas_data = data["DEMANDAS"]
linhas = data["LINHAS"]
contingencias_data = get(data, "CONTINGENCIAS", [])

# Salvar valores originais
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

# Aplicar cenário aleatório
gerar_cenario_aleatorio!(geradores_data, demandas_data, data["S_base"])

if isempty(contingencias_data)
    println("⚠️  Nenhuma contingência definida. Criando contingência base.")
    contingencias_data = [
        Dict(
            "ID_Contingencia" => "CTG-BASE",
            "Descricao" => "Caso Base - Sem contingência", 
            "Linhas_Removidas" => []
        )
    ]
end

# Configurações de base
SB = data["S_base"]
PB = data["P_base"]
VB = data["V_base"]
FB = data["f_base"]
ZB = (VB^2) / PB
YB = 1 / ZB

println("===== BASES DO SISTEMA =====")
println("Potência base (SB): ", SB, " MVA")
println("Tensão base (VB): ", VB, " kV") 
println("Impedância base (ZB): ", round(ZB, digits=4), " Ω")
println("============================")

# Conversão para PU
for b in barras
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

for g in geradores_data
    g["Pmax_pu"] = g["PGERmax_MW"] / SB
    g["Pmin_pu"] = g["PGERmin_MW"] / SB
    
    if haskey(g, "Qmax_MVAr")
        g["Qmax_pu"] = g["Qmax_MVAr"] / SB
    else
        g["Qmax_pu"] = 2.0
    end
    
    if haskey(g, "Qmin_MVAr")
        g["Qmin_pu"] = g["Qmin_MVAr"] / SB
    else
        g["Qmin_pu"] = -2.0
    end
    
    if haskey(g, "Pg_ref_MW")
        g["Pg_ref_pu"] = g["Pg_ref_MW"] / SB
    else
        g["Pg_ref_pu"] = 0.0
    end
end

for l in linhas
    l["R_pu"] = l["R"] / ZB
    l["X_pu"] = l["X"] / ZB
    
    if haskey(l, "Bsh")
        l["B_pu"] = l["Bsh"] * ZB
    else
        l["B_pu"] = 0.0
    end
    
    l["Fmax_pu"] = l["LIM_Fluxo"] / SB
end

# Mapeamento de barras
bus_ids = [b["ID_Barra"] for b in barras]
global idx_map = Dict(id => i for (i,id) in enumerate(bus_ids))
NBAR = length(bus_ids)
NLIN = length(linhas)

# Identificar barra slack
slack_list = filter(b->b["tipo"]=="Slack", barras)
if length(slack_list) != 1
    error("Deve haver exatamente 1 barra Slack")
end
slack_id = slack_list[1]["ID_Barra"]
global slack_idx = idx_map[slack_id]

# Parâmetros das linhas
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

# Matriz Bbus
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

# Dados dos geradores originais
NGER_ORIGINAL = length(geradores_data)
BARPG_ORIGINAL = Vector{Int}(undef, NGER_ORIGINAL)
BAR_GWD = Int[]
PGMIN_ORIGINAL = zeros(NGER_ORIGINAL)
PGMAX_ORIGINAL = zeros(NGER_ORIGINAL)
PGMIN_EFETIVO = zeros(NGER_ORIGINAL)
PGMAX_EFETIVO = zeros(NGER_ORIGINAL)
CPG_ORIGINAL = zeros(NGER_ORIGINAL)

# Primeiro identificar geradores GWD
for (i, g) in enumerate(geradores_data)
    id_barra = g["ID_Barra"]
    BARPG_ORIGINAL[i] = idx_map[id_barra]
    tipo_ger = g["Tipo"]
    
    if tipo_ger == "GWD"
        push!(BAR_GWD, idx_map[id_barra])
    end
end

# Agora processar todos os geradores
for (i, g) in enumerate(geradores_data)
    id_barra = g["ID_Barra"]
    BARPG_ORIGINAL[i] = idx_map[id_barra]
    tipo_ger = g["Tipo"]
    
    PGMIN_ORIGINAL[i] = g["Pmin_pu"]
    PGMAX_ORIGINAL[i] = g["Pmax_pu"]
    
    if tipo_ger == "UTE"
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 50.0) * SB
    elseif tipo_ger == "UTH"
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 80.0) * SB
    elseif tipo_ger == "GWD"
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 5.0) * SB
        
        potencia_disponivel = rand(Uniform(0.2, 1.0)) * PGMAX_ORIGINAL[i]
        PGMAX_EFETIVO[i] = potencia_disponivel
        PGMIN_EFETIVO[i] = potencia_disponivel
    else
        CPG_ORIGINAL[i] = 0.0
    end
    
    if tipo_ger != "GWD"
        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        PGMIN_EFETIVO[i] = PGMIN_ORIGINAL[i]
    end
end

# Demanda
PLOAD = zeros(NBAR)
for d in demandas_data
    id_barra = d["ID_Barra"]
    idx = idx_map[id_barra]
    potencia_demanda = get(d, "PLOAD", 0.0) / SB
    PLOAD[idx] += max(0, potencia_demanda)
end

# Geradores de curtailment e déficit
barras_PQ = filter(b -> b["tipo"] == "PQ", barras)
barras_com_gerador = Set(BARPG_ORIGINAL)
barras_PQ_sem_gerador = [b for b in barras_PQ if idx_map[b["ID_Barra"]] ∉ barras_com_gerador]

NGER_CURTAILMENT = length(BAR_GWD)
NGER_DEFICIT = NBAR

custo_maximo_existente = isempty(CPG_ORIGINAL) ? 1000.0 : maximum(CPG_ORIGINAL)
CUSTO_CURTAILMENT = 10.0 * custo_maximo_existente
CUSTO_DEFICIT = 1000.0 * custo_maximo_existente

# Curtailment
BARPG_CURTAILMENT = Vector{Int}(undef, NGER_CURTAILMENT)
PGMIN_CURTAILMENT = zeros(NGER_CURTAILMENT)
PGMAX_CURTAILMENT = zeros(NGER_CURTAILMENT)
CPG_CURTAILMENT = zeros(NGER_CURTAILMENT)

for (i, barra_idx) in enumerate(BAR_GWD)
    gwd_idx = findfirst(g -> idx_map[g["ID_Barra"]] == barra_idx && get(g, "Tipo", "") == "GWD", geradores_data)
    
    if gwd_idx !== nothing
        BARPG_CURTAILMENT[i] = barra_idx
        PGMIN_CURTAILMENT[i] = 0.0
        PGMAX_CURTAILMENT[i] = PGMAX_EFETIVO[gwd_idx]
        CPG_CURTAILMENT[i] = get(geradores_data[gwd_idx], "custo_curtailment_USD_MWh", CUSTO_CURTAILMENT) * SB
    end
end

# Déficit
BARPG_DEFICIT = Vector{Int}(undef, NGER_DEFICIT)
PGMIN_DEFICIT = zeros(NGER_DEFICIT)
PGMAX_DEFICIT = zeros(NGER_DEFICIT)
CPG_DEFICIT = zeros(NGER_DEFICIT)

for (i, b) in enumerate(barras)
    id = b["ID_Barra"]
    idx = idx_map[id]
    BARPG_DEFICIT[i] = idx
    PGMIN_DEFICIT[i] = 0.0
    PGMAX_DEFICIT[i] = PLOAD[idx] > 0 ? PLOAD[idx] * 100 : 1.0
    CPG_DEFICIT[i] = CUSTO_DEFICIT
end

# Combinar todos os geradores
NGER = NGER_ORIGINAL + NGER_CURTAILMENT + NGER_DEFICIT
println("Total de geradores: $NGER_ORIGINAL originais + $NGER_CURTAILMENT curtailment + $NGER_DEFICIT déficit = $NGER")

BARPG = vcat(BARPG_ORIGINAL, BARPG_CURTAILMENT, BARPG_DEFICIT)
PGMIN = vcat(PGMIN_EFETIVO, PGMIN_CURTAILMENT, PGMIN_DEFICIT)
PGMAX = vcat(PGMAX_EFETIVO, PGMAX_CURTAILMENT, PGMAX_DEFICIT)
CPG = vcat(CPG_ORIGINAL, CPG_CURTAILMENT, CPG_DEFICIT)

# ==============================================================================
# LOOP PRINCIPAL DE CONTINGÊNCIAS REVISADO - CORRIGIDO E MODIFICADO
# ==============================================================================

resultados_contingencias = Dict{String, Dict}()
global Pg_anterior = zeros(NGER_ORIGINAL)
MVu = zeros(length(contingencias_data), NGER_ORIGINAL)
MVd = zeros(length(contingencias_data), NGER_ORIGINAL)

ID_EXECUCAO = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_" * randstring(6)

# Salvar cópias dos parâmetros originais das linhas
r_line_original = copy(r_line)
x_line_original = copy(x_line)
y_line_original = copy(y_line)
g_line_original = copy(g_line)
FLIM_original = copy(FLIM)

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
    
    # Restaurar parâmetros originais antes de aplicar contingência
    r_line = copy(r_line_original)
    x_line = copy(x_line_original)
    y_line = copy(y_line_original)
    g_line = copy(g_line_original)
    FLIM = copy(FLIM_original)
    
    # Aplicar contingência (remover linhas)
    for linha_id in linhas_removidas
        linha_idx = findfirst(ln -> ln["ID_linha"] == linha_id, linhas)
        if linha_idx !== nothing
            # Remover linha definindo admitância para valor muito baixo
            y_line[linha_idx] = 1e-10
            g_line[linha_idx] = 0.0
            FLIM[linha_idx] = 0.0
            println("  - Removendo linha: $linha_id")
        else
            println("  ⚠️  Aviso: Linha $linha_id não encontrada")
        end
    end
    
    # Recalcular Bbus com as linhas removidas
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
    
    # Resolver OPF revisado - AGORA COM RETORNO DAS VARIÁVEIS DUAIS
    convergiu, final_PG, final_ANGLE, perdas, PINJ, lambda_balance, lambda_flow = solve_FPO_DC(
        NBAR, NGER, BARPG, PGMIN, PGMAX, CPG, PLOAD, Bbus_ctg, slack_idx,
        line_fr, line_to, y_line, g_line, FLIM, NGER_ORIGINAL, geradores_data,
        NGER_CURTAILMENT, NGER_DEFICIT, Pg_anterior, ctg_idx, SB,
        idx_map, BARPG_ORIGINAL, BARPG_CURTAILMENT, PGMAX_EFETIVO
    )
    
    # Calcular MVu/MVd apenas se a solução convergiu
    if convergiu && ctg_idx > 1
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
    
    # Processar resultados
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
    
    total_pg = sum(Pg_total)
    total_pl = sum(PLOAD)
    total_perdas_val = sum(perdas)
    total_pg_curtailment = sum(Pg_curtailment)
    total_pg_deficit = sum(Pg_deficit)
    
    # VERIFICAR BALANÇO DE POTÊNCIA
    balanco = total_pg - total_pl - total_perdas_val
    println("\n✓ Contingência $ctg_id finalizada:")
    println("  - Geração total: $(round(total_pg, digits=4)) pu")
    println("  - Demanda total: $(round(total_pl, digits=4)) pu") 
    println("  - Perdas totais: $(round(total_perdas_val, digits=4)) pu")
    println("  - Curtailment: $(round(total_pg_curtailment, digits=4)) pu")
    println("  - Déficit: $(round(total_pg_deficit, digits=4)) pu")
    println("  - BALANÇO (Geração - Demanda - Perdas): $(round(balanco, digits=6)) pu")
    println("  - Convergiu: $convergiu")
    
    if abs(balanco) > 1e-4
        println("  ⚠️  ALERTA: Balanço de potência não fechado adequadamente!")
    end
    
    # Verificar geração negativa
    if any(x -> x < -1e-6, final_PG)
        println("  ❌ ERRO CRÍTICO: Geração negativa detectada!")
    else
        println("  ✅ Geração não-negativa verificada")
    end
    
    # Atualizar Pg_anterior apenas se convergiu
    if convergiu
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
            "PERDAS" => copy(perdas),
            "PG_FINAL" => copy(final_PG),
            "BALANCO" => balanco,
            "CONVERGIU" => convergiu,
            "lambda_balance" => copy(lambda_balance),
            "lambda_flow" => copy(lambda_flow)
        )
        
        # Exportar para SQLite - AGORA COM AS VARIÁVEIS DUAIS
        exportar_contingencia_para_sqlite(ctg_id, ctg_descricao, linhas_removidas, 
                                        Pg_original, Pg_curtailment, Pg_deficit, Pg_total,
                                        final_ANGLE, perdas, final_PG,
                                        ID_EXECUCAO, barras, NGER_ORIGINAL, geradores_data,
                                        BARPG, PGMIN, PGMAX, CPG, SB, BAR_GWD, NGER_CURTAILMENT,
                                        CPG_ORIGINAL, CPG_CURTAILMENT, CPG_DEFICIT, PLOAD,
                                        linhas, FLIM, contingencias_data, MVu, MVd, y_line,
                                        lambda_balance, lambda_flow)  # NOVOS PARÂMETROS
    end
end

println("\n" * "="^100)
println("🎯 ANÁLISE DE CONTINGÊNCIAS CONCLUÍDA!")
println("📊 Dados exportados para: resultados_opf_contingencias.db")
println("📈 ID da execução: $ID_EXECUCAO")
println("="^100)

# Resumo dos resultados
println("\n=== RESUMO DAS CONTINGÊNCIAS ===")
for (ctg_id, resultado) in resultados_contingencias
    status = resultado["CONVERGIU"] ? "CONVERGIU" : "NÃO CONVERGIU"
    deficit = resultado["total_deficit"] > 0.01 ? "COM DÉFICIT" : "SEM DÉFICIT"
    println("$ctg_id: $status | $deficit | Balanço: $(round(resultado["BALANCO"], digits=6))")
end

println("\n=== EXECUÇÃO CONCLUÍDA ===")