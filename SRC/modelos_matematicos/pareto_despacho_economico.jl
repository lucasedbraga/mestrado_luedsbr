using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DataFrames
using OrderedCollections
using JuMP
using Clp
using LinearAlgebra
using JSON
using Distributions
using Plots

#input_name = "3barras_BASE"
input_name = "B6L8_BASE"
#input_name = "ieee14_BASE"

# Carrega dados da rede elétrica do arquivo JSON
data = JSON.parsefile("DATA/input/$input_name.json")

"""
Função que resolve o despacho econômico para dados e pesos dados
Retorna custo, emissões e despacho ótimo
"""
function despacho_economico(data, w_c, w_e, cenario_id) 

    # ==============================================================================
    # CONFIGURAÇÕES INICIAIS E PARÂMETROS
    # ==============================================================================

    NITER_MAX = 10  # Número máximo de iterações
    TOL = 1e-6   # Tolerância para convergência
    custo_credito_carbono_tonelada_co2 = 26 # R$/tCO2

    # Extrai informações das listas separadas
    barras = data["BARRAS"]
    geradores_data = data["GERADORES"]
    demandas_data = data["DEMANDAS"]
    linhas = data["LINHAS"]
    PB = data["P_base"] # Potência base do sistema (MVA)

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
        
        r = get(ln, "R", 0.0)
        x = get(ln, "X", 1e-6)
        
        r_line[e] = r
        x_line[e] = x
        
        denom = r^2 + x^2
        g_line[e] = denom > 0 ? r/denom : 0.0
        y_line[e] = abs(x) > 0 ? 1.0/x : 0.0
        
        FLIM[e] = get(ln, "LIM_Fluxo", 1.0)
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
    PGMIN_ORIGINAL = zeros(NGER_ORIGINAL)
    PGMAX_ORIGINAL = zeros(NGER_ORIGINAL) 
    PGMIN_EFETIVO = zeros(NGER_ORIGINAL)
    PGMAX_EFETIVO = zeros(NGER_ORIGINAL)   
    CPG_ORIGINAL = zeros(NGER_ORIGINAL)
    EMISS = zeros(NGER_ORIGINAL)

    for (i, g) in enumerate(geradores_data)
        id_barra = g["ID_Barra"]
        BARPG_ORIGINAL[i] = idx_map[id_barra]
        tipo_ger = g["Tipo"]
        
        if tipo_ger == "UTE"
            PGMIN_ORIGINAL[i] = get(g, "PGERmin_MW", 0.0) / PB
            PGMAX_ORIGINAL[i] = get(g, "PGERmax_MW", 1.0) / PB
            CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB
            EMISS[i] = get(g, "emissao_tCO2_MWh", 0.0)

            PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
            
        elseif tipo_ger == "UTH"
            PGMIN_ORIGINAL[i] = get(g, "PGERmin_MW", 0.0) / PB
            PGMAX_ORIGINAL[i] = get(g, "PGERmax_MW", 1.0) / PB
            CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB
            EMISS[i] = get(g, "emissao_tCO2_MWh", 0.0)

            PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
            
        elseif tipo_ger == "GWD"
            PGMIN_ORIGINAL[i] = get(g, "PGERmin", 0.0) / PB
            PGMAX_ORIGINAL[i] = get(g, "PGERmax", 1.0) / PB
            CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB
            EMISS[i] = get(g, "emissao_tCO2_MWh", 0.0)

            PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]

        else
            PGMIN_ORIGINAL[i] = 0.0
            PGMAX_ORIGINAL[i] = 1.0
            PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
            CPG_ORIGINAL[i] = 0.0
            EMISS[i] = 0.0
        end
    end

    # ==============================================================================
    # DADOS DAS DEMANDAS
    # ==============================================================================

    PLOAD = zeros(NBAR)
    for d in demandas_data
        id_barra = d["ID_Barra"]
        idx = idx_map[id_barra]
        
        potencia_demanda = get(d, "PLOAD",0.0)
        PLOAD[idx] += potencia_demanda / PB
    end

    # ==============================================================================
    # IDENTIFICAÇÃO DAS BARRAS PQ E ADIÇÃO DE GERADORES DE DÉFICIT
    # ==============================================================================

    barras_PQ = filter(b -> b["tipo"] == "PQ", barras)

    # Identifica barras que já têm geradores
    barras_com_gerador = Set(BARPG_ORIGINAL)
    barras_PQ_sem_gerador = [b for b in barras_PQ if idx_map[b["ID_Barra"]] ∉ barras_com_gerador]

    NGER_DEFICIT = length(barras_PQ_sem_gerador)
    custo_maximo_existente = isempty(CPG_ORIGINAL) ? 1000.0 : maximum(CPG_ORIGINAL)
    CUSTO_DEFICIT = 100.0 * custo_maximo_existente

    BARPG_DEFICIT = Vector{Int}(undef, NGER_DEFICIT)
    PGMIN_DEFICIT = zeros(NGER_DEFICIT)
    PGMAX_DEFICIT = zeros(NGER_DEFICIT)
    CPG_DEFICIT = zeros(NGER_DEFICIT)
    EMISS_DEFICIT = zeros(NGER_DEFICIT)

    for (i, b) in enumerate(barras_PQ_sem_gerador)
        id = b["ID_Barra"]
        idx = idx_map[id]
        BARPG_DEFICIT[i] = idx
        PGMIN_DEFICIT[i] = 0.0
        # Limite máximo baseado na carga da barra
        PGMAX_DEFICIT[i] = PLOAD[idx] > 0 ? PLOAD[idx] * 1.5 : 1.0
        CPG_DEFICIT[i] = CUSTO_DEFICIT
        EMISS_DEFICIT[i] = 0.0
    end

    # ==============================================================================
    # COMBINA GERADORES ORIGINAIS E DE DÉFICIT
    # ==============================================================================

    NGER = NGER_ORIGINAL + NGER_DEFICIT

    BARPG = vcat(BARPG_ORIGINAL, BARPG_DEFICIT)
    PGMIN = vcat(PGMIN_ORIGINAL, PGMIN_DEFICIT)
    PGMAX = vcat(PGMAX_ORIGINAL, PGMAX_DEFICIT)
    CPG = vcat(CPG_ORIGINAL, CPG_DEFICIT)
    EMISS = vcat(EMISS, EMISS_DEFICIT)

    # ==============================================================================
    # INICIALIZAÇÃO DAS VARIÁVEIS GLOBAIS - DENTRO DA FUNÇÃO
    # ==============================================================================

    # Variáveis de estado do sistema - INICIALIZADAS AQUI
    ANGLE = zeros(NBAR)
    PINJ = zeros(NBAR)
    perdas = zeros(NLIN)
    fij = zeros(NLIN)
    fji = zeros(NLIN)

    # Armazenamento da solução final
    final_PG = zeros(NGER)
    final_ANGLE = zeros(NBAR)

    # Coeficientes de Lagrange
    lambda_balance = zeros(NBAR)
    lambda_flow = zeros(2*NLIN)

    # Variável para controle de convergência de perdas
    prev_total_perdas = 0.0

    # Variáveis para armazenar resultados da última iteração
    last_custo_total = 0.0
    last_emiss_total = 0.0
    last_perdas = 0.0 
    last_status = MOI.INFEASIBLE

    # LOOP PRINCIPAL DE ITERAÇÕES
    for iter in 1:NITER_MAX        
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
        end
        
        for i in 1:NBAR
            set_lower_bound(v_ANG[i], -pi)
            set_upper_bound(v_ANG[i], pi)
        end
        
        # RESTRIÇÃO DA BARRA SLACK
        @constraint(model, v_ANG[slack_idx] == 0.0)
        
        # ==========================================================================
        # RESTRIÇÕES DE BALANÇO DE POTÊNCIA
        # ==========================================================================
        balance_constraints = @constraint(model, balance[i=1:NBAR],
            sum(v_PG[g] for g in 1:NGER if BARPG[g] == i) - 
            sum(Bbus[i,j] * v_ANG[j] for j in 1:NBAR) == PLOAD[i] + PINJ[i]
        )
        
        # ==========================================================================
        # RESTRIÇÕES DE LIMITES DE FLUXO NAS LINHAS
        # ==========================================================================
        
        flow_constraints = []
        for e in 1:NLIN
            i = line_fr[e]
            j = line_to[e]
            y = y_line[e]
            
            # Fluxo da linha i->j
            fluxo_ij = y * (v_ANG[i] - v_ANG[j])
            
            # Restrições de limite (ambos os sentidos)
            c1 = @constraint(model, fluxo_ij <= FLIM[e])
            c2 = @constraint(model, fluxo_ij >= -FLIM[e])
            
            push!(flow_constraints, c1)
            push!(flow_constraints, c2)
        end
        
        # ==========================================================================
        # FUNÇÃO OBJETIVO
        # ==========================================================================
        # Objetivos
        
        @expression(model, custo_total, sum(CPG[g] * v_PG[g] for g in 1:NGER))
        @expression(model, emiss_total,  sum(custo_credito_carbono_tonelada_co2 * EMISS[g] * v_PG[g] for g in 1:NGER))

        w_c_efetivo = w_c == 0 ? 0.001 : w_c
        w_e_efetivo = w_e == 0 ? 0.001 : w_e

        @objective(model, Min, w_c_efetivo * custo_total + w_e_efetivo * emiss_total)
        optimize!(model)
        
        # ==========================================================================
        # RESOLUÇÃO DO PROBLEMA
        # ==========================================================================
        status = termination_status(model)
        
        # ==========================================================================
        # PROCESSAMENTO DA SOLUÇÃO
        # ==========================================================================
        
        # Verifica se há solução disponível
        if has_values(model)
            PG_new = value.(v_PG)
            ANGLE_NEW = value.(v_ANG)
            
            DIFMAX = maximum(abs.(ANGLE_NEW - ANGLE))
            
            # ==========================================================================
            # ATUALIZAÇÃO DAS PERDAS E FLUXOS 
            # ==========================================================================
            
            total_perdas = 0.0
            fill!(PINJ, 0.0)
            
            for i in 1:NLIN
                fr = line_fr[i]
                to = line_to[i]
                delta_theta = ANGLE_NEW[fr] - ANGLE_NEW[to]
                g = g_line[i]
                y = y_line[i]
                
                # Cálculo de perdas
                perdas[i] = g * delta_theta^2
                total_perdas += perdas[i]
                
                # Fluxos com perdas
                fij[i] = y * delta_theta + g * delta_theta^2 / 2
                fji[i] = -y * delta_theta + g * delta_theta^2 / 2
                
                # Distribuição de perdas (50% em cada extremidade)
                PINJ[fr] += perdas[i] / 2
                PINJ[to] += perdas[i] / 2
            end
            
            # Critério de convergência
            loss_diff = abs(total_perdas - prev_total_perdas)
            
            # ==========================================================================
            # ATUALIZAÇÃO PARA PRÓXIMA ITERAÇÃO
            # ==========================================================================
            
            ANGLE = copy(ANGLE_NEW)
            prev_total_perdas = total_perdas
            final_PG = copy(PG_new)
            final_ANGLE = copy(ANGLE_NEW)
            
            # Armazena os resultados da última iteração bem-sucedida
            last_custo_total = value(custo_total)
            last_emiss_total = value(emiss_total)
            last_perdas = total_perdas
            last_status = status
        else
            println("Nenhuma solução disponível na iteração $iter")
            # Se não há solução, mantemos os valores anteriores
            DIFMAX = 1.0
            loss_diff = 1.0
        end
        
        # ==========================================================================
        # EXTRAÇÃO DOS MULTIPLICADORES DE LAGRANGE
        # ==========================================================================

        # Coeficiente do balanço de potência
        try
            for i in 1:NBAR
                lambda_balance[i] = dual(balance_constraints[i])
            end
        catch e
            println("Erro ao extrair multiplicadores de balanço: $e")
        end

        # Coeficientes das restrições de fluxo
        try
            for i in 1:length(flow_constraints)
                lambda_flow[i] = dual(flow_constraints[i])
            end
        catch e
            println("Erro ao extrair multiplicadores de fluxo: $e")
        end
        
        # Critério de convergência
        if has_values(model) && DIFMAX < TOL && loss_diff < TOL
            break
        elseif iter == NITER_MAX
            println("Número máximo de iterações atingido")
        end
    end

    # ==============================================================================
    # PÓS-PROCESSAMENTO DOS RESULTADOS
    # ==============================================================================

    if last_status != MOI.OPTIMAL
        error("Otimização não convergiu para o cenário $cenario_id. Status final: $last_status")
    end

    # Monta resultados por gerador
    resultados = []
    for (g_idx, g_data) in enumerate(geradores_data)
        gerador_id = g_data["ID_Gerador"]
        geracao = round(final_PG[g_idx] * PB, digits=4)  # Converte de pu para MW
        emissao = round(get(g_data, "emissao_tCO2_MWh", 0.0) * geracao, digits=4)
        
        push!(resultados, OrderedDict(
            "ID_Gerador" => gerador_id,
            "ID_Barra" => g_data["ID_Barra"],
            "geracao_MW" => geracao,
            "emissao_tCO2" => emissao
        ))
    end

    # Estrutura final com metadados do cenário
    DATA = OrderedDict(
        "cenario_id" => cenario_id,
        "peso_custo" => round(w_c, digits=2),
        "peso_emissao" => round(w_e, digits=2),
        "custo_total" => round(last_custo_total, digits=2),
        "emissao_total" => round(last_emiss_total, digits=2),
        "total_perdas" => round(last_perdas, digits=5)
    )

    # Salva JSON
    output_file = "/home/lucasedbraga/projetos/ufjf/mestrado_luedsbr/DATA/output/MCDA/fluxpot/output_fluxpot_cenario_$(cenario_id).json"
    open(output_file, "w") do io
        JSON.print(io, DATA, 4)
    end

    println("Cenário $cenario_id salvo com sucesso em $output_file")

    return last_custo_total, last_emiss_total / custo_credito_carbono_tonelada_co2, last_perdas
end

# Loop para construir a Fronteira de Pareto
pareto_points = []
for w_c in 1:-0.1:0
    w_e = 1 - w_c
    cenario_id = round(Int, 10 * w_e)
    println("Iniciando Rodada $cenario_id")
    custo, emiss, perdas_totais = despacho_economico(data, w_c, w_e, cenario_id)
    push!(pareto_points, (w_c, w_e, custo, emiss, perdas_totais))
end

println("\n--- Fronteira de Pareto ---")
alternativas = []

for (w_c, w_e, custo, emiss, perdas_totais) in pareto_points
    descricao = "w_c=$(round(w_c, digits=2)), w_e=$(round(w_e, digits=2))"
    println("$(descricao) -> Custo=$(round(custo, digits=2)) USD, Emissões=$(round(emiss, digits=2)) tCO2, Perdas=$(round(perdas_totais, digits=5))")

    push!(alternativas, Dict(
        "descricao" => descricao,
        "Custo Operacao" => [round(custo, digits=2)],
        "Emissao ton CO2" => [round(emiss, digits=2)],
        "Perdas" => [round(perdas_totais, digits=5)]
    ))
end

df = DataFrame(alternativas)
df[!,:_chave] = [string(row["Custo Operacao"][1]) * "|" * string(row["Emissao ton CO2"][1]) for row in eachrow(df)]

df_filtrado = combine(groupby(df, :_chave)) do sdf
    first(sdf)
end
select!(df_filtrado, Not(:_chave))


alternativas_com_id = [
    OrderedDict(
        "id_alternativa" => i,
        "descricao" => row["descricao"],
        "Custo Operacao" => row["Custo Operacao"][1],
        "Emissao ton CO2" => row["Emissao ton CO2"][1],
        "Perdas" => row["Perdas"][1]
    )
    for (i, row) in enumerate(eachrow(df_filtrado))
]

# Estrutura final com critérios + alternativas
DATA = OrderedDict(
    "criterios" => OrderedDict(
        "Custo Operacao" => "MIN",
        "Emissao ton CO2" => "MIN",
        "Perdas" => "MIN"
    ),
    "alternativas" => alternativas_com_id
)

# Escreve o JSON
open("/home/lucasedbraga/projetos/ufjf/mestrado_luedsbr/DATA/output/input_alternativas_NOVO.json", "w") do io
    JSON.print(io, DATA, 4)
end

custos = [row[1] for row in df_filtrado[!,"Custo Operacao"]]
emissoes = [row[1] for row in df_filtrado[!,"Emissao ton CO2"]]

scatter(
    custos, emissoes;
    xlabel = "Custo (milhares de \$)",
    ylabel = "Emissão (ton CO₂)",
    title = "Fronteira de Pareto: Custo vs Emissão",
    legend = false,
    markersize = 6,
    color = :blue,
    xlims = (0, maximum(custos) * 1.1),
    ylims = (0, maximum(emissoes) * 1.1),
    xformatter = x -> string(round(x / 1000, digits=1), "k"),
    yformatter = y -> string(round(y / 1000, digits=1), "k")
)

savefig("/home/lucasedbraga/projetos/ufjf/mestrado_luedsbr/relatorios/pareto_$input_name.png")
println("Gráfico salvo como pareto_$input_name.png")