using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DataFrames, JSON, OrderedCollections, DataStructures
using JuMP , GLPK

# Carrega os dados
data = JSON.parsefile("DATA/input/input_base_MCDA.json")

"""
Função que resolve o despacho econômico para dados e pesos dados
Retorna custo, emissões e despacho ótimo
"""
function despacho_economico(data, w_c, w_e)

    barras = data["BARRAS"]

    # Extraindo geradores e demanda
    geradores = Dict()
    demandas = []

    for barra in barras
        if haskey(barra, "DGER")
            for g in barra["DGER"]
                geradores[g["id"]] = g
            end
        end

        if haskey(barra, "DLOAD") && !isempty(barra["DLOAD"])
            append!(demandas, barra["DLOAD"][1]["demanda_total_MW"])
        end
    end

    if isempty(demandas)
        error("Nenhuma demanda encontrada no JSON! Verifique o campo DLOAD.")
    end

    T = length(demandas)

    # Modelo de despacho econômico
    model = Model(GLPK.Optimizer)

    # Variáveis
    @variable(model, p[g in keys(geradores), t=1:T] >= 0)

    # Limites de geração
    for (gid, g) in geradores
        for t in 1:T
            @constraint(model, p[gid,t] >= g["Pmin_MW"])
            @constraint(model, p[gid,t] <= g["Pmax_MW"])
        end
    end

    # Balanço de carga
    for t in 1:T
        @constraint(model, sum(p[g,t] for g in keys(geradores)) == demandas[t])
    end

    custo_credito_carbono_tonelada_co2 = 26 #R$/tCO2

    # Objetivos
    @expression(model, custo_total, sum(geradores[g]["custo_var_USD_MWh"] * p[g,t] for g in keys(geradores), t in 1:T))
    @expression(model, emis_total,  sum(custo_credito_carbono_tonelada_co2*geradores[g]["emissao_tCO2_MWh"] * p[g,t] for g in keys(geradores), t in 1:T))

    # Soma ponderada
    @objective(model, Min, w_c * custo_total + w_e * emis_total)
    optimize!(model)

    println("Resultados do despacho econômico:\n")
    for t in 1:T
        println("Hora $t - Demanda: $(demandas[t]) MW")
        for g in keys(geradores)
            println("  Gerador $g: $(value(p[g, t])) MW")
        end
        println()
    end

    # Retorna valores dos objetivos
    return value(custo_total), value(emis_total)/custo_credito_carbono_tonelada_co2
end


# Loop para construir o Pareto
pareto_points = []
for w_c in 0:0.1:1
    w_e = 1 - w_c
    custo, emis = despacho_economico(data, w_c, w_e)
    push!(pareto_points, (w_c, w_e, custo, emis))
end

println("\n--- Fronteira de Pareto ---")
alternativas = []

for (w_c, w_e, custo, emis) in pareto_points

    descricao = "w_c=$(round(w_c, digits=2)), w_e=$(round(w_e, digits=2))"
    println("$(descricao) -> Custo=$(round(custo, digits=2)) USD, Emissões=$(round(emis, digits=2)) tCO2")

    push!(alternativas, Dict(
        "descricao" => descricao,
        "Custo Operacao" => round(custo, digits=2),
        "Emissao ton CO2" => round(emis, digits=2)
    ))
end


# Converte a lista de alternativas para DataFrame
df = DataFrame(alternativas)
# Cria uma chave única com base nos valores de custo e emissão
df[!,:_chave] = string.(df[!,"Custo Operacao"]) .* "|" .* string.(df[!,"Emissao ton CO2"])
# Agrupa por chave e mantém apenas a primeira ocorrência
df_filtrado = combine(groupby(df, :_chave)) do sdf
    first(sdf)
end

# Remove a coluna auxiliar
select!(df_filtrado, Not(:_chave))

# Cria lista de OrderedDicts com id
alternativas_com_id = [
    OrderedDict(
        "id_alternativa" => i,
        "descricao" => row["descricao"],
        "Custo Operacao" => row["Custo Operacao"],
        "Emissao ton CO2" => row["Emissao ton CO2"]
    )
    for (i, row) in enumerate(eachrow(df_filtrado))
]

# Escreve o JSON
open("DATA/output/input_alternativas.json", "w") do io
    JSON.print(io, alternativas_com_id)
end


using Plots

# Extrai os eixos a partir do DataFrame filtrado
custos = df_filtrado[!,"Custo Operacao"]
emissoes = df_filtrado[!,"Emissao ton CO2"]

# Cria o gráfico de dispersão
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

# Salva o gráfico como imagem
savefig("pareto.png")
println("Gráfico salvo como pareto.png")
