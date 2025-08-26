using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DataFrames, JSON, OrderedCollections, DataStructures
using JuMP, GLPK
using Plots

# Carrega os dados
#data = JSON.parsefile("DATA/input/ieee_14_barras_MCDA.json")
data = JSON.parsefile("DATA/input/input_base_MCDA.json")

"""
Função que resolve o despacho econômico para dados e pesos dados
Retorna custo, emissões e despacho ótimo
"""
function despacho_economico(data, w_c, w_e)

    barras = data["BARRAS"]

    # Extraindo geradores
    geradores = Dict()
    curvas_por_barra = []

    for barra in barras
        if haskey(barra, "DGER")
            for g in barra["DGER"]
                geradores[g["id"]] = g
            end
        end

        if haskey(barra, "DLOAD") && !isempty(barra["DLOAD"])
            for d in barra["DLOAD"]
                push!(curvas_por_barra, d["demanda_total_MW"])
            end
        end
    end

    if isempty(curvas_por_barra)
        error("Nenhuma curva de demanda encontrada no JSON! Verifique o campo DLOAD.")
    end

    T = length(curvas_por_barra[1])

    # Soma total de demanda por hora
    demandas = [sum(curvas_por_barra[j][t] for j in 1:length(curvas_por_barra)) for t in 1:T]

    # Verifica se capacidade total é suficiente
    for t in 1:T
        capacidade_total = sum(g["Pmax_MW"] for g in values(geradores))
        if demandas[t] > capacidade_total
            error("Demanda na hora $t excede a capacidade total de geração!")
        end
    end

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

    # Balanço de carga por hora
    for t in 1:T
        @constraint(model, sum(p[g,t] for g in keys(geradores)) == demandas[t])
    end

    custo_credito_carbono_tonelada_co2 = 26 # R$/tCO2

    # Objetivos
    @expression(model, custo_total, sum(geradores[g]["custo_var_USD_MWh"] * p[g,t] for g in keys(geradores), t in 1:T))
    @expression(model, emis_total,  sum(custo_credito_carbono_tonelada_co2 * geradores[g]["emissao_tCO2_MWh"] * p[g,t] for g in keys(geradores), t in 1:T))

    w_c_efetivo = w_c == 0 ? 0.01 : w_c
    w_e_efetivo = w_e == 0 ? 0.01 : w_e
    
    @objective(model, Min, w_c_efetivo * custo_total + w_e_efetivo * emis_total)
    optimize!(model)

    println("Resultados do despacho econômico:\n")
    for t in 1:T
        println("Hora $t - Demanda: $(demandas[t]) MW")
        for g in keys(geradores)
            println("  Gerador $g: $(value(p[g, t])) MW")
        end
        println()
    end

    return value(custo_total), value(emis_total) / custo_credito_carbono_tonelada_co2
end

# Loop para construir a Fronteira de Pareto
pareto_points = []
for w_c in 1:-0.1:0
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
df[!,:_chave] = string.(df[!,"Custo Operacao"]) .* "|" .* string.(df[!,"Emissao ton CO2"])
df_filtrado = combine(groupby(df, :_chave)) do sdf
    first(sdf)
end
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

# Gráfico da Fronteira de Pareto
custos = df_filtrado[!,"Custo Operacao"]
emissoes = df_filtrado[!,"Emissao ton CO2"]

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

savefig("pareto.png")
println("Gráfico salvo como pareto.png")