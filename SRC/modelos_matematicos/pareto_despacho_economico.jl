using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP
using GLPK
using JSON

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

    custo_credito_carbono_tonelada_co2 = 26
    # Objetivos
    @expression(model, custo_total, sum(geradores[g]["custo_var_USD_MWh"] * p[g,t] for g in keys(geradores), t in 1:T))
    @expression(model, emis_total,  sum(custo_credito_carbono_tonelada_co2*geradores[g]["emissao_tCO2_MWh"] * p[g,t] for g in keys(geradores), t in 1:T))

    # Soma ponderada
    @objective(model, Min, w_c * custo_total + w_e * emis_total)

    optimize!(model)

    # Retorna valores dos objetivos
    return value(custo_total), value(emis_total)/custo_credito_carbono_tonelada_co2
end


# 🔹 Loop para construir o Pareto
pareto_points = []

for w_c in 0:0.1:1
    w_e = 1 - w_c
    custo, emis = despacho_economico(data, w_c, w_e)
    push!(pareto_points, (w_c, w_e, custo, emis))
end

println("\n--- Fronteira de Pareto ---")
for (w_c, w_e, custo, emis) in pareto_points
    println("w_c=$(round(w_c,digits=2)), w_e=$(round(w_e,digits=2)) -> Custo=$(round(custo, digits=2)) USD, Emissões=$(round(emis, digits=2)) tCO2")
end

#-------------

using Plots
# Extrair eixos
custos = [p[3] for p in pareto_points]
emissoes = [p[4] for p in pareto_points]

# Plotar
scatter(
    custos, emissoes;
    xlabel = "Custo (milhares de \$)",
    ylabel = "Emissão (ton CO₂)",
    title = "Fronteira de Pareto: Custo vs Emissão",
    legend = false,
    markersize = 6,
    color = :blue,
    xlims=(0, maximum(custos)*1.1),  # força eixo x começar em 0
    ylims=(0, maximum(emissoes)*1.1), # força eixo y começar em 0
    xformatter = x -> string(round(x/1000, digits=1), "k"), # escala em 10^3
    yformatter = y -> string(round(y/1000, digits=1), "k") # escala em 10^3
)


# Salvar em PNG
savefig("pareto.png")

println("Gráfico salvo como pareto.png")
