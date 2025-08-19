using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP
using Ipopt
using JSON
using LinearAlgebra

# Carrega os dados
data = JSON.parsefile("DATA/input/ieee14_BASE.json")

println("Dados carregados com sucesso.")
println("Barras: ", length(data["BARRAS"]), ", Linhas: ", length(data["LINHAS"]))

barras = data["BARRAS"]
linhas = data["LINHAS"]

# Identificação de barras
slack_bus = String[]
pv_buses = String[]
pq_buses = String[]
bus_ids = String[]
idx_map = Dict{String,Int}()

for (i, barra) in enumerate(barras)
    id = barra["ID_Barra"]
    tipo = barra["tipo"]
    push!(bus_ids, id)
    idx_map[id] = i

    if tipo == "Slack"
        push!(slack_bus, id)
    elseif tipo == "PV"
        push!(pv_buses, id)
    elseif tipo == "PQ"
        push!(pq_buses, id)
    end
end

# Verificação de barra slack
if isempty(slack_bus)
    error("Nenhuma barra Slack encontrada no sistema!")
elseif length(slack_bus) > 1
    error("Mais de uma barra Slack encontrada no sistema!")
end

# Acessa o ID da barra slack
slack_id = slack_bus[1]

model = Model(Ipopt.Optimizer)

# Configurações do solver
set_optimizer_attribute(model, "tol", 1e-8)
set_optimizer_attribute(model, "max_iter", 1000)
set_optimizer_attribute(model, "print_level", 0)

# Variáveis de decisão
@variable(model, θ[bus in bus_ids])  # Ângulo da tensão (radianos)
@variable(model, V[bus in bus_ids])  # Magnitude de tensão
@variable(model, Pg[bus in bus_ids]) # Geração de potência ativa
@variable(model, Qg[bus in bus_ids]) # Geração de potência reativa

# Fixa valores na barra slack
barra_slack = barras[idx_map[slack_id]]
@constraint(model, θ[slack_id] == 0)
@constraint(model, V[slack_id] == barra_slack["V_ref"])

# Fixa tensão nas barras PV
for pv_bus in pv_buses
    barra_pv = barras[idx_map[pv_bus]]
    @constraint(model, V[pv_bus] == barra_pv["V_ref"])
end

# Construir matriz de admitância Ybus
function construir_Ybus(barras, linhas, idx_map)
    n = length(barras)
    Ybus = zeros(ComplexF64, n, n)
    
    for linha in linhas
        orig = linha["ID_Barra_Origem"]
        dest = linha["ID_Barra_Destino"]
        r = linha["R"]
        x = linha["X"]
        z = complex(r, x)
        y = 1 / z  # admitância série
        
        i = idx_map[orig]
        j = idx_map[dest]
        
        Ybus[i, i] += y
        Ybus[j, j] += y
        Ybus[i, j] -= y
        Ybus[j, i] -= y
    end
    return Ybus
end

Ybus = construir_Ybus(barras, linhas, idx_map)
G = real(Ybus)
B = imag(Ybus)

# Expressões de potência injetada
P_inj = Dict{String, Any}()
Q_inj = Dict{String, Any}()

for id in bus_ids
    i = idx_map[id]
    P_inj[id] = @expression(model, 
        V[id] * sum(V[j] * (G[i, idx_map[j]] * cos(θ[id]-θ[j]) + 
                            B[i, idx_map[j]] * sin(θ[id]-θ[j])) 
                   for j in bus_ids)
    )
    
    Q_inj[id] = @expression(model,
        V[id] * sum(V[j] * (G[i, idx_map[j]] * sin(θ[id]-θ[j]) - 
                            B[i, idx_map[j]] * cos(θ[id]-θ[j])) 
                   for j in bus_ids)
    )
end

# Restrições de balanço
for barra in barras
    id = barra["ID_Barra"]
    P_load = get(barra, "P", 0.0)
    Q_load = get(barra, "Q", 0.0)
    
    # Balanço de potência ativa
    @constraint(model, Pg[id] - P_load == P_inj[id])
    
    # Balanço de potência reativa
    if barra["tipo"] == "PQ"
        @constraint(model, Qg[id] - Q_load == Q_inj[id])
    end
end

# Fixa gerações especificadas
for barra in barras
    id = barra["ID_Barra"]
    if barra["tipo"] == "PV"
        @constraint(model, Pg[id] == get(barra, "P_Gen", 0.0))
    elseif barra["tipo"] == "PQ"
        @constraint(model, Pg[id] == get(barra, "P_Gen", 0.0))
        @constraint(model, Qg[id] == get(barra, "Q_Gen", 0.0))
    end
end

# Função objetivo neutra
@objective(model, Min, 0)

# Inicialização
for id in bus_ids
    set_start_value(θ[id], 0)
    set_start_value(V[id], 1.0)
    set_start_value(Pg[id], get(barras[idx_map[id]], "P_Gen", 0.0))
    set_start_value(Qg[id], get(barras[idx_map[id]], "Q_Gen", 0.0))
end

optimize!(model)

# Resultados
if termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    println("\nResultados do Fluxo de Potência Ótimo:")
    println("Barra |   V (pu)   |  θ (graus)  | Pg (pu)   | Qg (pu)   | Tipo")
    
    for id in sort(bus_ids, by=x->parse(Int, x))
        i = idx_map[id]
        v_val = round(value(V[id]), digits=4)
        θ_deg = round(rad2deg(value(θ[id])), digits=4)
        pg_val = round(value(Pg[id]), digits=4)
        qg_val = round(value(Qg[id]), digits=4)
        tipo = barras[i]["tipo"]
        
        println("$(lpad(id,4)) | $(lpad(v_val,8)) | $(lpad(θ_deg,10)) | $(lpad(pg_val,8)) | $(lpad(qg_val,8)) | $tipo")
    end
    
    total_gen = sum(value(Pg[id]) for id in bus_ids)
    total_load = sum(get(barras[idx_map[id]], "P", 0.0) for id in bus_ids)
    losses = total_gen - total_load
    println("\nPerdas de transmissão: $(round(losses, digits=6)) pu")
else
    println("Otimização falhou: ", termination_status(model))
end