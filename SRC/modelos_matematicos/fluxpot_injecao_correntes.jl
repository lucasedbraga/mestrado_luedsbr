using JSON
using JuMP
using Ipopt
using Printf

# --- Carregar dados no formato do fluxpot_NR ---
data = JSON.parsefile("DATA/input/fpo_input_data.json")
barras = data["BARRAS"]
linhas = data["LINHAS"]

# --- Mapeamento de IDs para índices numéricos ---
bus_id_to_index = Dict()
for (idx, barra) in enumerate(barras)
    bus_id_to_index[barra["ID_Barra"]] = idx
end

# --- Extrair informações das barras ---
n = length(barras)
tipo_barras = [barra["tipo"] for barra in barras]
slack_idx = findfirst(t -> t == "Slack", tipo_barras)
pv_idxs = findall(t -> t == "PV", tipo_barras)
pq_idxs = findall(t -> t == "PQ", tipo_barras)

Pd = zeros(n)
Qd = zeros(n)
V_ref = zeros(n)
P_Gen = zeros(n)

for (idx, barra) in enumerate(barras)
    Pd[idx] = get(barra, "P", 0.0)
    Qd[idx] = get(barra, "Q", 0.0)
    V_ref[idx] = get(barra, "V_ref", 1.0)  # Default 1.0 se ausente
    P_Gen[idx] = get(barra, "P_Gen", 0.0)
end

# --- Construir Ybus (formato do fluxpot_NR) ---
Ybus = zeros(ComplexF64, n, n)
for linha in linhas
    orig = bus_id_to_index[linha["ID_Barra_Origem"]]
    dest = bus_id_to_index[linha["ID_Barra_Destino"]]
    z = linha["R"] + im * linha["X"]
    y = 1 / z
    Ybus[orig, orig] += y
    Ybus[dest, dest] += y
    Ybus[orig, dest] -= y
    Ybus[dest, orig] -= y
end

# --- Modelo JuMP com tratamento de tipos de barra ---
model = Model(Ipopt.Optimizer)
@variables(model, begin
    Vr[1:n]   # Parte real da tensão
    Vi[1:n]   # Parte imaginária
    Pg[1:n]   # Geração ativa (variável para Slack/PV, fixa em 0 para PQ)
    Qg[1:n]   # Geração reativa (variável para Slack, fixa para PQ/PV?)
end)

# Slack: Fixar tensão e ângulo
@constraint(model, Vr[slack_idx] == V_ref[slack_idx])
@constraint(model, Vi[slack_idx] == 0.0)

# PV: Fixar magnitude da tensão e geração ativa
for i in pv_idxs
    @constraint(model, Vr[i]^2 + Vi[i]^2 == V_ref[i]^2)
    @constraint(model, Pg[i] == P_Gen[i])  # Pg fixo no valor especificado
end

# PQ: Fixar geração ativa/reativa em 0 (sem gerador)
for i in pq_idxs
    @constraint(model, Pg[i] == 0.0)
    @constraint(model, Qg[i] == 0.0)
end

# --- Balanço de potência para todas as barras ---
for i in 1:n
    # Calcular injeção de corrente a partir de Ybus e tensões
    I_real = zero(AffExpr)
    I_imag = zero(AffExpr)
    for j in 1:n
        I_real += real(Ybus[i, j]) * Vr[j] - imag(Ybus[i, j]) * Vi[j]
        I_imag += real(Ybus[i, j]) * Vi[j] + imag(Ybus[i, j]) * Vr[j]
    end

    # Potência injetada: S = V * conj(I)
    P_inj = Vr[i] * I_real + Vi[i] * I_imag
    Q_inj = Vi[i] * I_real - Vr[i] * I_imag  # Sinal corrigido

    # Equações de balanço: Geração - Carga = Potência injetada
    @constraint(model, Pg[i] - Pd[i] == P_inj)
    @constraint(model, Qg[i] - Qd[i] == Q_inj)
end

# Objetivo arbitrário (equações já determinam a solução)
@objective(model, Min, Pg[slack_idx])  # Minimizar geração na slack

# --- Resolver o modelo ---
optimize!(model)

# --- Resultados compatíveis com fluxpot_NR ---
println("\nResultados finais:")
for i in 1:n
    Vm = sqrt(value(Vr[i])^2 + value(Vi[i])^2)
    θ_rad = atan(value(Vi[i]), value(Vr[i]))
    θ_deg = rad2deg(θ_rad)
    @printf("Barra %d: |V| = %.4f pu, θ = %.4f°\n", i, Vm, θ_deg)
end

# Geração na slack (Pg, Qg)
println("\nGeração na barra slack:")
@printf("Pg = %.4f pu, Qg = %.4f pu\n", value(Pg[slack_idx]), value(Qg[slack_idx]))