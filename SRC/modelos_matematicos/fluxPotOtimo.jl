using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP
using Clp
using LinearAlgebra
using JSON
using Distributions

# ==============================================================================
# CONFIGURAÇÕES INICIAIS E PARÂMETROS
# ==============================================================================

NITER_MAX = 10  # Número máximo de iterações
TOL = 1e-6   # Tolerância para convergência

println("=== OPF DC ITERATIVO COM PERDAS E CURTAILMENT ===")

# ==============================================================================
# CARREGAMENTO E PROCESSAMENTO DOS DADOS DO SISTEMA
# ==============================================================================

# Carrega dados da rede elétrica do arquivo JSON
#data = JSON.parsefile("DATA/input/3barras_Teste.json")
data = JSON.parsefile("../DATA/input/B6L8_BASE.json")
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

# Extrai informações das listas separadas
barras = data["BARRAS"]
geradores_data = data["GERADORES"]
demandas_data = data["DEMANDAS"]
linhas = data["LINHAS"]

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

# --- Conversão das linhas (impedâncias e limites) - CORRIGIDO
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

        # # Simulação eólica (mantida igual)
        # lambda_weibull = 8.0
        # fator_de_forma_weibull = 2.0
        # distribuicao_weibull = Weibull(lambda_weibull, fator_de_forma_weibull)
        # velocidade_vento = rand(distribuicao_weibull)
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
    # Encontrar o gerador GWD nesta barra
    gwd_idx = findfirst(g -> idx_map[g["ID_Barra"]] == barra_idx && get(g, "Tipo", "") == "GWD", geradores_data)
    
    if gwd_idx !== nothing
        BARPG_CURTAILMENT[i] = barra_idx
        PGMIN_CURTAILMENT[i] = 0.0
        # Limite máximo baseado na geração eólica disponível
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
    # Limite máximo baseado na carga da barra
    PGMAX_DEFICIT[i] = PLOAD[idx] > 0 ? PLOAD[idx] * 100 : 1.0  # 1000% da carga como limite
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
# INICIALIZAÇÃO DAS VARIÁVEIS GLOBAIS
# ==============================================================================

# Variáveis de estado do sistema
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

println("Iniciando processo iterativo...")

# LOOP PRINCIPAL DE ITERAÇÕES
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
    
    # ==========================================================================
    # RESTRIÇÕES DE BALANÇO DE POTÊNCIA 
    # ==========================================================================

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
    # RESTRIÇÃO DE CURTAILMENT 
    # ==========================================================================
    
    # Para cada gerador eólico: Geração Real + Curtailment = Geração Disponível
    for g in geradores_data
        if g["Tipo"] == "GWD"

            posicao_barra = idx_map[g["ID_Barra"]]
            posicao_var_gerador = findfirst(i -> BARPG_ORIGINAL[i] == posicao_barra, 1:NGER_ORIGINAL)
            posicao_var_curtailment = findfirst(i -> BARPG_CURTAILMENT[i] == posicao_barra, 1:NGER_CURTAILMENT)            
            
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
    for e in 1:NLIN
        i = line_fr[e]
        j = line_to[e]

        # # Restrições de limite de fluxo (ambos os sentidos)
        @constraint(model,  y_line[e] * (v_ANG[i] - v_ANG[j]) <=  FLIM[e])
        @constraint(model,  y_line[e] * (v_ANG[i] - v_ANG[j]) >= -FLIM[e])

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
    
    # ==========================================================================
    # PROCESSAMENTO DA SOLUÇÃO (restante do código permanece igual)
    # ==========================================================================
    
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
        
        # Critério de convergência
        loss_diff = abs(total_perdas - prev_total_perdas)
        
        println("Iteração $iter:")
        println("  delta_theta_max = $(round(DIFMAX, digits=6))")
        println("  delta_Perdas = $(round(loss_diff, digits=6))") 
        println("  Perdas = $(round(total_perdas, digits=6))")
        
        # Atualização para próxima iteração
        global ANGLE = copy(ANGLE_NEW)
        global prev_total_perdas = total_perdas
        global final_PG = copy(PG_new)
        global final_ANGLE = copy(ANGLE_NEW)
        
        # Extração dos multiplicadores de Lagrange
        try
            for i in 1:NBAR
                lambda_balance[i] = dual(balance_constraints[i])
            end
        catch e
            println("Erro ao extrair multiplicadores de balanço: $e")
        end

        try
            for i in 1:length(flow_constraints)
                lambda_flow[i] = dual(flow_constraints[i])
            end
        catch e
            println("Erro ao extrair multiplicadores de fluxo: $e")
        end
        
        # Critério de convergência
        if DIFMAX < TOL && loss_diff < TOL
            println("Convergência atingida na iteração $iter")
            break
        end
    else
        println("Nenhuma solução disponível na iteração $iter")
    end
    
    if iter == NITER_MAX
        println("Número máximo de iterações atingido")
    end
end

# ==============================================================================
# PÓS-PROCESSAMENTO DOS RESULTADOS
# ==============================================================================

Pg_original = zeros(NBAR)
Pg_curtailment = zeros(NBAR)
Pg_deficit = zeros(NBAR)
Pg_total = zeros(NBAR)

for g in 1:NGER
    bar_idx = BARPG[g]
    if g <= NGER_ORIGINAL
        Pg_original[bar_idx] += final_PG[g]
    elseif g <= NGER_ORIGINAL + NGER_CURTAILMENT
        Pg_curtailment[bar_idx] += final_PG[g]  # Curtailment é negativo no balanço
    else
        Pg_deficit[bar_idx] += final_PG[g]
    end
    Pg_total[bar_idx] += final_PG[g]
end

# Ajuste para considerar que o curtailment reduz a geração total
for bar_idx in BAR_GWD
    Pg_total[bar_idx] = Pg_original[bar_idx] - Pg_curtailment[bar_idx] + Pg_deficit[bar_idx]
end

V_final = ones(NBAR)
Qg_final = zeros(NBAR)

# ==============================================================================
# FUNÇÕES DE IMPRESSÃO DE RESULTADOS (adaptadas)
# ==============================================================================

function print_results_detalhado(theta, V, Pg_original, Pg_curtailment, Pg_deficit, Pg_total, Qg)
    println("\n" * "="^120)
    println("RESULTADOS FINAIS DO OPF DC COM GERADORES EÓLICOS E CURTAILMENT")
    println("="^120)
    println("Barra |   V (pu)   |  Ang (graus)  | Tipo")
    # Pg_orig (pu) | Pg_curt (pu) | Pg_def (pu) | Pg_total (pu) | Qg (pu)  |
    #$(lpad(p_orig,10)) | $(lpad(p_curt,9)) | $(lpad(p_def,9)) | $(lpad(p_total,11)) | $(lpad(q,8)) |
    println("-"^120)
    
    n = length(V)
    theta_deg = rad2deg.(theta)
    
    for i in 1:n
        v = round(V[i], digits=2)
        ang = round(theta_deg[i], digits=2)
        # p_orig = round(Pg_original[i], digits=2)
        # p_curt = round(Pg_curtailment[i], digits=2)
        # p_def = round(Pg_deficit[i], digits=2)
        # p_total = round(Pg_total[i], digits=2)
        # q = round(Qg[i], digits=2)
        
        tipo = "PQ"
        if i == slack_idx
            tipo = "SLACK"
        elseif any(BARPG_ORIGINAL .== i)
            tipo = "PV"
        end
        
        # marcador_curt = p_curt > 0.001 ? " 🌬️ " : "   "
        # marcador_def = p_def > 0.001 ? " ⚠️ " : "   "
        #$marcador_curt$marcador_def
        
        println("$(lpad(i,4)) | $(lpad(v,8)) | $(lpad(ang,10)) |  $tipo")
    end
    
    total_pg_original = round(sum(Pg_original), digits=10)
    total_pg_curtailment = round(sum(Pg_curtailment), digits=10)
    total_pg_deficit = round(sum(Pg_deficit), digits=10)
    total_pg = round(sum(Pg_total), digits=10)
    total_pl = round(sum(PLOAD), digits=10)
    total_perdas_val = round(sum(perdas), digits=10)
    
    custo_original = round(sum(CPG_ORIGINAL .* final_PG[1:NGER_ORIGINAL]), digits=10)
    custo_curtailment_calc = round(sum(CPG_CURTAILMENT .* final_PG[NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT]), digits=10)
    custo_deficit_calc = round(sum(CPG_DEFICIT .* final_PG[NGER_ORIGINAL+NGER_CURTAILMENT+1:end]), digits=10)
    custo_total = round(custo_original + custo_curtailment_calc + custo_deficit_calc, digits=10)

    # # Verificação de balanço energético
    balanco = total_pg - total_pl - total_perdas_val - total_pg_curtailment + total_pg_deficit

    println("-"^120)
    println("Total Geração :            $(round(total_pg, digits=7)) pu")
    println("Total Carga:               $(round(total_pl, digits=7)) pu")
    println("Total Perdas:              $(round(total_perdas_val, digits=7)) pu")
    println("Total Curtailment:         $(round(total_pg_curtailment, digits=7)) pu")
    println("Total Déficit:     $(round(total_pg_deficit, digits=7)) pu") 

    println("Custo Total Operação:      $(round(custo_total, digits=5)) USD/h")
    println("  - Custo Geração:         $(round(custo_original, digits=5)) USD/h")
    println("  - Custo Curtailment:     $(round(custo_curtailment_calc, digits=5)) USD/h")
    println("  - Custo Déficit:         $(round(custo_deficit_calc, digits=5)) USD/h")
    

    # if abs(balanco) > 0.001
    #     println("⚠️  ALERTA: Desbalanço energético significativo! Diferença: $(round(balanco, digits=4)) pu")
    # end
    
    # if total_pg_curtailment > 0.001
    #     println("\n🌬️  CURTAILMENT: $(round(total_pg_curtailment, digits=4)) pu de energia eólica cortada")
    #     println("   Custo do curtailment: $custo_curtailment_calc USD/h")
    # end
    
    if total_pg_deficit > 0.01
        println("\n⚠️  ALERTA: Sistema com déficit de geração!")
        println("   Foram necessários $(round(total_pg_deficit, digits=4)) pu de geração de déficit")
    else
        println("\n✓ Sistema operando sem déficit de geração")
    end
    println("="^100)
end

function print_geradores_detalhado()
    println("\n" * "="^100)
    println("DETALHAMENTO DOS GERADORES")
    println("="^100)
    println("GER_id | BAR | Tipo | Pg (pu) | Pmin (pu) | Pmax (pu) | Custo (USD/MWh)")
    println("-"^100)
    
    for g in 1:NGER
        barra = BARPG[g]
        pg_val = round(final_PG[g], digits=2)
        pmin = round(PGMIN[g], digits=2)
        pmax = round(PGMAX[g], digits=2)
        custo = round(CPG[g], digits=2)
        
        if g <= NGER_ORIGINAL
            tipo_ger = get(geradores_data[g], "Tipo", "Original")
            tipo = "$tipo_ger"
        elseif g <= NGER_ORIGINAL + NGER_CURTAILMENT
            tipo = "CUR"
        else
            tipo = "DEF"
        end
        
        marcador = ""
        if g > NGER_ORIGINAL && g <= NGER_ORIGINAL + NGER_CURTAILMENT && pg_val > 0.001
            marcador = " 🌬️"
        elseif g > NGER_ORIGINAL + NGER_CURTAILMENT && pg_val > 0.001
            marcador = " ⚠️"
        end
        
        println("$(lpad(g,6)) | $(lpad(barra,4)) | $tipo | $(lpad(pg_val,7)) | $(lpad(pmin,9)) | $(lpad(pmax,9)) | $(lpad(custo,15))$marcador")
    end
end

function print_lagrange()
    println("\n" * "="^60)
    println("ANÁLISE ECONÔMICA - COEFICIENTES DE LAGRANGE")
    println("="^60)
    
    println("\nPREÇOS NODAIS (Custo Marginal - USD/MWh):")
    println("Barra | Preço Nodal")
    println("-"^30)
    for i in 1:NBAR
        println("$(lpad(i,4)) | $(round(lambda_balance[i], digits=6))")
    end
    
    println("\nCUSTOS DE CONGESTIONAMENTO NAS LINHAS:")
    println("Linha | De->Para | Coef. Lagrange")
    println("-"^50)
    for i in 1:NLIN
        idx_fwd = 2*i - 1
        idx_rev = 2*i
        if idx_fwd <= length(lambda_flow) && idx_rev <= length(lambda_flow)
            if abs(lambda_flow[idx_fwd]) > 1e-6 || abs(lambda_flow[idx_rev]) > 1e-6
                fr = line_fr[i]
                to = line_to[i]
                println("$(lpad(i,4)) | $(lpad(fr,2))->$(lpad(to,2))   | $(float(round(lambda_flow[idx_fwd], digits=6)))")
            end
        end
    end
end

function print_fluxos_linhas()
    println("\n" * "="^70)
    println("FLUXOS DE POTÊNCIA NAS LINHAS")
    println("="^70)
    println("Linha |   De->Para   |   Fluxo (pu)   |  Limite (pu) | Utilização | Perdas (pu)")
    println("-"^85)
    
    for i in 1:NLIN
        fr = line_fr[i]
        to = line_to[i]
        fluxo = round(max(abs(fij[i]), abs(fji[i])), digits=2)
        limite = round(FLIM[i], digits=2)
        utilizacao = limite > 0 ? round(fluxo/limite * 100, digits=1) : 0.0
        perda_linha = round(perdas[i], digits=2)
        
        status = utilizacao > 95 ? "CRÍTICO" : "NORMAL"
        
        println("$(lpad(i,4)) | $(lpad(fr,3))->$(lpad(to,3))   | $(lpad(fluxo,8))     | $(lpad(limite,8))   | $(lpad(utilizacao,5))% ($status) | $perda_linha")
    end
end

# ==============================================================================
# EXECUÇÃO DAS IMPRESSÕES
# ==============================================================================

print_results_detalhado(final_ANGLE, V_final, Pg_original, Pg_curtailment, Pg_deficit, Pg_total, Qg_final)
print_geradores_detalhado()
print_lagrange() 
print_fluxos_linhas()

# ==============================================================================
# EXPORTAÇÃO PARA SQLite COM MULTIPLOS CENÁRIOS
# ==============================================================================

using SQLite
using DataFrames
using Dates
using Random

# Gerar ID único para o cenário
function gerar_id_cenario()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    random_id = randstring(6)  # 6 caracteres aleatórios
    return "CEN_$(timestamp)_$(random_id)"
end

# Configuração do cenário atual 
ID_CENARIO = gerar_id_cenario()
NOME_SIMULACAO = "OPF_DC_Curtailment"
DATA_SIMULACAO = now()

function exportar_para_sqlite()
    arquivo_db = "resultados_opf_series.db"
    
    # Calcular totais
    total_pg = sum(Pg_total)
    total_pl = sum(PLOAD)
    total_perdas_val = sum(perdas)
    total_pg_curtailment = sum(Pg_curtailment)
    total_pg_deficit = sum(Pg_deficit)
    
    custo_original = sum(CPG_ORIGINAL .* final_PG[1:NGER_ORIGINAL])
    custo_curtailment_calc = sum(CPG_CURTAILMENT .* final_PG[NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT])
    custo_deficit_calc = sum(CPG_DEFICIT .* final_PG[NGER_ORIGINAL+NGER_CURTAILMENT+1:end])
    custo_total = custo_original + custo_curtailment_calc + custo_deficit_calc

    db = SQLite.DB(arquivo_db)
    
    # ==========================================================================
    # TABELA: CENARIOS (Metadados) - SEMPRE INSERE NOVO
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS cenarios (
            id_cenario TEXT PRIMARY KEY,
            nome_simulacao TEXT,
            data_simulacao DATETIME,
            n_barras INTEGER,
            n_geradores INTEGER,
            n_linhas INTEGER,
            total_geracao_pu REAL,
            total_carga_pu REAL,
            total_perdas_pu REAL,
            custo_total_usd_h REAL,
            status TEXT,
            timestamp_criacao DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # SEMPRE INSERIR NOVO (não replace)
    SQLite.execute(db, """
        INSERT INTO cenarios (
            id_cenario, nome_simulacao, data_simulacao, n_barras, n_geradores, 
            n_linhas, total_geracao_pu, total_carga_pu, total_perdas_pu, 
            custo_total_usd_h, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [ID_CENARIO, NOME_SIMULACAO, DATA_SIMULACAO, NBAR, NGER, NLIN, 
          total_pg, total_pl, total_perdas_val, custo_total, 
          total_pg_deficit > 0.01 ? "COM_DEFICIT" : "NORMAL"])
    
    # ==========================================================================
    # TABELA: BARRAS - SEMPRE NOVOS REGISTROS
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS barras (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_cenario TEXT,
            id_barra TEXT,
            tipo TEXT,
            tensao_pu REAL,
            angulo_graus REAL,
            carga_pu REAL,
            preco_nodal_usd_mwh REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_cenario) REFERENCES cenarios (id_cenario)
        )
    """)
    
    for i in 1:length(barras)
        barra = barras[i]
        SQLite.execute(db, """
            INSERT INTO barras (id_cenario, id_barra, tipo, tensao_pu, angulo_graus, carga_pu, preco_nodal_usd_mwh)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [ID_CENARIO, barra["ID_Barra"], barra["tipo"], 
              round(V_final[i], digits=6), round(rad2deg(final_ANGLE[i]), digits=6),
              round(PLOAD[i], digits=6), round(lambda_balance[i], digits=6)])
    end
    
    # ==========================================================================
    # TABELA: GERADORES - SEMPRE NOVOS REGISTROS
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS geradores (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_cenario TEXT,
            id_gerador TEXT,
            id_barra TEXT,
            tipo TEXT,
            geracao_pu REAL,
            pmin_pu REAL,
            pmax_pu REAL,
            custo_marginal_usd_mwh REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_cenario) REFERENCES cenarios (id_cenario)
        )
    """)
    
    for g in 1:length(final_PG)
        barra_idx = BARPG[g]
        id_barra = barras[barra_idx]["ID_Barra"]
        
        if g <= length(geradores_data)
            gerador = geradores_data[g]
            tipo = gerador["Tipo"]
            id_gerador = gerador["ID_Gerador"]
        elseif g <= length(geradores_data) + length(BAR_GWD)
            tipo = "CUR"
            id_gerador = "CUR_$id_barra"
        else
            tipo = "DEF"
            id_gerador = "DEF_$id_barra"
        end
        
        SQLite.execute(db, """
            INSERT INTO geradores (id_cenario, id_gerador, id_barra, tipo, geracao_pu, pmin_pu, pmax_pu, custo_marginal_usd_mwh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_CENARIO, id_gerador, id_barra, tipo,
              round(final_PG[g], digits=6), round(PGMIN[g], digits=6),
              round(PGMAX[g], digits=6), round(CPG[g], digits=6)])
    end
    
    # ==========================================================================
    # TABELA: LINHAS - SEMPRE NOVOS REGISTROS
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS linhas (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_cenario TEXT,
            id_linha TEXT,
            id_barra_origem TEXT,
            id_barra_destino TEXT,
            fluxo_max_pu REAL,
            limite_pu REAL,
            fluxo_origem_destino_pu REAL,
            fluxo_destino_origem_pu REAL,
            perdas_pu REAL,
            custo_congestionamento_usd_mwh REAL,
            utilizacao_percentual REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_cenario) REFERENCES cenarios (id_cenario)
        )
    """)
    
    for e in 1:length(linhas)
        linha = linhas[e]
        
        idx_fwd = 2*e - 1
        custo_congestionamento = idx_fwd <= length(lambda_flow) && abs(lambda_flow[idx_fwd]) > 1e-6 ? lambda_flow[idx_fwd] : 0.0
        
        fluxo_max = max(abs(fij[e]), abs(fji[e]))
        utilizacao_percentual = FLIM[e] > 0 ? round(fluxo_max / FLIM[e] * 100, digits=2) : 0.0
        
        SQLite.execute(db, """
            INSERT INTO linhas (id_cenario, id_linha, id_barra_origem, id_barra_destino, fluxo_max_pu, limite_pu, fluxo_origem_destino_pu, fluxo_destino_origem_pu, perdas_pu, custo_congestionamento_usd_mwh, utilizacao_percentual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [ID_CENARIO, linha["ID_linha"], linha["ID_Barra_Origem"], linha["ID_Barra_Destino"],
              round(fluxo_max, digits=6), round(FLIM[e], digits=6),
              round(fij[e], digits=6), round(fji[e], digits=6),
              round(perdas[e], digits=6), round(custo_congestionamento, digits=6),
              utilizacao_percentual])
    end
    
    # ==========================================================================
    # TABELA: INDICADORES - SEMPRE NOVOS REGISTROS
    # ==========================================================================
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS indicadores (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_cenario TEXT,
            indicador TEXT,
            valor REAL,
            unidade TEXT,
            categoria TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_cenario) REFERENCES cenarios (id_cenario)
        )
    """)
    
    indicadores = [
        # Custos
        ("custo_total", custo_total, "USD/h", "custo"),
        ("custo_geracao", custo_original, "USD/h", "custo"),
        ("custo_curtailment", custo_curtailment_calc, "USD/h", "custo"),
        ("custo_deficit", custo_deficit_calc, "USD/h", "custo"),
        
        # Energia
        ("total_geracao", total_pg, "pu", "energia"),
        ("total_carga", total_pl, "pu", "energia"),
        ("total_perdas", total_perdas_val, "pu", "energia"),
        ("total_curtailment", total_pg_curtailment, "pu", "energia"),
        ("total_deficit", total_pg_deficit, "pu", "energia"),
        
        # Eficiência
        ("eficiencia_sistema", total_pl / max(total_pg, 0.001), "percentual", "eficiencia"),
        ("fator_perdas", total_perdas_val / max(total_pg, 0.001), "percentual", "eficiencia"),
        
        # Congestionamento
        ("linhas_congestionadas", count(e -> FLIM[e] > 0 && max(abs(fij[e]), abs(fji[e])) / FLIM[e] > 0.9, 1:NLIN), "unidades", "congestionamento"),
        
        # Diversidade
        ("participacao_eolica", sum(final_PG[findall(g -> geradores_data[g]["Tipo"] == "GWD", 1:NGER_ORIGINAL)]) / max(total_pg, 0.001), "percentual", "mix_energetico")
    ]
    
    for (nome, valor, unidade, categoria) in indicadores
        SQLite.execute(db, """
            INSERT INTO indicadores (id_cenario, indicador, valor, unidade, categoria)
            VALUES (?, ?, ?, ?, ?)
        """, [ID_CENARIO, nome, round(valor, digits=6), unidade, categoria])
    end
    
    # ==========================================================================
    # VERIFICAÇÃO E RELATÓRIO
    # ==========================================================================
    
    # Contar cenários existentes
    resultado = SQLite.DBInterface.execute(db, "SELECT COUNT(*) as total_cenarios FROM cenarios") |> DataFrame
    total_cenarios = resultado[1, :total_cenarios]
    
    println("\n🎯 NOVO CENÁRIO CRIADO: $ID_CENARIO")
    println("📊 Arquivo: $arquivo_db")
    println("📈 Total de cenários no banco: $total_cenarios")
    
    # Estatísticas do cenário atual
    println("\n📋 ESTATÍSTICAS DO CENÁRIO ATUAL:")
    println("   • Barras: $NBAR")
    println("   • Geradores: $NGER")
    println("   • Linhas: $NLIN")
    println("   • Custo Total: \$$(round(custo_total, digits=2))/h")
    println("   • Geração Total: $(round(total_pg, digits=4)) pu")
    println("   • Curtailment: $(round(total_pg_curtailment, digits=4)) pu")
    println("   • Déficit: $(round(total_pg_deficit, digits=4)) pu")
    println("   • Eficiência: $(round(total_pl/total_pg*100, digits=1))%")
    
    SQLite.close(db)
    
    return ID_CENARIO
end

# Executar exportação (SEMPRE CRIA NOVO REGISTRO)
id_cenario_atual = exportar_para_sqlite()

println("\n=== NOVO REGISTRO CRIADO COM SUCESSO ===")
println("ID do Cenário: $id_cenario_atual")


println("\n=== EXECUÇÃO CONCLUÍDA ===")
