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
data = JSON.parsefile("DATA/input/3barras_BASE.json")
#data = JSON.parsefile("DATA/input/B6L8_BASE.json")
#data = JSON.parsefile("DATA/input/ieee14_BASE.json")
#data = JSON.parsefile("DATA/input/IEEE_118_BASE.json")

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
    
    FLIM[e] = get(ln, "LIM_Fluxo", 0.0)
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
    
    if tipo_ger == "UTE"
        PGMIN_ORIGINAL[i] = get(g, "PGERmin_MW", 0.0) / PB
        PGMAX_ORIGINAL[i] = get(g, "PGERmax_MW", 0.0) / PB
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB

        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        PGMIN_EFETIVO[i] = PGMIN_ORIGINAL[i]
        
    elseif tipo_ger == "UTH"
        PGMIN_ORIGINAL[i] = get(g, "PGERmin_MW", 0.0) / PB
        PGMAX_ORIGINAL[i] = get(g, "PGERmax_MW", 1.0) / PB
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB

        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        PGMIN_EFETIVO[i] = PGMIN_ORIGINAL[i]
        
    elseif tipo_ger == "GWD"
        PGMIN_ORIGINAL[i] = get(g, "PGERmin", 0.0) / PB
        PGMAX_ORIGINAL[i] = get(g, "PGERmax", 0.0) / PB
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB

        # Parâmetros da distribuição de Weibull para a velocidade do vento
        lambda_weibull = 8.0  # parâmetro de escala (velocidade média do vento em m/s)
        fator_de_forma_weibull = 2.0  # parâmetro de forma - valor típico para vento

        # Cria a distribuição de Weibull
        distribuicao_weibull = Weibull(lambda_weibull, fator_de_forma_weibull)
        velocidade_vento = rand(distribuicao_weibull)

        # # CURVA DE POTÊNCIA EÓLICA - cálculo da potência instantânea
        # if velocidade_vento < 0.0
        #     # Velocidade de corte inferior - turbina não gera
        #     potencia_instantanea = 0.0
        # elseif velocidade_vento >= 0.0 && velocidade_vento < 10.0
        #     # # Região de operação normal - potência proporcional ao cubo da velocidade
        #     # # Potência = 0.5 * densidade_ar * area_varredura * coeficiente_performance * velocidade^3
        #     # # Simplificando: assumimos relação cúbica normalizada
        #     # potencia_normalizada = ((velocidade_vento - 3.0) / (12.0 - 3.0))^3
        #     # println(potencia_normalizada)
        #     # potencia_instantanea = potencia_normalizada * PGMAX_ORIGINAL[i]
            
        # elseif velocidade_vento >= 12.0 && velocidade_vento <= 25.0
        #     # Velocidade nominal - geração máxima
        #     potencia_instantanea = PGMAX_ORIGINAL[i]
        # else
        #     # Velocidade de corte superior - turbina para por segurança
        #     potencia_instantanea = 0.0
        # end

        potencia_normalizada = rand(Normal(1,0.5))
        potencia_instantanea = potencia_normalizada * PGMAX_ORIGINAL[i]
        println(potencia_instantanea)
        # Garantir que não exceda os limites físicos do gerador
        PGMAX_EFETIVO[i] = min(max(potencia_instantanea, 0.0), PGMAX_ORIGINAL[i])        
        PGMIN_EFETIVO[i] = 0.0

    else
        PGMIN_ORIGINAL[i] = 0.0
        PGMAX_ORIGINAL[i] = 1.0
        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        CPG_ORIGINAL[i] = 0.0
    end
end

# ==============================================================================
# DADOS DAS DEMANDAS
# ==============================================================================

PLOAD = zeros(NBAR)
for d in demandas_data
    id_barra = d["ID_Barra"]
    idx = idx_map[id_barra]

    # Potência ativa nominal
    potencia_demanda = get(d, "PLOAD", 0.0)

    # Desvio padrão
    σ = 0.03 #13

    # Fator de potência constante da barra
    fator_potencia = get(d, "FP", 0.95)

    # Distribuição truncada para garantir valores positivos dentro de NPi ± 3σ
    dist = Normal(potencia_demanda,σ)

    # Sorteio da potência ativa
    potencia_instantanea_incerta = max(0,rand(dist))

    # Normalização
    PLOAD[idx] += potencia_instantanea_incerta / PB
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
    # RESTRIÇÕES DE BALANÇO DE POTÊNCIA COM CURTAILMENT
    # ==========================================================================

    balance_constraints = @constraint(model, balance[i=1:NBAR],
    # GERAÇÃO CONVENCIONAL (UTE, UTH)
    sum(v_PG[g] for g in 1:NGER_ORIGINAL if BARPG[g] == i && geradores_data[g]["Tipo"] != "GWD") +
    # GERAÇÃO EÓLICA LÍQUIDA (GWD - Curtailment)
    sum(v_PG[g] for g in 1:NGER_ORIGINAL if BARPG[g] == i && geradores_data[g]["Tipo"] == "GWD") -
    sum(v_PG[g] for g in NGER_ORIGINAL+1:NGER_ORIGINAL+NGER_CURTAILMENT if BARPG[g] == i) +
    # DÉFICIT
    sum(v_PG[g] for g in NGER_ORIGINAL+NGER_CURTAILMENT+1:NGER if BARPG[g] == i) -
    sum(Bbus[i,j] * v_ANG[j] for j in 1:NBAR) == PLOAD[i] + PINJ[i])
    
    # ==========================================================================
    # RESTRIÇÕES ESPECÍFICAS PARA GERADORES EÓLICOS
    # ==========================================================================
    
    # Para cada gerador eólico: Geração Real + Curtailment = Geração Disponível
    for g in geradores_data
        if g["Tipo"] == "GWD"
            barra_idx = idx_map[g["ID_Barra"]]
            g_idx = findfirst(i -> BARPG_ORIGINAL[i] == barra_idx, 1:NGER_ORIGINAL)
            c_idx = findfirst(i -> BARPG_CURTAILMENT[i] == barra_idx, 1:NGER_CURTAILMENT)            
            if g_idx !== nothing && c_idx !== nothing
                @constraint(model, 
                    v_PG[g_idx] + v_PG[NGER_ORIGINAL + c_idx] == PGMAX_EFETIVO[g_idx]
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
    println("Gerador | Barra |   Tipo    | Pg (pu) | Pmin (pu) | Pmax (pu) | Custo (USD/MWh)")
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

println("\n=== ANÁLISE CONCLUÍDA ===")