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

println("=== OPF DC ITERATIVO COM PERDAS ===")

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

for (i, g) in enumerate(geradores_data)
    id_barra = g["ID_Barra"]
    BARPG_ORIGINAL[i] = idx_map[id_barra]
    tipo_ger = g["Tipo"]
    
    if tipo_ger == "UTE"
        PGMIN_ORIGINAL[i] = get(g, "PGERmin_MW", 0.0) / PB
        PGMAX_ORIGINAL[i] = get(g, "PGERmax_MW", 1.0) / PB
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB

        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        
    elseif tipo_ger == "UTH"
        PGMIN_ORIGINAL[i] = get(g, "PGERmin_MW", 0.0) / PB
        PGMAX_ORIGINAL[i] = get(g, "PGERmax_MW", 1.0) / PB
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB

        PGMAX_EFETIVO[i] = PGMAX_ORIGINAL[i]
        
    elseif tipo_ger == "GWD"
        PGMIN_ORIGINAL[i] = get(g, "PGERmin", 0.0) / PB
        PGMAX_ORIGINAL[i] = get(g, "PGERmax", 1.0) / PB
        CPG_ORIGINAL[i] = get(g, "custo_var_USD_MWh", 0.0) * PB

        # Parâmetros do gerador
        PGmax = PGMAX_ORIGINAL[i]

        # Parâmetros da distribuição de Weibull para a velocidade do vento
        lambda_weibull = 2.5 # parâmetro de escala
        fator_de_forma_weibull = 1.3 # parâmetro de forma

        # Cria a distribuição de Weibull
        distribuicao_weibull = Weibull(lambda_weibull, fator_de_forma_weibull)
        velocidade_vento = rand(distribuicao_weibull)
        PGMAX_EFETIVO[i] = velocidade_vento*PGMAX_ORIGINAL[i]

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
CUSTO_DEFICIT = 10.0 * custo_maximo_existente

BARPG_DEFICIT = Vector{Int}(undef, NGER_DEFICIT)
PGMIN_DEFICIT = zeros(NGER_DEFICIT)
PGMAX_DEFICIT = zeros(NGER_DEFICIT)
CPG_DEFICIT = zeros(NGER_DEFICIT)

for (i, b) in enumerate(barras_PQ_sem_gerador)
    id = b["ID_Barra"]
    idx = idx_map[id]
    BARPG_DEFICIT[i] = idx
    PGMIN_DEFICIT[i] = 0.0
    # Limite máximo baseado na carga da barra
    PGMAX_DEFICIT[i] = PLOAD[idx] > 0 ? PLOAD[idx] * 1.5 : 1.0  # 150% da carga como limite
    CPG_DEFICIT[i] = CUSTO_DEFICIT
end

# ==============================================================================
# COMBINA GERADORES ORIGINAIS E DE DÉFICIT
# ==============================================================================

NGER = NGER_ORIGINAL + NGER_DEFICIT
println("Total de geradores: $NGER_ORIGINAL originais + $NGER_DEFICIT de déficit = $NGER")

BARPG = vcat(BARPG_ORIGINAL, BARPG_DEFICIT)
PGMIN = vcat(PGMIN_ORIGINAL, PGMIN_DEFICIT)
PGMAX = vcat(PGMAX_ORIGINAL, PGMAX_DEFICIT)
CPG = vcat(CPG_ORIGINAL, CPG_DEFICIT)

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

    @objective(model, Min, sum(CPG[g] * v_PG[g] for g in 1:NGER))
    
    # ==========================================================================
    # RESOLUÇÃO DO PROBLEMA
    # ==========================================================================

    #println(model)    
    optimize!(model)

    status = termination_status(model)
    println("Status da solução: $status")
    
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
        
        println("Iteração $iter:")
        println("  delta_theta_max = $(round(DIFMAX, digits=6))")
        println("  delta_Perdas = $(round(loss_diff, digits=6))") 
        println("  Perdas = $(round(total_perdas, digits=6))")
        
        # ==========================================================================
        # ATUALIZAÇÃO PARA PRÓXIMA ITERAÇÃO
        # ==========================================================================
        
        global ANGLE = copy(ANGLE_NEW)
        global prev_total_perdas = total_perdas
        global final_PG = copy(PG_new)
        global final_ANGLE = copy(ANGLE_NEW)
    else
        println("Nenhuma solução disponível na iteração $iter")
        # Se não há solução, mantemos os valores anteriores
        DIFMAX = 1.0
        loss_diff = 1.0
    end
    
    # ==========================================================================
    # EXTRAÇÃO DOS MULTIPLICADORES DE LAGRANGE
    # ==========================================================================

    # Coeficiente do balanço de potência - ATUALIZAÇÃO GLOBAL
    try
        for i in 1:NBAR
            lambda_balance[i] = dual(balance_constraints[i])
        end
        println("Multiplicadores de balanço extraídos")
    catch e
        println("Erro ao extrair multiplicadores de balanço: $e")
        # Mantém os valores anteriores em caso de erro
    end

    # Coeficientes das restrições de fluxo
    try
        for i in 1:length(flow_constraints)
            lambda_flow[i] = dual(flow_constraints[i])
        end
    catch e
        println("Erro ao extrair multiplicadores de fluxo: $e")
        # Mantém os valores anteriores em caso de erro
    end
    
    # Critério de convergência
    if has_values(model) && DIFMAX < TOL && loss_diff < TOL
        println("Convergência atingida na iteração $iter")
        break
    elseif iter == NITER_MAX
        println("Número máximo de iterações atingido")
    end
end

# ==============================================================================
# PÓS-PROCESSAMENTO DOS RESULTADOS
# ==============================================================================

Pg_original = zeros(NBAR)
Pg_deficit = zeros(NBAR)
Pg_total = zeros(NBAR)

for g in 1:NGER
    bar_idx = BARPG[g]
    if g <= NGER_ORIGINAL
        Pg_original[bar_idx] += final_PG[g]
    else
        Pg_deficit[bar_idx] += final_PG[g]
    end
    Pg_total[bar_idx] += final_PG[g]
end

V_final = ones(NBAR)
Qg_final = zeros(NBAR)

# ==============================================================================
# FUNÇÕES DE IMPRESSÃO DE RESULTADOS - CORRIGIDAS
# ==============================================================================

function print_results_detalhado(theta, V, Pg_original, Pg_deficit, Pg_total, Qg)
    println("\n" * "="^100)
    println("RESULTADOS FINAIS DO OPF DC COM GERADORES DE DÉFICIT")
    println("="^100)
    println("Barra |   V (pu)   |  Ang (graus)  | Pg_orig (pu) | Pg_def (pu) | Pg_total (pu) | Qg (pu)  | Tipo")
    println("-"^100)
    
    n = length(V)
    theta_deg = rad2deg.(theta)
    
    for i in 1:n
        v = round(V[i], digits=2)
        ang = round(theta_deg[i], digits=2)
        p_orig = round(Pg_original[i], digits=2)
        p_def = round(Pg_deficit[i], digits=2)
        p_total = round(Pg_total[i], digits=2)
        q = round(Qg[i], digits=2)
        
        tipo = "PQ"
        if i == slack_idx
            tipo = "SLACK"
        elseif any(BARPG_ORIGINAL .== i)
            tipo = "PV"
        end
        
        marcador_def = p_def > 0.001 ? " ⚠️ " : "   "
        
        println("$(lpad(i,4)) | $(lpad(v,8)) | $(lpad(ang,10)) | $(lpad(p_orig,10)) | $(lpad(p_def,9)) | $(lpad(p_total,11)) | $(lpad(q,8)) | $tipo$marcador_def")
    end
    
    total_pg_original = round(sum(Pg_original), digits=10)
    total_pg_deficit = round(sum(Pg_deficit), digits=10)
    total_pg = round(sum(Pg_total), digits=10)
    total_pl = round(sum(PLOAD), digits=10)
    total_perdas_val = round(sum(perdas), digits=10)
    custo_original = round(sum(CPG_ORIGINAL .* final_PG[1:NGER_ORIGINAL]), digits=10)
    custo_deficit_calc = round(sum(CPG_DEFICIT .* final_PG[NGER_ORIGINAL+1:end]), digits=10)
    custo_total = round(custo_original + custo_deficit_calc, digits=10)
    
    println("-"^100)
    println("Total Geração Original: $total_pg_original pu")
    println("Total Geração Déficit:  $total_pg_deficit pu") 
    println("Total Geração:          $total_pg pu")
    println("Total Carga:            $total_pl pu") 
    println("Total Perdas:           $total_perdas_val pu")
    println("Custo Geração Original: $custo_original USD/h")
    println("Custo Geração Déficit:  $custo_deficit_calc USD/h")
    println("Custo Total:            $custo_total USD/h")
    
    # Verificação de balanço energético
    balanco = total_pg - total_pl - total_perdas_val
    println("Balanço (Geração - Carga - Perdas): $balanco pu")
    
    if abs(balanco) > 0.001
        println("⚠️  ALERTA: Desbalanço energético significativo!")
    end
    
    if total_pg_deficit > 0.01
        println("\n⚠️  ALERTA: Sistema com déficit de geração!")
        println("   Foram necessários $(round(total_pg_deficit, digits=4)) pu de geração de déficit")
        println("   Custo adicional: $custo_deficit_calc USD/h")
    else
        println("\n✓ Sistema operando sem déficit de geração")
    end
    println("="^100)
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

function print_geradores_detalhado()
    println("\n" * "="^80)
    println("DETALHAMENTO DOS GERADORES")
    println("="^80)
    println("Gerador | Barra |   Tipo    | Pg (pu) | Pmin (pu) | Pmax (pu) | Custo (USD/MWh)")
    println("-"^80)
    
    for g in 1:NGER
        barra = BARPG[g]
        pg_val = round(final_PG[g], digits=2)
        pmin = round(PGMIN[g], digits=2)
        pmax = round(PGMAX[g], digits=2)
        custo = round(CPG[g], digits=2)
        
        tipo = g <= NGER_ORIGINAL ? "Original " : "Déficit  "
        
        marcador = (g > NGER_ORIGINAL && pg_val > 0.001) ? " ⚠️" : ""
        
        println("$(lpad(g,6)) | $(lpad(barra,4)) | $tipo | $(lpad(pg_val,7)) | $(lpad(pmin,9)) | $(lpad(pmax,9)) | $(lpad(custo,15))$marcador")
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

print_results_detalhado(final_ANGLE, V_final, Pg_original, Pg_deficit, Pg_total, Qg_final)
print_geradores_detalhado()
print_lagrange() 
print_fluxos_linhas()

println("\n=== ANÁLISE CONCLUÍDA ===")