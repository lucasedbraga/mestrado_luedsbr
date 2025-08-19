using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JSON3
using LinearAlgebra
using Printf

arquivo_input = "DATA/input/ieee14_BASE.json"

function ler_dados_sistema(caminho_arquivo)
    json = JSON3.read(open(caminho_arquivo), Dict)
    barras = json["BARRAS"]
    linhas = json["LINHAS"]
    return barras, linhas
end

function construir_Ybus(barras, linhas)
    n = length(barras)
    Ybus = zeros(ComplexF64, n, n)
    
    idx_map = Dict{String, Int}()
    for (i, barra) in enumerate(barras)
        idx_map[barra["ID_Barra"]] = i
    end

    for linha in linhas
        i = idx_map[linha["ID_Barra_Origem"]]
        j = idx_map[linha["ID_Barra_Destino"]]
        r = linha["R"]
        x = linha["X"]
        z = complex(r, x)
        y = 1 / z

        Ybus[i, i] += y
        Ybus[j, j] += y
        Ybus[i, j] -= y
        Ybus[j, i] -= y
    end

    return Ybus
end

function fluxo_potencia_newtonraphson(arquivo_input; tol=1e-6, max_iter=20)
    barras, linhas = ler_dados_sistema(arquivo_input)
    n = length(barras)

    # Índices
    idx_map = Dict(b["ID_Barra"] => i for (i,b) in enumerate(barras))
    tipo_barras = [b["tipo"] for b in barras]
    idx_slack = findfirst(x -> x == "Slack", tipo_barras)
    idx_pv = findall(x -> x == "PV", tipo_barras)
    idx_pq = findall(x -> x == "PQ", tipo_barras)

    # Potências líquidas
    P_liq = [get(b, "P_Gen", 0.0) - get(b, "P", 0.0) for b in barras]
    Q_liq = [get(b, "Q_Gen", 0.0) - get(b, "Q", 0.0) for b in barras]

    # Estado inicial
    θ = zeros(n)
    V = ones(n)
    V[idx_slack] = barras[idx_slack]["V_ref"]
    for i in idx_pv
        V[i] = barras[i]["V_ref"]
    end

    Ybus = construir_Ybus(barras, linhas)
    G = real.(Ybus)
    B = imag.(Ybus)

    for iter = 1:max_iter
        Pcalc = zeros(n)
        Qcalc = zeros(n)

        for i = 1:n
            for j = 1:n
                θ_diff = θ[i] - θ[j]
                Pcalc[i] += V[i]*V[j]*(G[i,j]*cos(θ_diff) + B[i,j]*sin(θ_diff))
                Qcalc[i] += V[i]*V[j]*(G[i,j]*sin(θ_diff) - B[i,j]*cos(θ_diff))
            end
        end

        ΔP = P_liq .- Pcalc
        ΔQ = Q_liq .- Qcalc

        # Vetor de mismatches
        mismatch_P = [ΔP[i] for i in 1:n if i != idx_slack]
        mismatch_Q = [ΔQ[i] for i in idx_pq]
        mismatch = vcat(mismatch_P, mismatch_Q)

        if norm(mismatch, Inf) < tol
            println("Convergiu em $iter iterações.")
            break
        end

        npq = length(idx_pq)
        nθ = n - 1
        J = zeros(nθ + npq, nθ + npq)

        not_slack = [i for i in 1:n if i != idx_slack]

        # H - dP/dθ
        for (k,i) in enumerate(not_slack)
            for (l,j) in enumerate(not_slack)
                if i != j
                    J[k,l] = V[i]*V[j]*(G[i,j]*sin(θ[i]-θ[j]) - B[i,j]*cos(θ[i]-θ[j]))
                else
                    J[k,l] = -Qcalc[i] - V[i]^2*B[i,i]
                end
            end
        end

        # N - dP/dV
        for (k,i) in enumerate(not_slack)
            for (l,j) in enumerate(idx_pq)
                if i != j
                    J[k,nθ+l] = V[i]*(G[i,j]*cos(θ[i]-θ[j]) + B[i,j]*sin(θ[i]-θ[j]))
                else
                    J[k,nθ+l] = Pcalc[i]/V[i] + V[i]*G[i,i]
                end
            end
        end

        # M - dQ/dθ
        for (k,i) in enumerate(idx_pq)
            for (l,j) in enumerate(not_slack)
                if i != j
                    J[nθ+k,l] = -V[i]*V[j]*(G[i,j]*cos(θ[i]-θ[j]) + B[i,j]*sin(θ[i]-θ[j]))
                else
                    J[nθ+k,l] = Pcalc[i] - V[i]^2*G[i,i]
                end
            end
        end

        # L - dQ/dV
        for (k,i) in enumerate(idx_pq)
            for (l,j) in enumerate(idx_pq)
                if i != j
                    J[nθ+k,nθ+l] = V[i]*(G[i,j]*sin(θ[i]-θ[j]) - B[i,j]*cos(θ[i]-θ[j]))
                else
                    J[nθ+k,nθ+l] = Qcalc[i]/V[i] - V[i]*B[i,i]
                end
            end
        end

        Δx = J \ mismatch

        for (k,i) in enumerate(not_slack)
            θ[i] += Δx[k]
        end
        for (k,i) in enumerate(idx_pq)
            V[i] += Δx[nθ+k]
        end
    end

    # Calculo de Geração na Slack
    Pg_slack = 0.0
    Qg_slack = 0.0
    
    for j = 1:n
        θ_diff = θ[idx_slack] - θ[j]
        Pg_slack += V[idx_slack]*V[j]*(G[idx_slack,j]*cos(θ_diff) + B[idx_slack,j]*sin(θ_diff))
        Qg_slack += V[idx_slack]*V[j]*(G[idx_slack,j]*sin(θ_diff) - B[idx_slack,j]*cos(θ_diff))
    end
    
    # Atualizando os vetores de geração
    Pg = zeros(n)
    Qg = zeros(n)
    Pg[idx_slack] = Pg_slack + barras[idx_slack]["P"]
    Qg[idx_slack] = Qg_slack + barras[idx_slack]["Q"]
    
    # Calculando geração nas barras PV
    for i in idx_pv
        Pg[i] = barras[i]["P_Gen"]
        Qg[i] = 0.0
        for j = 1:n
            θ_diff = θ[i] - θ[j]
            Qg[i] += V[i]*V[j]*(G[i,j]*sin(θ_diff) - B[i,j]*cos(θ_diff))
        end
        Qg[i] += barras[i]["Q"]  # Adiciona a demanda reativa
    end

    return θ, V, Pg, Qg
end

function print_results(θ, V, Pg, Qg)
    println("\nResultados do Fluxo de Potência Newthon Raphson:")
    println("Barra |   V (pu)   |  Ang (graus)  | Pg (pu)   | Qg (pu)  ")
    
    # Garante que todos sejam arrays do mesmo tamanho
    n = length(V)
    θ = typeof(θ) <: Number ? fill(θ, n) : θ
    Pg = typeof(Pg) <: Number ? fill(Pg, n) : Pg
    Qg = typeof(Qg) <: Number ? fill(Qg, n) : Qg
    
    for i in 1:n
        v = round(V[i], digits=4)
        ang = round(rad2deg(θ[i]), digits=2)
        p = round(Pg[i], digits=4)
        q = round(Qg[i], digits=4)
        
        println("$(lpad(i,4)) | $(lpad(v,8)) | $(lpad(ang,10)) | $(lpad(p,8)) | $(lpad(q,8)) ")
    end
end

# Uso:
θ, V, Pg, Qg = fluxo_potencia_newtonraphson(arquivo_input)
print_results(θ, V, Pg, Qg)