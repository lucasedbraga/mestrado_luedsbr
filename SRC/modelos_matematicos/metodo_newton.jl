using SQLite
using DataFrames
using DBInterface
using Printf

# Função para carregar o DataFrame de coeficientes do SQLite3
function carregar_coefs(db_path::String, tabela::String)
    # 1. Conectar ao banco
    db = SQLite.DB(db_path)
    # 2. Montar e executar a consulta
    consulta = "SELECT * FROM $tabela"
    resultado = DBInterface.execute(db, consulta)
    # 3. Transformar em DataFrame
    df = DataFrame(resultado)
    # 4. Fechar conexão
    SQLite.close(db)

    return df
end
# Monta f(x) e f'(x) a partir de coeficientes (polinômio)
# O DataFrame df deve ter colunas: :grau_polinomio (Int) e :coeficiente (Float64)
function montar_funcoes(df::DataFrame)
    println(df)
    powers = Int.(df.grau_polinomio)
    coefs  = df.coeficiente

    f = x -> sum(coefs[i] * x^(powers[i]) for i in 1:length(coefs))
    dfun = x -> sum(coefs[i] * powers[i] * x^(powers[i] - 1) for i in 1:length(coefs))

    return f, dfun
end

# Implementação do método de Newton-Raphson
function newton_raphson(f, dfun, x0::Float64; tol::Float64=1e-8, max_iter::Int=100)
    x = x0
    for k in 1:max_iter
        fx = f(x)
        dfx = dfun(x)
        if abs(dfx) < eps()
            error("Derivada muito próxima de zero na iteração $k, x = $x")
        end
        x_new = x - fx / dfx
        @printf("Iter %3d: x = %.10f, f(x) = %.10e\n", k, x_new, f(x_new))
        if abs(x_new - x) < tol
            return x_new, k
        end
        x = x_new
    end
    error("Não convergiu após $max_iter iterações")
end

# Programa principal
function main()
    db_path = "DATA/input/input_teste.db"
    tabela = "input_teste"
    chute_inicial = 1.0

    println("Carregando coeficientes da tabela '$tabela' em '$db_path'...")
    df = carregar_coefs(db_path, tabela)
    f, dfun = montar_funcoes(df)

    println("Executando método de Newton-Raphson (tol=1e-8, max_iter=100)...")
    raiz, iter = newton_raphson(f, dfun, chute_inicial)

    @printf("\nConvergiu para x = %.10f em %d iterações.\n", raiz, iter)
end

# # Executa quando rodar o script diretamente
# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end
main()