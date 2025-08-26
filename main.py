import subprocess
import sys
import os

# Caminho para o script Julia
scripts = ["fluxpot_NR","fluxpot_otimo","fluxpot_injecao_correntes"]

for script in scripts:
# Verificar se o arquivo existe
    script_julia = "SRC/modelos_matematicos/"+script+".jl"
    if not os.path.exists(script_julia):
        print(f"Arquivo Julia '{script_julia}' não encontrado.")
        sys.exit(1)

    # Comando para executar
    comando = ["julia", script_julia]

    print(f"Executando '{script_julia}' via subprocesso...")

    # Executa e captura a saída
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
        print("Execução concluída com sucesso:")
        print(resultado.stdout)
    except subprocess.CalledProcessError as e:
        print("Erro durante a execução do script Julia:")
        print(e.stderr)
