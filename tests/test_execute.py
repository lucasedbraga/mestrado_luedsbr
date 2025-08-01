import subprocess
import os
import pytest

# Caminho para o script Julia
SCRIPT_JULIA = "SRC/modelos_matematicos/metodo_newton.jl"

def test_script_julia_existe():
    assert os.path.exists(SCRIPT_JULIA), f"Arquivo Julia '{SCRIPT_JULIA}' não encontrado."

def test_execucao_script_julia():
    comando = ["julia", SCRIPT_JULIA]
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
        saida = resultado.stdout

        # Verificações básicas na saída
        assert "Convergiu para x =" in saida
        assert "Executando método de Newton-Raphson" in saida
        assert "ERROR" not in saida

        print("\nSaída do script Julia:")
        print(saida)

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Erro na execução do script Julia:\n{e.stderr}")