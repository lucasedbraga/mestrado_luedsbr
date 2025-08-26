# Caminho do arquivo JSON
caso_base = 'DATA\input\ieee14_BASE.json'

import json
import random

def gerar_cenarios(caminho_arquivo, num_cenarios):
    for cen in range(1,num_cenarios+1):
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            barras = dados.get("BARRAS", [])

            print(f"Gerando Cenário {cen}/{num_cenarios}\n")
            
            for barra in barras:

                # Alterar P_inst
                valor_P = barra.get("P_inst", 0.0)
                novo_P = round(random.uniform(valor_P * 0.6, valor_P * 1.4), 4)
                barra["P_inst"] = novo_P

                # Alterar Custo_P_inst
                valor_custo = barra.get("Custo_P_inst", 0.0)
                novo_custo = round(random.uniform(valor_custo * 0.6, valor_custo * 1.4), 2)
                barra["Custo_P_inst"] = novo_custo

            # Salvar em novo arquivo
            with open(f'DATA\input\cenarios\IEEE_14_Barras\ieee14_cenario_{cen}.json', 'w', encoding='utf-8') as f_out:
                json.dump(dados, f_out, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Erro: {e}")

# Executa a função
dados_modificados = gerar_cenarios(caso_base, num_cenarios=15)
