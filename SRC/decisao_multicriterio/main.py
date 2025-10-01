import sys
import os
import subprocess
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decisao_multicriterio.MCDA_DecisaoSubjetiva import *
from decisao_multicriterio.MCDA_DecisaoDados import *



def exporta_comparativo_excel():
    # Caminho para o script Julia
    script = "pareto_despacho_economico"    
    script_julia = "SRC/modelos_matematicos/"+script+".jl"

    if not os.path.exists(script_julia):
        print(f"Arquivo Julia '{script_julia}' não encontrado.")
        sys.exit(1)
    else:
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


    # Caminho do arquivo
    path = "DATA/output/input_alternativas.json"

    with open(path, "r") as f:
        DATA = json.load(f)



    # Matriz de critérios para WASPAS (exemplo simplificado)
    matriz_preferencias_decisor = pd.DataFrame({
        'Custo Operacao': [1, 2],
        'Emissao ton CO2': [1/2, 1]
    }, index=['Custo Operacao','Emissao ton CO2'])


    # Métodos SUBJETIVOS
    ahp = AHP(arquivo_alternativas=DATA, matriz_preferencias_decisor=matriz_preferencias_decisor).rank_alternativas()
    waspas = WASPAS(arquivo_alternativas=DATA, matriz_preferencias_decisor=matriz_preferencias_decisor).rank_alternativas()
    #wisp = WISP(arquivo_alternativas=DATA).rank_alternativas(matriz_criterios=matriz_criterios)
    
    # Métodos ANALÍTICOS
    ahp_gauss = AHP_Gaussiano(arquivo_alternativas=DATA).rank_alternativas()
    lopcow = LOPCOW(arquivo_alternativas=DATA).rank_alternativas()
    mpsi = MPSI(arquivo_alternativas=DATA).rank_alternativas()
    
    # Gera os rankings
    df_ahp = ahp.rename(columns={"Score": "AHP"})
    df_waspas = waspas.rename(columns={"Score": "WASPAS"})
    #df_wisp = wisp.rename(columns={"Score": "WISP"})
    df_ahp_gauss = ahp_gauss.rename(columns={"Score": "AHP_Gaussiano"})
    df_lopcow = lopcow.rename(columns={"Score": "LOPCOW"})
    df_mpsi = mpsi.rename(columns={"Score": "MPSI"})

    # Seleciona colunas-chave
    base_cols = ["id_alternativa", "descricao"]

    df_merged =  df_ahp[base_cols + ["AHP"]] \
        .merge(df_waspas[base_cols + ["WASPAS"]], on=base_cols, how="outer") \
        .merge(df_ahp_gauss[base_cols + ["AHP_Gaussiano"]], on=base_cols, how="outer") \
        .merge(df_lopcow[base_cols + ["LOPCOW"]], on=base_cols, how="outer") \
        .merge(df_mpsi[base_cols + ["MPSI"]], on=base_cols, how="outer")       
        #.merge(df_wisp[base_cols + ["WISP"]], on=base_cols, how="outer")

    # Ordena por uma média dos scores
    df_merged = df_merged.fillna(0)
    df_merged["Score_Medio"] = df_merged[["AHP","WASPAS","AHP_Gaussiano","LOPCOW", "MPSI"]].mean(axis=1)
    df_merged = df_merged.sort_values(by="Score_Medio", ascending=False).reset_index(drop=True)

    print('-'*90)
    # Exporta para Excel
    df_merged.to_excel("DATA/output/MCDA/ranking_comparativo_metodos.xlsx", index=False)
    print(df_merged)

if __name__ == '__main__':
    exporta_comparativo_excel()