import sys
import os
import subprocess


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

    # # Instancia os métodos
    # ahp = AHP()
    # ahp_gauss = AHP_Gaussiano()

    DATA = """
[{"id_alternativa":1,"descricao":"w_c=1.0, w_e=0.0","Custo Operacao":[5643.6,"MIN"],"Emissao ton CO2":[1785.05,"MIN"]},{"id_alternativa":2,"descricao":"w_c=0.9, w_e=0.1","Custo Operacao":[6354.8,"MIN"],"Emissao ton CO2":[1518.35,"MIN"]},{"id_alternativa":3,"descricao":"w_c=0.6, w_e=0.4","Custo Operacao":[14783.2,"MIN"],"Emissao ton CO2":[921.68,"MIN"]},{"id_alternativa":4,"descricao":"w_c=0.4, w_e=0.6","Custo Operacao":[18373.2,"MIN"],"Emissao ton CO2":[803.18,"MIN"]},{"id_alternativa":5,"descricao":"w_c=0.3, w_e=0.7","Custo Operacao":[19949.2,"MIN"],"Emissao ton CO2":[763.78,"MIN"]}]
"""
    # Matriz de critérios para WASPAS (exemplo simplificado)
    matriz_criterios = pd.DataFrame({
        'Custo Operacao': [1, 2],
        'Emissao ton CO2': [1/2, 1]
    })


    # Métodos ANALÍTICOS
    ahp_gauss = AHP_Gaussiano(arquivo_alternativas=DATA).rank_alternativas()
    lopcow = LOPCOW(arquivo_alternativas=DATA).rank_alternativas()
    mpsi = MPSI(arquivo_alternativas=DATA).rank_alternativas()

    # Métodos SUBJETIVOS
    ahp = AHP(arquivo_alternativas= DATA).rank_alternativas(matriz_criterios=matriz_criterios)
    waspas = WASPAS(arquivo_alternativas= DATA).rank_alternativas(matriz_criterios=matriz_criterios)
    wisp = WISP(arquivo_alternativas=DATA).rank_alternativas(matriz_criterios=matriz_criterios)

    # Gera os rankings
    df_ahp = ahp.rename(columns={"Score": "AHP"})
    df_waspas = waspas.rename(columns={"Score": "WASPAS"})
    df_wisp = wisp.rename(columns={"Score": "WISP"})
    df_ahp_gauss = ahp_gauss.rename(columns={"Score": "AHP_Gaussiano"})
    df_lopcow = lopcow.rename(columns={"Score": "LOPCOW"})
    df_mpsi = mpsi.rename(columns={"Score": "MPSI"})
    # Seleciona colunas-chave
    base_cols = ["id_alternativa", "descricao"]


    df_merged = df_ahp_gauss[base_cols + ["AHP_Gaussiano"]] \
        .merge(df_lopcow[base_cols + ["LOPCOW"]], on=base_cols, how="outer") \
        .merge(df_mpsi[base_cols + ["MPSI"]], on=base_cols, how="outer") \
        .merge(df_ahp[base_cols + ["AHP"]], on=base_cols, how="outer") \
        .merge(df_waspas[base_cols + ["WASPAS"]], on=base_cols, how="outer") \
        # .merge(df_wisp[base_cols + ["WISP"]], on=base_cols, how="outer")

    # Ordena por uma média dos scores
    df_merged = df_merged.fillna(0)
    df_merged["Score_Medio"] = df_merged[["AHP_Gaussiano","LOPCOW", "MPSI","AHP","WASPAS"]].mean(axis=1)
    df_merged = df_merged.sort_values(by="Score_Medio", ascending=False).reset_index(drop=True)

    print('-'*80)
    # Exporta para Excel
    df_merged.to_excel("DATA/output/MCDA/ranking_comparativo_metodos.xlsx", index=False)
    print(df_merged)

if __name__ == '__main__':
    exporta_comparativo_excel()