import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decisao_multicriterio.familia_AHP import *
from decisao_multicriterio.familia_LesteEuropeu import *



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

    # Instancia os métodos
    ahp = AHP()
    ahp_gauss = AHP_Gaussiano()
    waspas = WASPAS()
    lopcow = LOPCOW()
    mpsi = MPSI()
    wisp = WISP()

    # Gera os rankings
    df_ahp = ahp.rank_alternativas().rename(columns={"Score": "AHP"})
    df_ahp_gauss = ahp_gauss.rank_alternativas().rename(columns={"Score": "AHP_Gaussiano"})
    df_waspas = waspas.rank_alternativas().rename(columns={"Score": "WASPAS"})
    df_lopcow = lopcow.rank_alternativas().rename(columns={"Score": "LOPCOW"})
    df_mpsi = mpsi.rank_alternativas().rename(columns={"Score": "MPSI"})
    df_wisp = wisp.rank_alternativas().rename(columns={"Score": "WISP"})

    # Seleciona colunas-chave
    base_cols = ["id_alternativa", "descricao"]

    # Faz merge sucessivo
    df_merged = df_ahp[base_cols + ["AHP"]] \
        .merge(df_ahp_gauss[base_cols + ["AHP_Gaussiano"]], on=base_cols, how="outer") \
        .merge(df_waspas[base_cols + ["WASPAS"]], on=base_cols, how="outer") \
        .merge(df_lopcow[base_cols + ["LOPCOW"]], on=base_cols, how="outer") \
        .merge(df_mpsi[base_cols + ["MPSI"]], on=base_cols, how="outer") \
        .merge(df_wisp[base_cols + ["WISP"]], on=base_cols, how="outer")

    # Ordena por uma média dos scores (opcional)
    df_merged = df_merged.fillna(0)
    df_merged["Score_Medio"] = df_merged[["AHP", "AHP_Gaussiano", "WASPAS", "LOPCOW", "MPSI","WISP"]].mean(axis=1)
    df_merged = df_merged.sort_values(by="Score_Medio", ascending=False).reset_index(drop=True)

    print('-'*80)
    # Exporta para Excel
    df_merged.to_excel("DATA/output/MCDA/ranking_comparativo_metodos.xlsx", index=False)
    print(df_merged)

if __name__ == '__main__':
    exporta_comparativo_excel()