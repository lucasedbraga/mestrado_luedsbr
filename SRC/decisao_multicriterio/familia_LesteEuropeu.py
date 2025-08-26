import json
import pandas as pd
import numpy as np

try:
    import fireducks.pandas as pd
except:
    pass

class WASPAS:
    
    def __init__(self, arquivo_json='DATA/output/input_alternativas.json'):
        self.alternativas = self.leitura_alternativas(arquivo_json)

    def leitura_alternativas(self, arquivo):
        with open(arquivo, 'r') as file:
            alternativas = json.load(file)  # carrega lista de dicts
        df = pd.DataFrame(alternativas)

        # força a ordem das colunas
        colunas = ["id_alternativa", "descricao", "Custo Operacao", "Emissao ton CO2"]
        return df[colunas]
    
    def monta_matriz_avaliacao_criterios(self):
        # Exemplo de matriz de avaliação dos critérios
        matriz = pd.DataFrame({
            'Custo Operacao': [1, 2],
            'Emissao ton CO2': [1/2, 1]
        }, index=['Custo Operacao', 'Emissao ton CO2'])
        return matriz
    
    def rank_alternativas(self, lambda_waspas=0.5):
        matriz_criterios = self.monta_matriz_avaliacao_criterios()
        colunas_criterios = ["Custo Operacao", "Emissao ton CO2"]

        # Calcula os pesos dos critérios
        pesos = matriz_criterios.sum(axis=1) / matriz_criterios.sum().sum()

        # Normaliza os valores das alternativas (quanto menor, melhor)
        df_normalizado = self.alternativas.copy()
        for col in colunas_criterios:
            df_normalizado[f"{col}_norm"] = df_normalizado[col] / df_normalizado[col].max()

        # Calcula WSM (soma ponderada)
        df_normalizado["WSM"] = sum(
            df_normalizado[f"{col}_norm"] * pesos[col] for col in colunas_criterios
        )

        # Calcula WPM (produto ponderado)
        df_normalizado["WPM"] = 1
        for col in colunas_criterios:
            df_normalizado["WPM"] *= df_normalizado[f"{col}_norm"] ** pesos[col]

        # Score WASPAS
        df_normalizado["Score"] = (
            lambda_waspas * df_normalizado["WSM"] +
            (1 - lambda_waspas) * df_normalizado["WPM"]
        )

        # Mantém colunas relevantes, incluindo id e descrição
        df_resultados = df_normalizado[
            ["id_alternativa", "descricao"] + colunas_criterios + ["Score"]
        ].sort_values(by="Score", ascending=False)

        print('-'*80)
        print("Pesos dos critérios WASPAS:")
        print(pesos)

        # Elimina duplicados com base nos critérios + Score
        colunas_criterios_score = colunas_criterios + ['Score']
        df_unico = df_resultados.drop_duplicates(subset=colunas_criterios_score, keep="first").reset_index(drop=True)

        # Normaliza os scores
        df_unico["Score"] = df_unico["Score"] / df_unico["Score"].sum()

        # Exibe o resultado
        print("\nRanking das Alternativas: - WASPAS")
        print(df_unico)

        return df_unico

class LOPCOW(WASPAS):

    def __init__(self, arquivo_json='DATA/output/input_alternativas.json'):
        super().__init__(arquivo_json)
    
    def rank_alternativas(self):
        df = self.alternativas.copy()
        colunas_criterios = ["Custo Operacao", "Emissao ton CO2"]

        # Etapa 1: Normalização (quanto menor, melhor)
        df_normalizado = df.copy()
        for col in colunas_criterios:
            df_normalizado[col] = df[col] / df[col].max()

        # Etapa 2: Cálculo da matriz de proporções
        matriz_proporcao = df_normalizado[colunas_criterios].copy()
        soma_colunas = matriz_proporcao.sum()
        matriz_proporcao = matriz_proporcao / soma_colunas

        # Etapa 3: Cálculo da entropia logarítmica
        import numpy as np
        epsilon = 1e-12  # para evitar log(0)
        entropia = - (matriz_proporcao * np.log(matriz_proporcao + epsilon)).sum() / np.log(len(df))

        # Etapa 4: Grau de divergência
        grau_divergencia = 1 - entropia

        # Etapa 5: Cálculo dos pesos objetivos
        pesos = grau_divergencia / grau_divergencia.sum()

        # Etapa 6: Score final por soma ponderada
        df["Score"] = (df_normalizado[colunas_criterios] * pesos).sum(axis=1)

        # Ordena e normaliza os scores
        df_resultados = df[["id_alternativa", "descricao"] + colunas_criterios + ["Score"]].sort_values(by="Score", ascending=False)
        print('-'*80)
        print("Pesos dos critérios LOPCOW:")
        print(pd.Series(pesos, index=colunas_criterios))

        # Elimina duplicatas com base nos critérios + Score
        df_unico = df_resultados.drop_duplicates(subset=colunas_criterios + ["Score"], keep="first").reset_index(drop=True)
        df_unico["Score"] = df_unico["Score"] / df_unico["Score"].sum()
        
        print("\nRanking das Alternativas: - LOPCOW")
        print(df_unico)

        return df_unico

class MPSI(WASPAS):
    def __init__(self, arquivo_json='DATA/output/input_alternativas.json'):
        super().__init__(arquivo_json)
    
    def rank_alternativas(self):
        df = self.alternativas.copy()
        colunas_criterios = ["Custo Operacao", "Emissao ton CO2"]

        # Normalização (quanto menor, melhor)
        df_normalizado = df.copy()
        for col in colunas_criterios:
            df_normalizado[col] = df[col] / df[col].max()

        # Alternativa ideal (menores valores normalizados)
        ideal = df_normalizado[colunas_criterios].min()

        # Distância euclidiana até a alternativa ideal
        distancias = ((df_normalizado[colunas_criterios] - ideal) ** 2).sum(axis=1) ** 0.5

        # Índice MPSI: quanto menor a distância, maior o score
        df["Score"] = 1 / (1 + distancias)

        # Seleciona colunas relevantes, incluindo id e descrição
        df_resultados = df[["id_alternativa", "descricao"] + colunas_criterios + ["Score"]].sort_values(by="Score", ascending=False)
        print('-'*80)
        # Elimina duplicatas com base nos critérios + Score
        df_unico = df_resultados.drop_duplicates(subset=colunas_criterios + ["Score"], keep="first").reset_index(drop=True)
        # Normaliza os scores
        df_unico["Score"] = df_unico["Score"] / df_unico["Score"].sum()
        
        print('\nRanking das Alternativas: - MPSI')
        print(df_unico)

        return df_unico


class WISP(WASPAS):
    def __init__(self, arquivo_json='DATA/output/input_alternativas.json'):
        super().__init__(arquivo_json)
    
    def rank_alternativas(self):
        df = self.alternativas.copy()
        colunas_criterios = ["Custo Operacao", "Emissao ton CO2"]

        # Normalização (quanto menor, melhor)
        df_normalizado = df.copy()
        for col in colunas_criterios:
            df_normalizado[col] = df[col] / df[col].max()

        # Cálculo dos pesos objetivos via desvio padrão (quanto mais variável, mais peso)
        desvios = df_normalizado[colunas_criterios].std()
        pesos = desvios / desvios.sum()

        # Matriz de influência: diferença entre alternativas
        influencia = pd.DataFrame(0.0, index=df.index, columns=df.index)
        for i in df.index:
            for j in df.index:
                if i != j:
                    diff = df_normalizado.loc[i, colunas_criterios] - df_normalizado.loc[j, colunas_criterios]
                    influencia.loc[i, j] = (diff * pesos).sum()

        # Score WISP: soma das influências positivas
        df["Score"] = influencia[influencia > 0].sum(axis=1)

        # Normaliza e ordena
        df_resultados = df[["id_alternativa", "descricao"] + colunas_criterios + ["Score"]].sort_values(by="Score", ascending=False)
        print('-'*80)
        print("Pesos dos critérios WISP:")
        print(pesos)
        df_unico = df_resultados.drop_duplicates(subset=colunas_criterios + ["Score"], keep="first").reset_index(drop=True)
        df_unico["Score"] = df_unico["Score"] / df_unico["Score"].sum()
        
        # Exibe os resultados
        print('\nRanking das Alternativas: - WISP')
        print(df_unico)
    
        return df_unico


if __name__ == "__main__":
    waspas = WASPAS().rank_alternativas()
    lopcow = LOPCOW().rank_alternativas()
    mpsi = MPSI().rank_alternativas()
    wisp = WISP().rank_alternativas()
