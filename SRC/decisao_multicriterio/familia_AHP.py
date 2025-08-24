import json
import pandas as pd
import numpy as np

try:
    import fireducks.pandas as pd
except:
    pass

class AHP:
    
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
    
    def rank_alternativas(self):
        # Exemplo de ranking simples baseado em soma ponderada
        matriz_criterios = self.monta_matriz_avaliacao_criterios()
        colunas_criterios = ["Custo Operacao", "Emissao ton CO2"]
        # Calcula os pesos dos critérios
        pesos = matriz_criterios.sum(axis=1) / matriz_criterios.sum().sum()
        
        # Normaliza os valores das alternativas
        df_normalizado = self.alternativas.copy()
        df_normalizado["Custo Operacao"] = df_normalizado["Custo Operacao"] / df_normalizado["Custo Operacao"].max()
        df_normalizado["Emissao ton CO2"] = df_normalizado["Emissao ton CO2"] / df_normalizado["Emissao ton CO2"].max()
        
        # Calcula o score ponderado
        df_normalizado['Score'] = (df_normalizado["Custo Operacao"]  * pesos['Custo Operacao'] +
                                   df_normalizado["Emissao ton CO2"] * pesos['Emissao ton CO2'])
        
        df_resultados = df_normalizado.sort_values(by='Score', ascending=False)
        
        colunas_criterios = colunas_criterios + ['Score']
        # elimina duplicados com base apenas nos critérios
        df_unico = df_resultados.drop_duplicates(subset=colunas_criterios, keep="first").reset_index(drop=True)
        return df_unico

class AHP_Gaussiano(AHP):

    def __init__(self, arquivo_json='DATA/output/input_alternativas.json'):
        super().__init__(arquivo_json)
    
    def rank_alternativas(self):
        df = self.alternativas.copy()
        # Critérios a serem considerados
        criterios = ["Custo Operacao", "Emissao ton CO2"]

        # 1) Normalização por valor máximo (benefício: menor melhor, já invertido se necessário)
        norm = df[criterios] / df[criterios].max()

        # 2) Média e desvio padrão dos critérios
        media = norm.mean()
        desvio = norm.std(ddof=0)  # populational std

        # 3) Fator gaussiano para cada critério
        # Usamos a média e desvio em toda a coluna, mas aplica individual?
        # Simplificação: w_i = exp(-σ_i^2)
        w = np.exp(-desvio**2)

        # Normaliza pesos dos critérios
        w_norm = w / w.sum()

        # 4) Calcula Score
        df["Score"] = norm.dot(w_norm)

        # 5) Ordena e remove duplicatas nos critérios + Score
        df = df.sort_values(by="Score", ascending=False)
        df = df.drop_duplicates(subset=criterios + ["Score"], keep="first").reset_index(drop=True)

        print("Pesos dos critérios (Gaussian):")
        print(w_norm)

        return df


if __name__ == "__main__":
    ahp = AHP()
    ranking_ahp = ahp.rank_alternativas()
    print("\nRanking das Alternativas: - AHP")
    print(ranking_ahp)

    ahp_gaussiano = AHP_Gaussiano()
    ranking_ahp_g = ahp_gaussiano.rank_alternativas()
    print("\nRanking das Alternativas: - AHP Gaussiano")
    print(ranking_ahp_g)