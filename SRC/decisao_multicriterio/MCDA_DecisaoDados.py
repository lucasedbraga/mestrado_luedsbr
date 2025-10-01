import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

try:
    import fireducks.pandas as pd
except:
    pass


class BaseMCDA_DADOS(ABC):
    """
    Classe base para todos os métodos MCDA:

    - Leitura de alternativas a partir de JSON
    - Estrutura para normalização
    - Colunas de critérios definidas
    - Método abstrato 'rank_alternativas'
    """

    def __init__(self, arquivo_alternativas=None):
        """       
        Args:
            arquivo_alternativas: Dicionário ou JSON com dados
        """
        # Carrega dados
        if isinstance(arquivo_alternativas, dict):
            data = arquivo_alternativas
        else:
            data = json.loads(arquivo_alternativas)
        
        # Processa dados das alternativas
        df_alternativas, criterios_list, tipo_criterio_list = self.leitura_alternativas(data)
        self.alternativas = df_alternativas
        self.colunas_criterios = criterios_list
        self.tipo_criterio_list = tipo_criterio_list


    def leitura_alternativas(self, data):
        """Lê dados das alternativas e critérios"""
        criterios_dict = data["criterios"]
        criterios_list = list(criterios_dict.keys())
        tipo_criterio_list = criterios_dict
        
        alternativas_data = data["alternativas"]
        df_alternativas = pd.DataFrame(alternativas_data)
        
        return df_alternativas, criterios_list, tipo_criterio_list
    
    def normalizar(self, df):
        df_norm = df.copy()
        for col in self.colunas_criterios:
            if self.tipo_criterio_list[col] == 'MAX':
                # Normalização para critérios de benefício (quanto maior melhor)
                df_norm[col] = df[col] / df[col].max()
            else:
                # Normalização para critérios de custo (quanto menor melhor)
                df_norm[col] = df[col].min() / df[col]
        return df_norm

    @abstractmethod
    def rank_alternativas(self):
        pass


### ESCOLA AMERICANA

class AHP_Gaussiano(BaseMCDA_DADOS):
    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()
        for col in df_norm.columns:
            if col in self.tipo_criterio_list and self.tipo_criterio_list[col] in ['MAX', 'MIN']:
                if self.tipo_criterio_list[col] == 'MAX':
                    pass
                else:
                    # Normalização para critérios de custo (quanto menor melhor)
                    df_norm[col] = (1) / (df[col])
        return df_norm
    
    def rank_alternativas(self):
        # Normalização da Matriz de Desempenho
        df = self.alternativas.copy()
        matriz_desempenho = df[self.colunas_criterios]
        matriz_desempenho = self.normalizar(matriz_desempenho)
        matriz_desempenho = matriz_desempenho / matriz_desempenho.sum()
        # 2) Média e desvio padrão dos critérios
        media = matriz_desempenho.mean()
        desvio = matriz_desempenho.std()

        # 3) Fator gaussiano para cada critério
        # Usamos a média e desvio em toda a coluna, mas aplica individual?
        # Simplificação: w_i = exp(-σ_i^2)
        w = desvio/media

        # Normaliza pesos dos critérios
        pesos = w / w.sum()

        # 4) Calcula Score
        df["Score"] = matriz_desempenho.dot(pesos)

        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print("Pesos dos critérios (AHP-Gaussiano):")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates(subset=self.colunas_criterios + ["Score"]).reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print("\nRanking das Alternativas: - (AHP-Gaussiano)")
        print(df_resultados)
        return df_resultados

### LESTE EUROPEU

class LOPCOW(BaseMCDA_DADOS):

    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()
        for col in self.colunas_criterios:
            if self.tipo_criterio_list[col] == 'MAX':
                # Normalização para critérios de benefício (quanto maior melhor)
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                # Normalização para critérios de custo (quanto menor melhor)
                df_norm[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
        return df_norm

    def rank_alternativas(self):
        # Normalização da Matriz de Desempenho
        df = self.alternativas.copy()
        matriz_desempenho = df[self.colunas_criterios]
        matriz_desempenho = self.normalizar(matriz_desempenho)

        # Cálculo dos Valores Percentuais (PV) de cada critério
        RMS = np.sqrt(np.mean(np.square(matriz_desempenho), axis=0))
        DESVPAD = matriz_desempenho.std()
        PV = abs(np.log(RMS/DESVPAD))
        # Cálculo dos Pesos Objetivos
        pesos = PV/PV.sum()
        # Ordenação dos Critérios
        df["Score"] = (matriz_desempenho * pesos).sum(axis=1)
        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print("Pesos dos critérios LOPCOW:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates(subset=self.colunas_criterios + ["Score"]).reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print("\nRanking das Alternativas: - LOPCOW")
        print(df_resultados)
        return df_resultados
    
class MPSI(BaseMCDA_DADOS):
    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()
        for col in self.colunas_criterios:
            if self.tipo_criterio_list[col] == 'MAX':
                # Normalização para critérios de benefício (quanto maior melhor)
                df_norm[col] = (df[col]) / (df[col].max())
            else:
                # Normalização para critérios de custo (quanto menor melhor)
                df_norm[col] = (df[col].min()) / (df[col])
        return df_norm

    def rank_alternativas(self):
        # Normalização da Matriz de Desempenho
        df = self.alternativas.copy()
        matriz_desempenho = df[self.colunas_criterios]
        matriz_desempenho = self.normalizar(matriz_desempenho)

        # Cálculo MPSI
        valor_medio = matriz_desempenho.mean()
        VariacaoPreferencia = ((matriz_desempenho - valor_medio) ** 2).sum(axis=0)
        # Cálculo dos Pesos Objetivos
        pesos = VariacaoPreferencia / VariacaoPreferencia.sum()

        # Ordenação dos Critérios
        df["Score"] = (matriz_desempenho * pesos).sum(axis=1)
        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print("Pesos dos critérios MPSI:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates(subset=self.colunas_criterios + ["Score"]).reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print("\nRanking das Alternativas: - MPSI")
        print(df_resultados)
        return df_resultados
    

if __name__ == "__main__":
    # DADOS DE ENTRADA
    DATA = {
        "criterios": {
            "Custo": "MIN",
            "Camera": "MAX",
            "Armazenamento": "MAX",
            "Bateria": "MAX"
        },
        "alternativas": [
            {
                "id_alternativa": 1,
                "descricao": "SMP A",
                "Custo": 1500,
                "Camera": 12,
                "Armazenamento": 64,
                "Bateria": 24
            },
            {
                "id_alternativa": 2,
                "descricao": "SMP B",
                "Custo": 1800,
                "Camera": 12,
                "Armazenamento": 128,
                "Bateria": 18
            },
            {
                "id_alternativa": 3,
                "descricao": "SMP C",
                "Custo": 5000,
                "Camera": 20,
                "Armazenamento": 128,
                "Bateria": 10
            },
            {
                "id_alternativa": 4,
                "descricao": "SMP D",
                "Custo": 3200,
                "Camera": 20,
                "Armazenamento": 128,
                "Bateria": 16
            },
            {
                "id_alternativa": 5,
                "descricao": "SMP E",
                "Custo": 2150,
                "Camera": 14,
                "Armazenamento": 64,
                "Bateria": 12
            }
        ]
    }

    ahp_gauss = AHP_Gaussiano(arquivo_alternativas=DATA).rank_alternativas()
    lopcow = LOPCOW(arquivo_alternativas=DATA).rank_alternativas()
    mpsi = MPSI(arquivo_alternativas=DATA).rank_alternativas()


