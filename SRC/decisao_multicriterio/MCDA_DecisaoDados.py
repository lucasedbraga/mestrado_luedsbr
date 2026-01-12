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
        
        # Identifica critérios objetivos 
        self.criterios_objetivos = [c for c, t in tipo_criterio_list.items() if t != "QUALITATIVO"]



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


### ESCOLA BRASILEIRA

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


class AHP_Gaussiano_WASPAS(BaseMCDA_DADOS):
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
    
    def rank_alternativas(self,lambda_waspas=0.5):
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

        # WSM
        matriz_desempenho["WSM"] = sum(matriz_desempenho[col] * pesos[col] for col in self.colunas_criterios)
        # WPM
        matriz_desempenho["WPM"] = 1
        for col in self.colunas_criterios:
            matriz_desempenho["WPM"] *= matriz_desempenho[col] ** pesos[col]

        # Score final
        df["Score"] = (
            lambda_waspas * matriz_desempenho["WSM"] +
            (1 - lambda_waspas) * matriz_desempenho["WPM"]
        )

        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print(f"Pesos dos critérios AHP-Gaussiano+WASPAS com lambda = {lambda_waspas}:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates().reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print(f'\nRanking das Alternativas: - AHP-Gaussiano+WASPAS')
        print(df_resultados)
        return df_resultados

    
### LESTE EUROPEU

class LOPCOW_WASPAS(BaseMCDA_DADOS):

    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()
        col = df_norm.name
        if col in self.tipo_criterio_list and self.tipo_criterio_list[col] in ['MAX', 'MIN']:
            if self.tipo_criterio_list[col] == 'MAX':
                # Normalização para critérios de benefício (quanto maior melhor)
                df_norm = (df - df.min()) / (df.max() - df.min())
            else:
                # Normalização para critérios de custo (quanto menor melhor)
                df_norm = (df.max() - df) / (df.max() - df.min())
        return df_norm

    def rank_alternativas(self,lambda_waspas=0.5):

        df = self.alternativas.copy()
        # Critérios objetivos
        criterios_objetivos = [c for c in self.criterios_objetivos]
        matriz_desempenho = pd.DataFrame()
        for criterio in criterios_objetivos:
            desempenho_objetivo = self.normalizar(df[criterio])
            desempenho_objetivo = desempenho_objetivo / desempenho_objetivo.sum()
            matriz_desempenho = pd.concat([matriz_desempenho, desempenho_objetivo], axis=1)

        for c in self.colunas_criterios:
                if c not in matriz_desempenho.columns:
                    matriz_desempenho[c] = 0.0

        print(matriz_desempenho)
        print("LUCAS"*10)
        # Cálculo dos Valores Percentuais (PV) de cada critério
        RMS = np.sqrt(np.mean(np.square(matriz_desempenho), axis=0))
        DESVPAD = matriz_desempenho.std()
        PV = abs(np.log(RMS/DESVPAD))
        # Cálculo dos Pesos Objetivos
        pesos = PV/PV.sum()
        
        # WSM
        matriz_desempenho["WSM"] = sum(matriz_desempenho[col] * pesos[col] for col in self.colunas_criterios)
        # WPM
        matriz_desempenho["WPM"] = 1
        for col in self.colunas_criterios:
            matriz_desempenho["WPM"] *= matriz_desempenho[col] ** pesos[col]

        # Score final
        df["Score"] = (
            lambda_waspas * matriz_desempenho["WSM"] +
            (1 - lambda_waspas) * matriz_desempenho["WPM"]
        )
        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print(f"Pesos dos critérios LOPCOW+WASPAS com lambda = {lambda_waspas}:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates().reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print(f'\nRanking das Alternativas: - LOPCOW+WASPAS')
        print(df_resultados)
        return df_resultados

    
class MPSI_WASPAS(BaseMCDA_DADOS):
    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()
        col = df_norm.name
        if col in self.tipo_criterio_list and self.tipo_criterio_list[col] in ['MAX', 'MIN']:
            if self.tipo_criterio_list[col] == 'MAX':
                # Normalização para critérios de benefício (quanto maior melhor)
                df_norm = (df) / (df.max())
            else:
                # Normalização para critérios de custo (quanto menor melhor)
                df_norm = (df.min()) / (df)
        return df_norm

    def rank_alternativas(self,lambda_waspas=0.5):
        df = self.alternativas.copy()
        # Critérios objetivos
        criterios_objetivos = [c for c in self.criterios_objetivos]
        matriz_desempenho = pd.DataFrame()
        for criterio in criterios_objetivos:
            desempenho_objetivo = self.normalizar(df[criterio])
            desempenho_objetivo = desempenho_objetivo / desempenho_objetivo.sum()
            matriz_desempenho = pd.concat([matriz_desempenho, desempenho_objetivo], axis=1)

        for c in self.colunas_criterios:
                if c not in matriz_desempenho.columns:
                    matriz_desempenho[c] = 0.0

        # Cálculo MPSI
        valor_medio = matriz_desempenho.mean()
        VariacaoPreferencia = ((matriz_desempenho - valor_medio) ** 2).sum(axis=0)
        # Cálculo dos Pesos Objetivos
        pesos = VariacaoPreferencia / VariacaoPreferencia.sum()
        
        # WSM
        matriz_desempenho["WSM"] = sum(matriz_desempenho[col] * pesos[col] for col in self.colunas_criterios)
        # WPM
        matriz_desempenho["WPM"] = 1
        for col in self.colunas_criterios:
            matriz_desempenho["WPM"] *= matriz_desempenho[col] ** pesos[col]

        # Score final
        df["Score"] = (
            lambda_waspas * matriz_desempenho["WSM"] +
            (1 - lambda_waspas) * matriz_desempenho["WPM"]
        )
        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print(f"Pesos dos critérios MPSI+WASPAS com lambda = {lambda_waspas}:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates().reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print('\nRanking das Alternativas: - MPSI+WASPAS ')
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
    ahp_gauss_waspas = AHP_Gaussiano_WASPAS(arquivo_alternativas=DATA).rank_alternativas()
    lopcow = LOPCOW_WASPAS(arquivo_alternativas=DATA).rank_alternativas()
    mpsi = MPSI_WASPAS(arquivo_alternativas=DATA).rank_alternativas()


