import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

try:
    import fireducks.pandas as pd
except:
    pass


class BaseMCDA(ABC):
    """
    Classe base para todos os métodos MCDA:

    - Leitura de alternativas a partir de JSON
    - Estrutura para normalização
    - Colunas de critérios definidas
    - Método abstrato 'rank_alternativas'
    """

    def __init__(self, arquivo_alternativas=None):
        

        df_alternativas, criterios_list, tipo_criterio_list = self.leitura_alternativas(arquivo_alternativas)
        self.alternativas = df_alternativas
        self.colunas_criterios = criterios_list
        self.tipo_criterio_list = tipo_criterio_list
    
    def leitura_alternativas(self, arquivo_alternativas):
        # Processar JSON para extrair valores e tipos dos critérios
        alternativas = json.loads(arquivo_alternativas)
        criterios = [c for c in alternativas[0] if c not in ("id_alternativa", "descricao")]
        
        # Criar DataFrame com valores numéricos
        dados = []
        tipo_criterio = {}
        for alt in alternativas:
            row = {k: alt[k][0] if k in criterios else alt[k] for k in alt}
            dados.append(row)
            # Preencher dicionário de tipos (assume que todos os valores têm o mesmo tipo para cada critério)
            if not tipo_criterio:
                tipo_criterio = {c: alt[c][1] for c in criterios}
        
        df_alternativas = pd.DataFrame(dados)
        criterios_list  = criterios
        tipo_criterio_list =tipo_criterio
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

class WASPAS(BaseMCDA):
    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()
        for col in self.colunas_criterios:
            if self.tipo_criterio_list[col] == 'MAX':
                # Normalização para critérios de benefício (quanto maior melhor)
                df_norm[col] = (df[col] ) / (df[col].max())
            else:
                # Normalização para critérios de custo (quanto menor melhor)
                df_norm[col] = (df[col].min()) / (df[col])
        return df_norm

    def rank_alternativas(self, matriz_criterios, lambda_waspas=0.5):
        # Normalização da Matriz de Desempenho
        df = self.alternativas.copy()
        matriz_desempenho = df[self.colunas_criterios]
        matriz_desempenho = self.normalizar(matriz_desempenho)
        
        # pesos
        pesos = matriz_criterios.sum(axis=0) / matriz_criterios.sum().sum()

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
        print("Pesos dos critérios WASPAS:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates(subset=self.colunas_criterios + ["Score"]).reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print('\nRanking das Alternativas: - WASPAS ')
        print(df_resultados)
        return df_resultados

class LOPCOW(BaseMCDA):

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
    
class MPSI(BaseMCDA):
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
    
class WISP(BaseMCDA):

    def __init__(self, arquivo_alternativas=None):
        super().__init__(arquivo_alternativas)
    
    def normalizar(self, df):
        df_norm = df.copy()

        for col in self.colunas_criterios:
            df_norm[col] = (df[col] ) / (df[col].max())
        return df_norm

    def rank_alternativas(self, matriz_criterios, lambda_waspas=0.5):
        # Normalização da Matriz de Desempenho
        df = self.alternativas.copy()
        matriz_desempenho = df[self.colunas_criterios]
        matriz_desempenho = self.normalizar(matriz_desempenho)
        
        # pesos
        pesos = matriz_criterios.sum(axis=0) / matriz_criterios.sum().sum()
        pesos = np.round(pesos,3)
        
        MedidaUtilidade_wsd_min = 0
        MedidaUtilidade_wsd_max = 0

        MedidaUtilidade_wsp_min = 1
        MedidaUtilidade_wsp_max = 1

        for col in self.colunas_criterios:
            if self.tipo_criterio_list[col] == 'MAX':
                MedidaUtilidade_wsd_max += matriz_desempenho[col]*pesos[col]
                MedidaUtilidade_wsp_max *= matriz_desempenho[col]*pesos[col]

            if self.tipo_criterio_list[col] == 'MIN':
                MedidaUtilidade_wsd_min += matriz_desempenho[col]*pesos[col]
                MedidaUtilidade_wsp_min *= matriz_desempenho[col]*pesos[col]
               
        MedidaUtilidade_wsd = MedidaUtilidade_wsd_max - MedidaUtilidade_wsd_min
        MedidaUtilidade_wsp = MedidaUtilidade_wsp_max - MedidaUtilidade_wsp_min
        MedidaUtilidade_wsr = MedidaUtilidade_wsd_max/MedidaUtilidade_wsd_min
        MedidaUtilidade_wpr = MedidaUtilidade_wsp_max/MedidaUtilidade_wsp_min

        recalculoUtilidade_wsd = MedidaUtilidade_wsd/(1+MedidaUtilidade_wsd.max())
        recalculoUtilidade_wpd = MedidaUtilidade_wsp/(1+MedidaUtilidade_wsp.max())
        recalculoUtilidade_wsr = MedidaUtilidade_wsr/(1+MedidaUtilidade_wsr.max())
        recalculoUtilidade_wpr = MedidaUtilidade_wpr/(1+MedidaUtilidade_wpr.max())

        MedidaUtilidade_global = (1/4)*(recalculoUtilidade_wsd+
                                        recalculoUtilidade_wpd+
                                        recalculoUtilidade_wsr+
                                        recalculoUtilidade_wpr)    
         # Ordenação dos Critérios
        df["Score"] = MedidaUtilidade_global
        df_resultados = df.sort_values(by="Score", ascending=False)

        print('-'*80)
        print("Pesos dos critérios WISP:")
        print(pd.Series(np.round(pesos, 3), index=self.colunas_criterios))

        df_resultados = df_resultados.drop_duplicates(subset=self.colunas_criterios + ["Score"]).reset_index(drop=True)
        # Subtrai o valor mínimo para eliminar negativos e normaliza pela soma
        df_resultados["Score"] = df_resultados["Score"] - df_resultados["Score"].min()
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"], 3)
        print('\nRanking das Alternativas: - WISP')
        print(df_resultados)
        return df_resultados

    
if __name__ == '__main__':
    DATA = """
    [
        {
            "id_alternativa": 1,
            "descricao": "Samsung",
            "Custo": [3700, "MIN"],
            "Camera": [13, "MAX"],
            "Armazenamento": [64, "MAX"],
            "Bateria": [12, "MAX"]
        },
        {
            "id_alternativa": 2,
            "descricao": "Xiaomi",
            "Custo": [2500, "MIN"],
            "Camera": [12, "MAX"],
            "Armazenamento": [32, "MAX"],
            "Bateria": [20, "MAX"]
        },
        {
            "id_alternativa": 3,
            "descricao": "iPhone",
            "Custo": [4200, "MIN"],
            "Camera": [13, "MAX"],
            "Armazenamento": [32, "MAX"],
            "Bateria": [7, "MAX"]
        }
    ]
    """
    
    # Matriz de critérios para WASPAS (exemplo simplificado)
    matriz_criterios = pd.DataFrame({
        'Custo':         [1, 3, 2, 2],
        'Camera':        [1/3, 1, 1/2, 1],
        'Armazenamento': [1/2, 2, 1, 1],
        'Bateria':       [1/2, 1, 1, 1]
    })
    

    
    # Instanciar e executar métodos MCDA
    waspas = WASPAS(arquivo_alternativas= DATA).rank_alternativas(matriz_criterios=matriz_criterios)
    lopcow = LOPCOW(arquivo_alternativas=DATA).rank_alternativas()
    mpsi = MPSI(arquivo_alternativas=DATA).rank_alternativas()
    wisp = WISP(arquivo_alternativas=DATA).rank_alternativas(matriz_criterios=matriz_criterios)
