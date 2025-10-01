import json
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from fractions import Fraction
from abc import ABC, abstractmethod

try:
    import fireducks.pandas as pd
except:
    pass


class BaseMCDA_SUBJETIVO(ABC):
    """
    Classe base para todos os métodos MCDA:

    - Leitura de alternativas a partir de JSON
    - Estrutura para normalização
    - Colunas de critérios definidas
    - Método abstrato 'rank_alternativas'
    """

    def __init__(self, arquivo_alternativas=None, matriz_preferencias_decisor=None):
        """       
        Args:
            arquivo_alternativas: Dicionário ou JSON com dados
            matriz_preferencias_decisor: Matriz de comparação entre critérios
        """
        # Carrega dados
        if isinstance(arquivo_alternativas, dict):
            data = arquivo_alternativas
        else:
            data = json.loads(arquivo_alternativas)
        
        # Processa dados das alternativas
        df_alternativas, criterios_list, tipo_criterio_list, avaliacoes_subjetivas = self.leitura_alternativas(data)
        self.alternativas = df_alternativas
        self.colunas_criterios = criterios_list
        self.tipo_criterio_list = tipo_criterio_list
        self.avaliacoes_subjetivas = avaliacoes_subjetivas
        self.matriz_preferencias_decisor = matriz_preferencias_decisor.T
        
        # Identifica critérios subjetivos automaticamente
        self.criterios_subjetivos = [c for c, t in tipo_criterio_list.items() if t == "QUALITATIVO"]
        self.matrizes_subjetivas = {}

    def leitura_alternativas(self, data):
        """Lê dados das alternativas e critérios"""
        criterios_dict = data["criterios"]
        criterios_list = list(criterios_dict.keys())
        tipo_criterio_list = criterios_dict
        
        alternativas_data = data["alternativas"]
        df_alternativas = pd.DataFrame(alternativas_data)
        
        # Extrai avaliações subjetivas
        avaliacoes_subjetivas = {}
        for key, value in data.items():
            if key.startswith("avaliacao_"):
                criterio = key.replace("avaliacao_", "")
                avaliacoes_subjetivas[criterio] = value
        
        return df_alternativas, criterios_list, tipo_criterio_list, avaliacoes_subjetivas

    def construir_matrizes_subjetivas(self):
        """Constrói matrizes de comparação para critérios subjetivos"""
        for criterio in self.criterios_subjetivos:
            if criterio in self.avaliacoes_subjetivas:
                avaliacoes = self.avaliacoes_subjetivas[criterio]
                alternativas_ids = self.alternativas["id_alternativa"].tolist()
                n = len(alternativas_ids)
                
                matriz = np.ones((n, n))
                
                for par, valor in avaliacoes.items():
                    a, b = map(int, par.split("-"))
                    i = alternativas_ids.index(a)
                    j = alternativas_ids.index(b)
                    matriz[i, j] = valor
                    matriz[j, i] = 1 / valor
                
                self.matrizes_subjetivas[criterio] = matriz

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

    def calcula_pesos(self, matriz):
        if isinstance(matriz, pd.DataFrame):
            arr = matriz.values.astype(float)
        else:
            arr = np.array(matriz, dtype=float)

        n = arr.shape[0]
    
        pesos = matriz / matriz.sum()
        vetor_prioridade = pesos.sum(axis=1) / n

        return vetor_prioridade
    
    @abstractmethod
    def rank_alternativas(self):
        pass


### ESCOLA AMERICANA

class AHP(BaseMCDA_SUBJETIVO):
    
    def __init__(self, arquivo_alternativas=None, matriz_preferencias_decisor=None):
        """
        Inicializa o AHP com dados das alternativas e matriz de preferências
        
        Args:
            arquivo_alternativas: Dicionário ou JSON com dados
            matriz_preferencias_decisor: Matriz de comparação entre critérios
        """
        super().__init__(arquivo_alternativas,matriz_preferencias_decisor)

    def normalizar(self, df):
        """Normaliza critérios objetivos"""
        df_norm = df.copy()
        col = df_norm.name
        if col in self.tipo_criterio_list and self.tipo_criterio_list[col] in ['MAX', 'MIN']:
            if self.tipo_criterio_list[col] == 'MAX':
                df_norm = df / df.sum()  # Quanto maior melhor
            else:
                df_norm = 1 / df  # Quanto menor melhor
        return df_norm
 
    def calcula_pesos_ahp(self, matriz):
        """Calcula pesos usando método AHP de Saaty"""
        if isinstance(matriz, pd.DataFrame):
            arr = matriz.values.astype(float)
        else:
            arr = np.array(matriz, dtype=float)

        n = arr.shape[0]
    
        pesos = matriz / matriz.sum()
        vetor_prioridade = pesos.sum(axis=1) / n

        # Verificação de consistência
        matriz_de_consistencia = matriz * vetor_prioridade

        pesos_consistencia = matriz_de_consistencia.sum(axis=1)
        vetor_lambdas = pesos_consistencia / vetor_prioridade

        lambda_max = float(vetor_lambdas.sum())/n

        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0        
        RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
        ri = RI.get(n, 0)
        cr = ci / ri if ri > 0 else float('inf')
        
        # if cr > 0.1:
        #     print(f"AVISO: CR = {cr:.3f} > 0.10 - Revisar comparações!")
        # else:
        #     print(f"CR = {cr:.3f} (ok)")
        
        return vetor_prioridade

    def rank_alternativas(self):
        """Calcula ranking final das alternativas"""
        
        # 1. Construir matrizes para critérios subjetivos
        self.construir_matrizes_subjetivas()
        
        # 2. Calcular pesos dos critérios objetivos
        pesos = self.calcula_pesos_ahp(self.matriz_preferencias_decisor)
        
        # 3. Construir matriz de desempenho
        df = self.alternativas.copy()
        matriz_desempenho = pd.DataFrame(index=df.index)
        
        # Critérios objetivos
        criterios_objetivos = [c for c in self.colunas_criterios if c not in self.criterios_subjetivos]
        for criterio in criterios_objetivos:
            desempenho_objetivo = self.normalizar(df[criterio])*pesos[criterio]
            desempenho_objetivo = desempenho_objetivo / desempenho_objetivo.sum()
            matriz_desempenho = pd.concat([matriz_desempenho, desempenho_objetivo], axis=1)
        
        # Critérios subjetivos
        for criterio in self.criterios_subjetivos:
            if criterio in self.matrizes_subjetivas:
                df_criterio_subjetivo = pd.DataFrame(self.matrizes_subjetivas[criterio])
                desempenho_subjetivo = self.calcula_pesos_ahp(df_criterio_subjetivo)
                matriz_desempenho[criterio] = desempenho_subjetivo
        
        for c in self.colunas_criterios:
            if c not in matriz_desempenho.columns:
                matriz_desempenho[c] = 0.0
        
        # 4. Calcular scores finais
        df["Score"] = matriz_desempenho.dot(pesos)
        df["Score"] = df["Score"].round(3)
        
        df_resultados = df.sort_values("Score", ascending=False)


        print('-'*80)
        print("Pesos dos critérios AHP:")
        print(pd.Series(np.round(pesos, 3), index=pesos.index))

        df_resultados = df_resultados.drop_duplicates().reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print('\nRanking das Alternativas: - AHP ')
        print(df_resultados)
        
        return df_resultados

### LESTE EUROPEU 

class WASPAS(BaseMCDA_SUBJETIVO):
    def __init__(self, arquivo_alternativas=None, matriz_preferencias_decisor=None):
        super().__init__(arquivo_alternativas,matriz_preferencias_decisor)
    
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

    def rank_alternativas(self, lambda_waspas=0.5):
        
        # 1. Construir matrizes para critérios subjetivos
        self.construir_matrizes_subjetivas()
        
        # 2. Calcular pesos dos critérios objetivos
        pesos = self.calcula_pesos(self.matriz_preferencias_decisor)
  
        # 3. Construir matriz de desempenho
        df = self.alternativas.copy()
        matriz_desempenho = pd.DataFrame(index=df.index)
        
        # Critérios objetivos
        criterios_objetivos = [c for c in self.colunas_criterios if c not in self.criterios_subjetivos]
        for criterio in criterios_objetivos:
            desempenho_objetivo = self.normalizar(df[criterio])*pesos[criterio]
            desempenho_objetivo = desempenho_objetivo / desempenho_objetivo.sum()
            matriz_desempenho = pd.concat([matriz_desempenho, desempenho_objetivo], axis=1)

        # Critérios subjetivos
        for criterio in self.criterios_subjetivos:
            if criterio in self.matrizes_subjetivas:
                df_criterio_subjetivo = pd.DataFrame(self.matrizes_subjetivas[criterio])
                desempenho_subjetivo = self.calcula_pesos(df_criterio_subjetivo)
                matriz_desempenho[criterio] = desempenho_subjetivo
        
        for c in self.colunas_criterios:
            if c not in matriz_desempenho.columns:
                matriz_desempenho[c] = 0.0


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

        df_resultados = df_resultados.drop_duplicates().reset_index(drop=True)
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"],3)
        print('\nRanking das Alternativas: - WASPAS ')
        print(df_resultados)
        return df_resultados

class WISP(BaseMCDA_SUBJETIVO):

    def __init__(self, arquivo_alternativas=None, matriz_preferencias_decisor=None):
        super().__init__(arquivo_alternativas,matriz_preferencias_decisor)
    
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

    def rank_alternativas(self):
         # 1. Construir matrizes para critérios subjetivos
        self.construir_matrizes_subjetivas()
        
        # 2. Calcular pesos dos critérios objetivos
        pesos = self.calcula_pesos(self.matriz_preferencias_decisor)
  
        # 3. Construir matriz de desempenho
        df = self.alternativas.copy()
        matriz_desempenho = pd.DataFrame(index=df.index)
        
        # Critérios objetivos
        criterios_objetivos = [c for c in self.colunas_criterios if c not in self.criterios_subjetivos]
        for criterio in criterios_objetivos:
            desempenho_objetivo = self.normalizar(df[criterio])*pesos[criterio]
            desempenho_objetivo = desempenho_objetivo / desempenho_objetivo.sum()
            matriz_desempenho = pd.concat([matriz_desempenho, desempenho_objetivo], axis=1)

        # Critérios subjetivos
        for criterio in self.criterios_subjetivos:
            if criterio in self.matrizes_subjetivas:
                df_criterio_subjetivo = pd.DataFrame(self.matrizes_subjetivas[criterio])
                desempenho_subjetivo = self.calcula_pesos(df_criterio_subjetivo)
                matriz_desempenho[criterio] = desempenho_subjetivo
        
        for c in self.colunas_criterios:
            if c not in matriz_desempenho.columns:
                matriz_desempenho[c] = 0.0
        
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

        df_resultados = df_resultados.drop_duplicates().reset_index(drop=True)
        # Subtrai o valor mínimo para eliminar negativos e normaliza pela soma
        df_resultados["Score"] = df_resultados["Score"] - df_resultados["Score"].min()
        df_resultados["Score"] = df_resultados["Score"] / df_resultados["Score"].sum()
        df_resultados["Score"] = np.round(df_resultados["Score"], 3)
        print('\nRanking das Alternativas: - WISP')
        print(df_resultados)
        return df_resultados

if __name__ == '__main__':
    # DADOS DE ENTRADA
    DATA = {
        "criterios": {
            "Custo": "MIN",
            "Armazenamento": "MAX",
            "Camera": "QUALITATIVO", 
            "Design": "QUALITATIVO"
        },
        "alternativas": [
            {
                "id_alternativa": 1, 
                "descricao": "Xiaomi",
                "Custo": 1200,
                "Armazenamento": 64,
            },
            {
                "id_alternativa": 2,
                "descricao": "Samsung",
                "Custo": 1500,
                "Armazenamento": 64,
            },
            {
                "id_alternativa": 3,
                "descricao": "iPhone", 
                "Custo": 3000,
                "Armazenamento": 128,
            }
        ],
        "avaliacao_Design": {
            "1-2": 1,    
            "1-3": 1/7,  
            "2-3": 1/5   
        },
        "avaliacao_Camera": {
            "1-2": 3,    
            "1-3": 1,  
            "2-3": 1/3   
        },
    }

    # Matriz de preferências do decisor
    matriz_preferencias_decisor = pd.DataFrame({
        'Custo': [1, 1, 3, 5],
        'Camera': [1, 1, 3, 7],
        'Armazenamento': [1/3, 1/3, 1, 3],
        'Design': [1/5, 1/7, 1/3, 1]
    }, index=['Custo', 'Camera', 'Armazenamento','Design'])


    ahp = AHP(arquivo_alternativas=DATA, matriz_preferencias_decisor=matriz_preferencias_decisor).rank_alternativas()
    waspas = WASPAS(arquivo_alternativas=DATA, matriz_preferencias_decisor=matriz_preferencias_decisor).rank_alternativas()
    #wisp = WISP(arquivo_alternativas=DATA, matriz_preferencias_decisor=matriz_preferencias_decisor).rank_alternativas()