import numpy as np
import pandas as pd

class EvaluateFactors:
    """
    Gera fatores de carga e vento para simulação multi-período.
    
    Para carga: cada barra tem seu próprio fator, variando aleatoriamente dentro de um intervalo
    definido por um perfil horário base e uma incerteza global.
    
    Para vento: utiliza um arquivo CSV com histórico de fatores de vento (normalizados).
    Para cada período, sorteia um fator base do histórico e aplica uma pequena variação
    individual para cada gerador eólico.
    """
    
    def __init__(self,
                 sistema,
                 n_dias,
                 n_horas,
                 hora_interesse=None,
                 carga_incerteza=0.2,
                 vento_variacao=0.1,
                 seed=None):
        """
        Args:
            sistema: objeto com atributos NBAR, NGER_GWD, etc.
            n_dias: número de dias
            n_horas: horas por dia (máximo 24)
            carga_incerteza: amplitude da variação uniforme em torno do perfil (ex: 0.2 = ±10%)
            vento_variacao: amplitude da variação relativa para cada gerador (ex: 0.1 = ±5%)
            seed: semente para reprodutibilidade
        """
        self.sistema = sistema
        self.n_dias = n_dias
        self.n_horas = n_horas
        self.T = n_dias * n_horas
        
        if (n_dias == 1
            and n_horas ==1):
                    self.carga_perfil_horario = np.array([0.94])
        else:
            # Perfil horário normalizado (pico = 1.0)
            self.carga_perfil_horario = np.array([
                0.65, 0.69, 0.60, 0.61, 0.62, 0.70,  # 0-5h (madrugada)
                0.75, 0.71, 0.80, 0.85, 0.88, 0.90,  # 6-11h (manhã)
                0.92, 0.91, 0.87, 0.93, 0.94, 0.96,  # 12-17h (tarde)
                1.00, 0.98, 0.95, 0.97, 0.74, 0.55   # 18-23h (noite/pico)
            ])


        
        self.carga_incerteza = carga_incerteza
        
        vento_arquivo = r"C:\Users\LucasBraga\Documents\repos\ufjf\gopt-BessWindAgentOperator\SRC\DB\getters\intermittent-renewables-production-france.csv"
        #vento_arquivo = r"/home/lucasedbraga/repositorios/ufjf/gopt-BessWindAgentOperator/SRC/DB/getters/intermittent-renewables-production-france.csv"
        self.vento_arquivo = vento_arquivo
        self.vento_variacao = vento_variacao
        self.seed = seed
        
        # Carregar dados de vento
        self._load_vento_data()
    
    def _load_vento_data(self):
        """Carrega fatores de vento do CSV e organiza por hora do dia."""
        import os
        if not os.path.exists(self.vento_arquivo):
            raise FileNotFoundError(f"Arquivo de vento não encontrado: {self.vento_arquivo}")

        df = pd.read_csv(self.vento_arquivo)

        # Verifica colunas necessárias
        required_cols = ['Date and Hour', 'Source', 'Production', 'StartHour']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no arquivo de vento.")

        # Converte 'Date and Hour' para datetime
        df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6), errors='coerce')
        df = df.sort_values('DateTime')
        df_wind = df[df['Source'] == 'Wind'].copy()

        if df_wind.empty:
            raise ValueError("Nenhum dado de vento encontrado no arquivo.")

        # Garante que Production seja numérico (converte strings inválidas para NaN)
        df_wind['Production'] = pd.to_numeric(df_wind['Production'], errors='coerce')
        # Substitui NaN por 0 (ajuste conforme necessidade) – sem inplace
        df_wind['Production'] = df_wind['Production'].fillna(0)

        # Normaliza os fatores de vento (0–1)
        max_prod = df_wind['Production'].max()
        if max_prod > 0:
            df_wind['Factor'] = df_wind['Production'] / max_prod
        else:
            df_wind['Factor'] = 0.0

        # Extrai a hora do início (StartHour) de forma robusta
        def extract_hour(s):
            try:
                # Tenta converter com formato padrão HH:MM:SS
                return pd.to_datetime(s, format='%H:%M:%S', errors='coerce').hour
            except:
                # Fallback: pega os dois primeiros caracteres como hora
                try:
                    return int(str(s)[:2])
                except:
                    return -1  # inválido

        df_wind['Hour'] = df_wind['StartHour'].apply(extract_hour)
        # Remove linhas com hora inválida
        df_wind = df_wind[df_wind['Hour'] >= 0]

        if df_wind.empty:
            raise ValueError("Nenhuma hora válida extraída dos dados de vento.")

        # Cria lista de listas para cada hora (0-23)
        self.vento_fatores_por_hora = [[] for _ in range(24)]
        # Usa itertuples para maior eficiência e segurança
        for row in df_wind.itertuples(index=False):
            hora = row.Hour
            if 0 <= hora < 24:
                self.vento_fatores_por_hora[hora].append(row.Factor)

        # Verifica se todas as horas têm pelo menos um dado
        for h, lista in enumerate(self.vento_fatores_por_hora):
            if len(lista) == 0:
                raise ValueError(f"Nenhum dado de vento para a hora {h}:00")
    
    def gerar_fatores_carga(self):
        """
        Retorna array de fatores de carga com shape (n_dias, n_horas, NBAR).
        Cada barra tem seu próprio fator, sorteado uniformemente entre
        lim_inf[h] e lim_sup[h] para cada hora h.
        """
        np.random.seed(self.seed)
        n_dias = self.n_dias
        n_horas = self.n_horas
        NBAR = self.sistema.NBAR
        
        # Seleciona apenas as primeiras n_horas do perfil
        perfil_horas = self.carga_perfil_horario[:n_horas]
        
        lim_inf = perfil_horas - self.carga_incerteza/2
        lim_sup = perfil_horas + self.carga_incerteza/2
        
        # Garantir que limites não fiquem negativos
        lim_inf = np.maximum(lim_inf, 0.0)
        
        # Shape final: (dias, horas, barras)
        fatores = np.zeros((n_dias, n_horas, NBAR))
        
        for d in range(n_dias):
            for h in range(n_horas):
                # Sorteia para cada barra independentemente
                fatores[d, h, :] = np.random.uniform(lim_inf[h], lim_sup[h], size=NBAR)
        
        return fatores
    
    def gerar_fatores_vento(self, use_weibull=False, weibull_shape=None, weibull_scale=None,
                        cut_in=3.0, rated_speed=12.0, cut_out=25.0, power_curve=None):
        """
        Retorna array de fatores de vento com shape (n_dias, n_horas, NGER_GWD).
        Se use_weibull=True, gera velocidades Weibull e aplica curva de potência.
        Caso contrário, usa o método original (sorteio dos históricos por hora).

        Parâmetros:
        - use_weibull: bool, se True usa Weibull + curva de potência, senão usa histórico.
        - weibull_shape: float ou array (n_horas, NGER_GWD) - parâmetro de forma (k)
        - weibull_scale: float ou array (n_horas, NGER_GWD) - parâmetro de escala (λ) em m/s
        - cut_in: float ou array, velocidade de partida (m/s) - padrão 3.0
        - rated_speed: float ou array, velocidade nominal (m/s) - padrão 12.0
        - cut_out: float ou array, velocidade de corte (m/s) - padrão 25.0
        - power_curve: callable opcional, função que recebe velocidade (array) e retorna fator [0,1].
                    Se fornecida, substitui a curva linear/cúbica padrão.
        """
        if not use_weibull:
            # --- Código original (histórico) ---
            if not hasattr(self, 'vento_fatores_por_hora'):
                raise ValueError("Dados de vento não carregados corretamente.")
            print("USANDO DADOS HISTORICOS PARA VENTO")

            np.random.seed(self.seed)
            n_dias = self.n_dias
            n_horas = self.n_horas
            T = n_dias * n_horas
            NGER_GWD = self.sistema.NGER_GWD

            fatores_base = np.zeros((n_dias, n_horas))
            for h in range(n_horas):
                lista = self.vento_fatores_por_hora[h]
                if len(lista) == 0:
                    raise ValueError(f"Não há dados históricos para a hora {h}.")
                indices = np.random.choice(len(lista), size=n_dias, replace=True)
                fatores_base[:, h] = np.array(lista)[indices]

            fatores_base = fatores_base.ravel()
            fatores_base = np.tile(fatores_base.reshape(-1, 1), (1, NGER_GWD))

            delta = np.random.uniform(-self.vento_variacao/2, self.vento_variacao/2,
                                    size=(T, NGER_GWD))
            fatores = fatores_base * (1 + delta)
            fatores = np.maximum(fatores, 0.0)
            fatores = np.minimum(fatores, 1.0)   # garantir limite superior

            fatores = fatores.reshape(n_dias, n_horas, NGER_GWD)
            return fatores
        
        print("USANDO WEIBULL PARA VENTO")

        # --- Weibull + curva de potência ---
        np.random.seed(self.seed)
        n_dias = self.n_dias
        n_horas = self.n_horas
        NGER_GWD = self.sistema.NGER_GWD
        T = n_dias * n_horas

        # Definir parâmetros Weibull
        if weibull_shape is None:
            weibull_shape = 2.0      # valor típico
        if weibull_scale is None:
            weibull_scale = 10.0     # m/s, valor típico

        # Expandir para (n_horas, NGER_GWD) se escalar
        if np.isscalar(weibull_shape):
            weibull_shape = np.full((n_horas, NGER_GWD), weibull_shape)
        if np.isscalar(weibull_scale):
            weibull_scale = np.full((n_horas, NGER_GWD), weibull_scale)

        weibull_shape = np.asarray(weibull_shape)
        weibull_scale = np.asarray(weibull_scale)
        if weibull_shape.shape != (n_horas, NGER_GWD):
            raise ValueError(f"weibull_shape deve ter formato ({n_horas}, {NGER_GWD})")
        if weibull_scale.shape != (n_horas, NGER_GWD):
            raise ValueError(f"weibull_scale deve ter formato ({n_horas}, {NGER_GWD})")

        # Gerar velocidades do vento (m/s) via transformada inversa da Weibull
        u = np.random.uniform(0, 1, size=(n_dias, n_horas, NGER_GWD))
        shape_exp = np.tile(weibull_shape.reshape(1, n_horas, NGER_GWD), (n_dias, 1, 1))
        scale_exp = np.tile(weibull_scale.reshape(1, n_horas, NGER_GWD), (n_dias, 1, 1))
        wind_speed = scale_exp * (-np.log(1 - u)) ** (1 / shape_exp)   # velocidades em m/s

        # Converter velocidades em fatores de capacidade (entre 0 e 1)
        if power_curve is not None:
            # Usar curva fornecida pelo usuário
            fatores = power_curve(wind_speed)
        else:
            # Curva padrão: linear entre cut_in e rated_speed
            # Expandir parâmetros da curva se fornecidos como escalares
            if np.isscalar(cut_in):
                cut_in = np.full((n_horas, NGER_GWD), cut_in)
            if np.isscalar(rated_speed):
                rated_speed = np.full((n_horas, NGER_GWD), rated_speed)
            if np.isscalar(cut_out):
                cut_out = np.full((n_horas, NGER_GWD), cut_out)

            cut_in_exp = np.tile(cut_in.reshape(1, n_horas, NGER_GWD), (n_dias, 1, 1))
            rated_exp = np.tile(rated_speed.reshape(1, n_horas, NGER_GWD), (n_dias, 1, 1))
            cut_out_exp = np.tile(cut_out.reshape(1, n_horas, NGER_GWD), (n_dias, 1, 1))

            fatores = np.zeros_like(wind_speed)
            # Região 1: abaixo de cut_in -> 0
            mask = (wind_speed >= cut_in_exp) & (wind_speed < rated_exp)
            # Região 2: entre cut_in e rated_speed -> linear
            fatores[mask] = (wind_speed[mask] - cut_in_exp[mask]) / (rated_exp[mask] - cut_in_exp[mask])
            # Região 3: entre rated_speed e cut_out -> 1
            mask2 = (wind_speed >= rated_exp) & (wind_speed < cut_out_exp)
            fatores[mask2] = 1.0
            # Região 4: acima de cut_out -> 0 (já está zero)

        # Aplicar variação adicional (opcional)
        if self.vento_variacao > 0:
            delta = np.random.uniform(-self.vento_variacao/2, self.vento_variacao/2,
                                    size=(n_dias, n_horas, NGER_GWD))
            fatores = fatores * (1 + delta)
        
        # Garantir limites finais [0,1]
        fatores = np.clip(fatores, 0.0, 1.0)
        return fatores
        
    def gerar_tudo(self):
        """
        Retorna tuple (fatores_carga, fatores_vento) com shapes:
        carga: (n_dias, n_horas, NBAR)
        vento: (n_dias, n_horas, NGER_GWD)
        """
        fatores_carga = self.gerar_fatores_carga()
        fatores_vento = self.gerar_fatores_vento(use_weibull=False)  # use weibull por padrão
        return fatores_carga, fatores_vento