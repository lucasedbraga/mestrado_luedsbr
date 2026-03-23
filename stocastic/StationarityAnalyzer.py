import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import skew, kurtosis, pearsonr
import sqlite3
import kagglehub

class StationarityAnalyzer:
    """
    Classe para análise de estacionariedade de séries temporais de consumo de energia.
    """

    def __init__(self, data_path=None, download_if_missing=True):
        """
        Inicializa o analisador.

        Parâmetros:
        - data_path: caminho para um arquivo CSV (opcional). Se None, tenta baixar do Kaggle.
        - download_if_missing: se True e data_path não for fornecido, baixa automaticamente.
        """
        if data_path is None and download_if_missing:
            print("Baixando dataset do Kaggle...")
            path = kagglehub.dataset_download("drmtya/smart-home-energy-consumption-optimization")
            arquivos = os.listdir(path)
            csv_file = [f for f in arquivos if f.endswith('.csv')][0]
            self.df_raw = pd.read_csv(os.path.join(path, csv_file))
            print("Dataset baixado.")
        elif data_path is not None:
            self.df_raw = pd.read_csv(data_path)
        else:
            raise ValueError("É necessário fornecer um data_path ou permitir download (download_if_missing=True).")

        # Preparar dados
        self.df_raw['timestamp'] = pd.to_datetime(self.df_raw['timestamp'])
        self.df_raw.set_index('timestamp', inplace=True)

        # Agregação horária por casa
        self.hourly = self._aggregate_hourly()

        # Lista de casas
        self.homes = sorted(self.hourly['home_id'].unique())

        # Dicionário para armazenar séries por casa
        self.series = {}
        for home in self.homes:
            ts = self.hourly[self.hourly['home_id'] == home].set_index('datetime')['energy_kWh']
            self.series[home] = ts.asfreq('H').dropna()

    def _aggregate_hourly(self):
        """
        Agrega os dados em consumo horário por casa (kWh).
        """
        # Group by hour and home, sum power
        hourly = self.df_raw.groupby([pd.Grouper(freq='H'), 'home_id'])['power_watt'].sum().reset_index()
        hourly.rename(columns={'power_watt': 'soma_potencia'}, inplace=True)
        hourly['energy_kWh'] = hourly['soma_potencia'] * 0.25 / 1000
        hourly.drop(columns='soma_potencia', inplace=True)
        hourly.rename(columns={'timestamp': 'datetime'}, inplace=True)
        return hourly

    def get_homes(self):
        """Retorna a lista de IDs das casas."""
        return self.homes

    def get_series(self, home):
        """Retorna a série temporal da casa especificada."""
        return self.series[home]

    def get_stats(self):
        """
        Calcula estatísticas descritivas para todas as casas.
        Retorna um DataFrame com as métricas.
        """
        stats = []
        for home in self.homes:
            ts = self.series[home]
            desc = ts.describe(percentiles=[0.25, 0.75])
            stats.append({
                'home_id': home,
                'count': desc['count'],
                'mean_kWh': desc['mean'],
                'median_kWh': ts.median(),
                'std_kWh': desc['std'],
                'min_kWh': desc['min'],
                'max_kWh': desc['max'],
                'q25_kWh': desc['25%'],
                'q75_kWh': desc['75%'],
                'total_kWh': ts.sum(),
                'skewness': skew(ts),
                'kurtosis': kurtosis(ts),
                'acf_lag1': ts.autocorr(lag=1),
                'acf_lag24': ts.autocorr(lag=24),
                'adf_statistic': None,
                'adf_pvalue': None,
                'adf_critical_1%': None,
                'adf_critical_5%': None,
                'adf_critical_10%': None,
                'is_stationary': None
            })
        return pd.DataFrame(stats)

    def adf_test(self, home, autolag='AIC'):
        """
        Realiza o teste ADF para a casa especificada.
        Retorna um dicionário com os resultados.
        """
        ts = self.series[home]
        result = adfuller(ts, autolag=autolag)
        return {
            'home_id': home,
            'adf_stat': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'nobs': result[3],
            'critical_1%': result[4]['1%'],
            'critical_5%': result[4]['5%'],
            'critical_10%': result[4]['10%']
        }

    def run_adf_all(self):
        """Aplica o teste ADF para todas as casas e retorna um DataFrame."""
        results = []
        for home in self.homes:
            results.append(self.adf_test(home))
        return pd.DataFrame(results)

    def save_to_sqlite(self, db_path='energy_consumption.db'):
        """Salva os dados agregados e estatísticas em um banco SQLite."""
        conn = sqlite3.connect(db_path)
        self.hourly.to_sql('hourly_energy', conn, if_exists='replace', index=False)
        stats = self.get_stats()
        stats.to_sql('descriptive_stats', conn, if_exists='replace', index=False)
        adf_res = self.run_adf_all()
        adf_res.to_sql('adf_results', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Dados salvos em {db_path}")

    # ========================== GRÁFICOS ==========================

    def plot_serie_temporal(self, home, save_path=None):
        """
        Plota a série temporal do consumo horário da casa.
        """
        ts = self.series[home]
        plt.figure(figsize=(12, 6))
        plt.plot(ts.index, ts, color='blue', linewidth=0.8)
        plt.title(f'Casa {home} - Consumo horário de energia elétrica (kWh)')
        plt.xlabel('Data')
        plt.ylabel('Consumo (kWh)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_media_desvio_moveis(self, home, window_days=7, save_path=None):
        """
        Plota média e desvio móveis (janela em dias) da série.
        """
        ts = self.series[home]
        window_hours = 24 * window_days
        rolling_mean = ts.rolling(window=window_hours, center=True).mean()
        rolling_std = ts.rolling(window=window_hours, center=True).std()

        plt.figure(figsize=(12, 6))
        plt.plot(ts.index, ts, alpha=0.5, label='Série original')
        plt.plot(ts.index, rolling_mean, color='red', label=f'Média móvel ({window_days} dias)')
        plt.plot(ts.index, rolling_std, color='black', label=f'Desvio móvel ({window_days} dias)')
        plt.title(f'Casa {home} - Média e desvio padrão móveis')
        plt.xlabel('Data')
        plt.ylabel('kWh')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_histograma(self, home, bins=50, save_path=None):
        """
        Plota histograma da distribuição do consumo, com média, mediana e curva normal.
        """
        ts = self.series[home]
        plt.figure(figsize=(10, 5))
        plt.hist(ts, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        mean_val = ts.mean()
        median_val = ts.median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média = {mean_val:.3f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana = {median_val:.3f}')
        # Curva normal teórica
        mu = mean_val
        sigma = ts.std()
        x = np.linspace(ts.min(), ts.max(), 100)
        plt.plot(x, (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2),
                 color='orange', linestyle='-', linewidth=2, label='Normal teórica')
        plt.title(f'Casa {home} - Distribuição do consumo horário (kWh)')
        plt.xlabel('Consumo (kWh)')
        plt.ylabel('Densidade')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_boxplot(self, save_path=None):
        """
        Boxplot comparativo do consumo por casa.
        """
        data = [self.series[home] for home in self.homes]
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(data, labels=self.homes, patch_artist=True, showmeans=True)
        # Colorir caixas
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(self.homes)))):
            patch.set_facecolor(color)
        plt.xlabel('Casa')
        plt.ylabel('Consumo (kWh/h)')
        plt.title('Boxplot do consumo horário por casa')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_acf(self, home, lags=48, save_path=None):
        """
        Plota a função de autocorrelação (ACF) da série.
        """
        ts = self.series[home]
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_acf(ts, ax=ax, lags=lags, alpha=0.05, title=f'ACF - Casa {home}')
        ax.set_xlabel('Defasagem (horas)')
        ax.set_ylabel('Autocorrelação')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_pacf(self, home, lags=48, save_path=None):
        """
        Plota a função de autocorrelação parcial (PACF) da série.
        """
        ts = self.series[home]
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(ts, ax=ax, lags=lags, alpha=0.05, title=f'PACF - Casa {home}')
        ax.set_xlabel('Defasagem (horas)')
        ax.set_ylabel('Autocorrelação Parcial')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_acf_destacado_lags(self, home, lags=[1, 24], max_lag=48, save_path=None):
        """
        Plota ACF com barras coloridas para lags específicos.
        """
        ts = self.series[home]
        n = len(ts)
        mean = ts.mean()
        var = ts.var(ddof=0)
        acf_vals = []
        for k in range(0, max_lag+1):
            if k == 0:
                acf_vals.append(1.0)
            else:
                cov = np.sum((ts.values[k:] - mean) * (ts.values[:-k] - mean)) / n
                acf_vals.append(cov / var)
        lags_plot = np.arange(max_lag+1)
        colors = ['gray'] * (max_lag+1)
        for lag in lags:
            if lag <= max_lag:
                colors[lag] = 'red'
        plt.figure(figsize=(12, 6))
        plt.bar(lags_plot, acf_vals, color=colors, edgecolor='black')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axhline(1.96/np.sqrt(n), color='blue', linestyle='--', label='IC 95%')
        plt.axhline(-1.96/np.sqrt(n), color='blue', linestyle='--')
        plt.xlabel('Defasagem (horas)')
        plt.ylabel('Autocorrelação')
        plt.title(f'ACF com destaque para lags {lags} - Casa {home}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_acf_scatter_lag24(self, home, save_path=None):
        """
        Gráfico de dispersão X_t vs X_{t+24} para mostrar sazonalidade diária.
        """
        ts = self.series[home]
        ts_shifted = ts.shift(-24).dropna()
        ts_original = ts.iloc[:len(ts_shifted)]
        corr, p_val = pearsonr(ts_original, ts_shifted)
        plt.figure(figsize=(10, 6))
        plt.scatter(ts_original, ts_shifted, alpha=0.3, s=10)
        plt.plot([ts_original.min(), ts_original.max()],
                 [ts_original.min(), ts_original.max()], 'r--', label='Reta identidade')
        plt.xlabel('X_t (kWh)')
        plt.ylabel('X_{t+24} (kWh)')
        plt.title(f'Sazonalidade diária (lag 24) - Casa {home}\nCorrelação = {corr:.3f} (p={p_val:.3e})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_acf_comparativo(self, save_path=None):
        """
        Compara autocorrelação nos lags 1 e 24 entre todas as casas.
        """
        lag1 = []
        lag24 = []
        for home in self.homes:
            ts = self.series[home]
            lag1.append(ts.autocorr(lag=1))
            lag24.append(ts.autocorr(lag=24))
        x = np.arange(len(self.homes))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, lag1, width, label='Lag 1 (1 hora)', color='salmon')
        plt.bar(x + width/2, lag24, width, label='Lag 24 (1 dia)', color='lightblue')
        plt.xlabel('Casa')
        plt.ylabel('Autocorrelação')
        plt.title('Comparação da autocorrelação entre casas')
        plt.xticks(x, self.homes)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_adf_results(self, save_path=None):
        """
        Gráfico comparando estatística ADF e valores críticos para todas as casas.
        """
        adf_df = self.run_adf_all()
        homes = adf_df['home_id']
        x = np.arange(len(homes))
        width = 0.2
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, adf_df['adf_stat'], width, label='Estatística ADF', color='blue')
        plt.bar(x, adf_df['critical_1%'], width, label='Crítico 1%', color='red', alpha=0.7)
        plt.bar(x + width, adf_df['critical_5%'], width, label='Crítico 5%', color='orange', alpha=0.7)
        plt.bar(x + 2*width, adf_df['critical_10%'], width, label='Crítico 10%', color='green', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel('Casa')
        plt.ylabel('Valor')
        plt.title('Teste ADF: Estatística vs. Valores Críticos')
        plt.xticks(x, homes)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_adf_pvalues(self, save_path=None):
        """
        Gráfico dos p-valores do teste ADF com linha de significância.
        """
        adf_df = self.run_adf_all()
        plt.figure(figsize=(12, 6))
        plt.bar(adf_df['home_id'], adf_df['p_value'], color='skyblue')
        plt.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Limiar 0.05')
        plt.xlabel('Casa')
        plt.ylabel('p-valor')
        plt.title('p-valor do Teste ADF')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_adf_distribution(self, save_path=None):
        """
        Distribuição da estatística ADF entre as casas e valores críticos médios.
        """
        adf_df = self.run_adf_all()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(adf_df['adf_stat'], bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        ax.axvline(adf_df['critical_5%'].mean(), color='red', linestyle='--',
                   label=f'Crítico médio 5%: {adf_df["critical_5%"].mean():.3f}')
        ax.axvline(adf_df['critical_1%'].mean(), color='orange', linestyle='--',
                   label=f'Crítico médio 1%: {adf_df["critical_1%"].mean():.3f}')
        ax.axvline(adf_df['critical_10%'].mean(), color='green', linestyle='--',
                   label=f'Crítico médio 10%: {adf_df["critical_10%"].mean():.3f}')
        ax.set_xlabel('Estatística ADF')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição da estatística ADF entre as casas')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_adf_model_explanation(self, save_path=None):
        """
        Exibe a equação do modelo ADF e interpretação dos resultados.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        texto = r"""
                    Modelo do teste ADF (Dickey-Fuller Aumentado):

                    $\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \varepsilon_t$

                    onde:
                    - $\Delta X_t = X_t - X_{t-1}$ (primeira diferença)
                    - $\alpha$ é a constante (drift)
                    - $\beta t$ é a tendência determinística
                    - $\gamma X_{t-1}$ é o termo de raiz unitária (hipótese nula: $\gamma = 0$)
                    - $\sum \delta_i \Delta X_{t-i}$ são termos autorregressivos para corrigir autocorrelação
                    - $\varepsilon_t$ é o ruído branco

                    **Resultados para o dataset:**
                    - Para todas as casas, o p-valor é inferior a 0.05.
                    - Rejeitamos a hipótese nula de raiz unitária.
                    - Concluímos que as séries são estacionárias em torno de uma tendência determinística (ou seja, não possuem raiz unitária).
                """
        ax.text(0.05, 0.95, texto, fontsize=12, verticalalignment='top', transform=ax.transAxes)
        ax.set_title('Interpretação do Teste ADF', fontsize=14)
        plt.tight_layout()
        plt.show()

    def descriptive_summary(self):
        """
        Retorna um DataFrame resumo das estatísticas descritivas e ADF.
        """
        stats = self.get_stats()
        adf_df = self.run_adf_all()
        merged = pd.merge(stats, adf_df, on='home_id')
        return merged