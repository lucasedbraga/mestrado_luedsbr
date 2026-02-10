# renewable_data_loader.py
import pandas as pd
import os

def load_renewable_data(filepath=None):
    """
    Carrega dados de geração renovável e retorna DataFrames separados para vento e solar.
    
    Args:
        filepath: Caminho para o arquivo CSV (opcional, busca automaticamente se None)
    
    Returns:
        tuple: (df_wind, df_solar) - DataFrames com dados de vento e solar
    """
    
    # Caminho padrão do arquivo no seu projeto
    if filepath is None:
        filepath = r"C:\Users\lucas\repositorios\mestrado_luedsbr\SRC\SOLVER\DB\getters\intermittent-renewables-production-france.csv"
    
    # Carregar dados brutos
    print(f"Carregando: {filepath}")
    df = pd.read_csv(filepath)
    
    # Processar data/hora
    df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6))
    df = df.sort_values('DateTime')
    
    # Separar vento e solar
    df_wind = df[df['Source'] == 'Wind'].copy()
    df_solar = df[df['Source'] == 'Solar'].copy()
    
    # Normalizar produção (fator 0-1)
    for df_source in [df_wind, df_solar]:
        if len(df_source) > 0:
            max_prod = df_source['Production'].max()
            if max_prod > 0:
                df_source['Factor'] = df_source['Production'] / max_prod
            else:
                df_source['Factor'] = 0
    
    # Selecionar colunas finais
    cols = ['DateTime', 'Production', 'Factor']
    df_wind = df_wind[cols].copy()
    df_solar = df_solar[cols].copy()
    
    # Renomear
    df_wind.columns = ['timestamp', 'wind_production_mw', 'wind_factor']
    df_solar.columns = ['timestamp', 'solar_production_mw', 'solar_factor']
    
    print(f"✓ Vento: {len(df_wind)} registros")
    print(f"✓ Solar: {len(df_solar)} registros")
    
    return df_wind, df_solar

# Uso mínimo
if __name__ == "__main__":
    df_wind, df_solar = load_renewable_data()
    print("\nPrimeiras linhas - Vento:")
    print(df_wind.head())
    print("\nPrimeiras linhas - Solar:")
    print(df_solar.head())