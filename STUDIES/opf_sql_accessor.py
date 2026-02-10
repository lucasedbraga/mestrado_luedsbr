import sqlite3
import pandas as pd
import os

def load_opf_results(db_path):
    """Carrega todas as tabelas relevantes do resultados_PL.db em DataFrames."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {db_path}")
    conn = sqlite3.connect(db_path)
    dfs = {}
    for table in ['resultados_opf', 'detalhes_barras', 'detalhes_geradores', 'detalhes_linhas', 'bess_wind_operacao']:
        try:
            dfs[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception as e:
            print(f"Erro ao ler tabela {table}: {e}")
    conn.close()
    return dfs

# Exemplo de uso:
# dfs = load_opf_results('DATA/SMA/resultados_PL.db')
# print(dfs['detalhes_barras'].head())
