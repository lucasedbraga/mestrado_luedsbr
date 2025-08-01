import sqlite3
import pandas as pd
import os

# --- Configurações ---
csv_path = "DATA/input/input_teste.csv"              # Caminho do CSV
sqlite_path = "DATA/input/input_teste.db"       # Caminho do SQLite
tabela = "input_teste"                       # Nome da tabela

# --- Verificações iniciais ---
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

# --- Leitura do CSV ---
df = pd.read_csv(csv_path, sep=';')

print(df)
# --- Conexão com SQLite ---
conn = sqlite3.connect(sqlite_path)
cursor = conn.cursor()

# --- Criar a tabela (sobrescrevendo se existir) ---
cursor.execute(f"DROP TABLE IF EXISTS {tabela}")
col_defs = ", ".join(f"{col} REAL" for col in df.columns)
cursor.execute(f"CREATE TABLE {tabela} ({col_defs})")

# --- Inserir os dados ---
df.to_sql(tabela, conn, if_exists='append', index=False)

# --- Encerrar ---
conn.commit()
conn.close()

print(f"Tabela '{tabela}' criada com sucesso em '{sqlite_path}' com {len(df)} linhas.")
