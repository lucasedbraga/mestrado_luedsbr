import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class OPFAnalyzer:
    def __init__(self, db_path="resultados_opf_series.db"):
        self.db_path = db_path
        self.conn = None
        self.dfs = {}
        
    def connect(self):
        """Conectar ao banco de dados"""
        self.conn = sqlite3.connect(self.db_path)
        
    def load_all_data(self):
        """Carregar todos os dados em DataFrames"""
        if self.conn is None:
            self.connect()
            
        tables = ['cenarios', 'barras', 'geradores', 'linhas', 'indicadores']
        
        for table in tables:
            try:
                query = f"SELECT * FROM {table}"
                self.dfs[table] = pd.read_sql_query(query, self.conn)
            except:
                self.dfs[table] = pd.DataFrame()
        
        return self.dfs
    
    def get_scenario_summary(self):
        """Resumo geral dos cenários"""
        if 'cenarios' not in self.dfs:
            self.load_all_data()
            
        df = self.dfs['cenarios']
        print(f"Total de cenários: {len(df)}")
        
        if len(df) > 0:
            stats = df[['total_geracao_pu', 'total_carga_pu', 'custo_total_usd_h']].describe()
            print(stats.round(4))
            
        return df
    
    def close(self):
        """Fechar conexão"""
        if self.conn:
            self.conn.close()

# ==============================================================================
# FUNÇÕES SIMPLIFICADAS PARA ANÁLISE
# ==============================================================================

def plot_comparacao_cenarios(analyzer):
    """Gráfico simples comparando carga, geração GWD e curtailment"""
    cenarios_df = analyzer.dfs.get('cenarios', pd.DataFrame())
    geradores_df = analyzer.dfs.get('geradores', pd.DataFrame())
    
    if len(cenarios_df) == 0:
        print("Nenhum cenário disponível")
        return
    
    # Preparar dados
    dados = []
    for _, cenario in cenarios_df.iterrows():
        cenario_id = cenario['id_cenario']
        
        # Geração GWD
        geradores_cenario = geradores_df[geradores_df['id_cenario'] == cenario_id]
        geracao_gwd = geradores_cenario[geradores_cenario['tipo'] == 'GWD']['geracao_pu'].sum()
        curtailment = geradores_cenario[geradores_cenario['tipo'] == 'CUR']['geracao_pu'].sum()
        
        dados.append({
            'cenario': cenario_id,
            'carga': cenario.get('total_carga_pu', 0),
            'geracao_gwd': geracao_gwd,
            'curtailment': curtailment,
            'geracao_liquida': geracao_gwd - curtailment
        })
    
    df = pd.DataFrame(dados)
    
    # Gráfico de barras agrupadas
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Carga', x=df['cenario'], y=df['carga'], marker_color='red'))
    fig.add_trace(go.Bar(name='Geração GWD', x=df['cenario'], y=df['geracao_gwd'], marker_color='green'))
    fig.add_trace(go.Bar(name='Curtailment', x=df['cenario'], y=df['curtailment'], marker_color='orange'))
    
    fig.update_layout(
        title='Comparação entre Cenários',
        xaxis_title='Cenários',
        yaxis_title='Potência (pu)',
        barmode='group'
    )
    
    fig.show()
    return df

def analisar_rampas_simples(analyzer):
    """Análise simplificada das rampas eólicas com demanda e curtailment"""
    cenarios_df = analyzer.dfs.get('cenarios', pd.DataFrame())
    geradores_df = analyzer.dfs.get('geradores', pd.DataFrame())
    
    if len(cenarios_df) < 2:
        print("Mínimo 2 cenários necessários")
        return
    
    # Coletar dados por cenário
    dados = []
    for _, cenario in cenarios_df.iterrows():
        cenario_id = cenario['id_cenario']
        geradores_cenario = geradores_df[geradores_df['id_cenario'] == cenario_id]
        
        geracao_gwd = geradores_cenario[geradores_cenario['tipo'] == 'GWD']['geracao_pu'].sum()
        curtailment = geradores_cenario[geradores_cenario['tipo'] == 'CUR']['geracao_pu'].sum()
        demanda = cenario.get('total_carga_pu', 0)
        
        dados.append({
            'cenario': cenario_id,
            'geracao_gwd': geracao_gwd,
            'curtailment': curtailment,
            'demanda': demanda,
            'geracao_liquida': geracao_gwd - curtailment
        })
    
    df_dados = pd.DataFrame(dados)
    
    # Calcular rampas
    rampas = []
    for i in range(1, len(df_dados)):
        anterior = df_dados.iloc[i-1]
        atual = df_dados.iloc[i]
        
        delta_geracao = atual['geracao_gwd'] - anterior['geracao_gwd']
        delta_curtailment = atual['curtailment'] - anterior['curtailment']
        delta_demanda = atual['demanda'] - anterior['demanda']
        
        rampa_geracao = abs(delta_geracao)
        rampa_curtailment = abs(delta_curtailment)
        rampa_demanda = abs(delta_demanda)
        
        # direcao_geracao = "↑" if delta_geracao > 0 else "↓" if delta_geracao < 0 else "→"
        # direcao_curtailment = "↑" if delta_curtailment > 0 else "↓" if delta_curtailment < 0 else "→"
        # direcao_demanda = "↑" if delta_demanda > 0 else "↓" if delta_demanda < 0 else "→"
        
        rampas.append({
            'transicao': f"{i}→{i+1}",
            'rampa_geracao': rampa_geracao,
            'rampa_curtailment': rampa_curtailment,
            'rampa_demanda': rampa_demanda,
            # 'direcao_geracao': direcao_geracao,
            # 'direcao_curtailment': direcao_curtailment,
            # 'direcao_demanda': direcao_demanda,
            'variacao_geracao': delta_geracao,
            'variacao_curtailment': delta_curtailment,
            'variacao_demanda': delta_demanda
        })
    
    df_rampas = pd.DataFrame(rampas)
    df_rampas.to_excel('rampas_TASTEE3_IEEE_118.xlsx')
    # Exibir DataFrame
    print("Rampas da Geração Eólica, Curtailment e Demanda:")
    print(df_rampas.round(4))
    
    # Gráfico com múltiplas séries
    fig = go.Figure()
    
    # Geração GWD
    fig.add_trace(go.Scatter(
        x=df_dados.index, 
        y=df_dados['geracao_gwd'],
        mode='lines+markers',
        name='Geração GWD',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    # Demanda
    fig.add_trace(go.Scatter(
        x=df_dados.index, 
        y=df_dados['demanda'],
        mode='lines+markers',
        name='Demanda',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Curtailment
    fig.add_trace(go.Scatter(
        x=df_dados.index, 
        y=df_dados['curtailment'],
        mode='lines+markers',
        name='Curtailment',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    # Geração Líquida
    fig.add_trace(go.Scatter(
        x=df_dados.index, 
        y=df_dados['geracao_liquida'],
        mode='lines+markers',
        name='Geração Líquida',
        line=dict(color='blue', width=2, dash='dot'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Evolução da Geração Eólica, Demanda e Curtailment',
        xaxis_title='Ordem dos Cenários',
        yaxis_title='Potência (pu)',
        height=500
    )
    
    fig.show()

    
    
    return df_rampas, df_dados

def plot_custos_simples(analyzer):
    """Gráfico simples de evolução de custos"""
    cenarios_df = analyzer.dfs.get('cenarios', pd.DataFrame())
    
    if len(cenarios_df) == 0:
        return
    
    fig = px.line(cenarios_df, x=cenarios_df.index, y='custo_total_usd_h',
                 title='Evolução do Custo Total',
                 markers=True)
    fig.show()
    
    return cenarios_df[['id_cenario', 'custo_total_usd_h']]

# ==============================================================================
# USO SIMPLES
# ==============================================================================

def analise_simples(db_path="resultados_opf_series.db"):
    """Análise completa simplificada"""
    print(db_path)
    analyzer = OPFAnalyzer(db_path=db_path)
    
    try:
        # Carregar dados
        analyzer.load_all_data()
        
        # Resumo
        print("=== RESUMO DOS CENÁRIOS ===")
        analyzer.get_scenario_summary()
        
        # Gráficos essenciais
        print("\n=== GRÁFICOS ===")
        #plot_comparacao_cenarios(analyzer)
        analisar_rampas_simples(analyzer)
        #plot_custos_simples(analyzer)
        
    finally:
        analyzer.close()

# Executar análise
if __name__ == "__main__":
    # Se executado como script, usar caminho padrão
    analise_simples()

# Para usar com caminho específico (no notebook)
#analise_simples(db_path=r'C:\Users\lucas\repositorios\mestrado_luedsbr\relatorios\resultados_opf_series.db')