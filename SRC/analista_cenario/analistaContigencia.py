#!/usr/bin/env python3
"""
ANALISADOR SIMPLIFICADO DE CONTINGÊNCIAS
Classe otimizada para análise e visualização de resultados
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

class AnalisadorContingencias:
    def __init__(self, db_path="resultados_opf_contingencias.db"):
        """Inicializa o analisador com o caminho do banco de dados"""
        self.db_path = db_path
        self.conn = None
        self.dados = {}
        
        print(f"📂 Analisador inicializado com banco: {self.db_path}")
    
    def conectar(self):
        """Conecta ao banco de dados"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.db_path}")
        
        self.conn = sqlite3.connect(self.db_path)
        print("✅ Conectado ao banco de dados")
    
    def carregar_dados(self):
        """Carrega todos os dados do banco de forma simplificada"""
        if self.conn is None:
            self.conectar()
        
        print("\n📥 CARREGANDO DADOS...")
        
        # Tabelas principais
        tabelas = {
            'remocao': 'remocao_linha_contingencias',
            'duplicacao': 'duplicacao_linha_contingencias',
            'remocao_geradores': 'remocao_linha_geradores', 
            'duplicacao_geradores': 'duplicacao_linha_geradores',
            'remocao_linhas': 'remocao_linha_linhas',
            'duplicacao_linhas': 'duplicacao_linha_linhas'
        }
        
        for nome, tabela in tabelas.items():
            try:
                self.dados[nome] = pd.read_sql_query(f"SELECT * FROM {tabela}", self.conn)
                print(f"✅ {tabela}: {len(self.dados[nome])} registros")
            except Exception as e:
                print(f"❌ {tabela}: {e}")
                self.dados[nome] = pd.DataFrame()
    
    def resumo_dados(self):
        """Mostra resumo simples dos dados carregados"""
        print("\n📊 RESUMO DOS DADOS:")
        print("=" * 40)
        
        if 'remocao' in self.dados and not self.dados['remocao'].empty:
            linhas = self.dados['remocao']['linha_removida'].unique()
            print(f"🔴 REMOÇÃO: {len(linhas)} contingências")
            print(f"   Linhas: {list(linhas)}")
        
        if 'duplicacao' in self.dados and not self.dados['duplicacao'].empty:
            linhas = self.dados['duplicacao']['linha_original'].unique()
            print(f"🔵 DUPLICAÇÃO: {len(linhas)} contingências") 
            print(f"   Linhas: {list(linhas)}")
        
        print("=" * 40)
    
    def _corrigir_custos_negativos(self, df):
        """Corrige custos negativos nos dados"""
        if 'custo_total_usd_h' in df.columns:
            custos_negativos = df[df['custo_total_usd_h'] < 0]
            if not custos_negativos.empty:
                print(f"⚠️  Corrigindo {len(custos_negativos)} custos negativos")
                df['custo_total_usd_h'] = df['custo_total_usd_h'].clip(lower=0)
        return df
    
    def plotar_custos(self):
        """Gráfico simples de custos para remoção e duplicação"""
        print("\n💰 GERANDO GRÁFICO DE CUSTOS")
        
        fig = go.Figure()
        
        # Dados de remoção
        if 'remocao' in self.dados and not self.dados['remocao'].empty:
            df_remocao = self._corrigir_custos_negativos(self.dados['remocao'])
            df_remocao = df_remocao[df_remocao['linha_removida'] != 'CASO_BASE']
            
            if not df_remocao.empty:
                fig.add_trace(go.Bar(
                    name='Remoção',
                    x=df_remocao['linha_removida'],
                    y=df_remocao['custo_total_usd_h'],
                    marker_color='red',
                    text=df_remocao['custo_total_usd_h'].round(0),
                    textposition='auto'
                ))
        
        # Dados de duplicação
        if 'duplicacao' in self.dados and not self.dados['duplicacao'].empty:
            df_duplicacao = self._corrigir_custos_negativos(self.dados['duplicacao'])
            
            if not df_duplicacao.empty:
                fig.add_trace(go.Bar(
                    name='Duplicação',
                    x=df_duplicacao['linha_original'],
                    y=df_duplicacao['custo_total_usd_h'],
                    marker_color='blue', 
                    text=df_duplicacao['custo_total_usd_h'].round(0),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='Custo das Contingências',
            xaxis_title='Linha',
            yaxis_title='Custo (USD/h)',
            barmode='group',
            height=500
        )
        
        fig.show()
        
        # Estatísticas
        self._mostrar_estatisticas_custos()
    
    def _mostrar_estatisticas_custos(self):
        """Mostra estatísticas simples de custos"""
        print("\n📊 ESTATÍSTICAS DE CUSTOS:")
        
        if 'remocao' in self.dados and not self.dados['remocao'].empty:
            df_remocao = self._corrigir_custos_negativos(self.dados['remocao'])
            df_remocao = df_remocao[df_remocao['linha_removida'] != 'CASO_BASE']
            
            if not df_remocao.empty:
                custo_medio = df_remocao['custo_total_usd_h'].mean()
                custo_max = df_remocao['custo_total_usd_h'].max()
                print(f"🔴 REMOÇÃO: Média = {custo_medio:.0f} USD/h, Máximo = {custo_max:.0f} USD/h")
        
        if 'duplicacao' in self.dados and not self.dados['duplicacao'].empty:
            df_duplicacao = self._corrigir_custos_negativos(self.dados['duplicacao'])
            
            if not df_duplicacao.empty:
                custo_medio = df_duplicacao['custo_total_usd_h'].mean()
                custo_max = df_duplicacao['custo_total_usd_h'].max()
                print(f"🔵 DUPLICAÇÃO: Média = {custo_medio:.0f} USD/h, Máximo = {custo_max:.0f} USD/h")
    
    def plotar_geracao_por_tipo(self):
        """Gráfico de geração por tipo de gerador (incluindo DEF)"""
        print("\n⚡ GERANDO GRÁFICO DE GERAÇÃO POR TIPO")
        
        if 'remocao_geradores' not in self.dados or self.dados['remocao_geradores'].empty:
            print("❌ Dados de geradores não disponíveis")
            return
        
        df_geradores = self.dados['remocao_geradores']
        
        # Agrupar por contingência e tipo
        geracao_agrupada = df_geradores.groupby(['id_contingencia', 'tipo'])['geracao_pu'].sum().reset_index()
        
        # Juntar com dados principais para obter nome da linha
        if 'remocao' in self.dados and not self.dados['remocao'].empty:
            df_remocao = self.dados['remocao'][['id_contingencia', 'linha_removida']]
            geracao_agrupada = geracao_agrupada.merge(df_remocao, on='id_contingencia', how='left')
        
        # Pivot para ter colunas por tipo
        geracao_pivot = geracao_agrupada.pivot_table(
            index='linha_removida',
            columns='tipo',
            values='geracao_pu',
            fill_value=0
        ).reset_index()
        
        # Ordenar excluindo caso base
        geracao_pivot = geracao_pivot[geracao_pivot['linha_removida'] != 'CASO_BASE']
        
        fig = go.Figure()
        
        # Cores para cada tipo de gerador (incluindo DEF)
        cores = {'UTH': 'blue', 'UTE': 'red', 'GWD': 'green', 'DEF': 'black'}
        
        # Ordem de empilhamento: DEF primeiro (negativo), depois os outros
        tipos_ordenados = ['DEF'] + [t for t in geracao_pivot.columns if t != 'linha_removida' and t != 'DEF']
        
        for tipo in tipos_ordenados:
            if tipo in geracao_pivot.columns and tipo in cores:
                fig.add_trace(go.Bar(
                    name=tipo,
                    x=geracao_pivot['linha_removida'],
                    y=geracao_pivot[tipo],
                    marker_color=cores[tipo],
                    text=geracao_pivot[tipo].round(3),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='Geração por Tipo de Gerador - Contingências de Remoção',
            xaxis_title='Linha Removida',
            yaxis_title='Geração (pu)',
            barmode='stack',
            height=500
        )
        
        fig.show()
        
        # Estatísticas
        print(f"\n📊 DISTRIBUIÇÃO DE GERAÇÃO POR TIPO:")
        for tipo in geracao_pivot.columns:
            if tipo != 'linha_removida':
                media = geracao_pivot[tipo].mean()
                if abs(media) > 0.001:  # Considerar apenas valores significativos
                    print(f"   {tipo}: {media:.3f} pu (média)")
    
    def plotar_fluxos_linhas(self):
        """Gráfico de fluxos nas linhas"""
        print("\n📈 GERANDO GRÁFICO DE FLUXOS")
        
        if 'remocao_linhas' not in self.dados or self.dados['remocao_linhas'].empty:
            print("❌ Dados de linhas não disponíveis")
            return
        
        df_linhas = self.dados['remocao_linhas']
        
        # Agrupar por linha e calcular estatísticas
        fluxos_por_linha = df_linhas.groupby('id_linha').agg({
            'fluxo_pu': ['mean', 'max', 'min'],
            'utilizacao_percentual': 'mean',
            'id_barra_origem': 'first',
            'id_barra_destino': 'first'
        }).round(3)
        
        fluxos_por_linha.columns = ['fluxo_medio', 'fluxo_max', 'fluxo_min', 'utilizacao_media', 'origem', 'destino']
        fluxos_por_linha = fluxos_por_linha.reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Fluxo Médio',
            x=fluxos_por_linha['id_linha'],
            y=fluxos_por_linha['fluxo_medio'],
            marker_color='green',
            text=fluxos_por_linha['fluxo_medio'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Fluxo Médio nas Linhas',
            xaxis_title='Linha',
            yaxis_title='Fluxo (pu)',
            height=500
        )
        
        fig.show()
        
        # Mostrar linhas mais carregadas
        linhas_carregadas = fluxos_por_linha.nlargest(5, 'utilizacao_media')
        print(f"\n📊 LINHAS MAIS CARREGADAS:")
        for _, linha in linhas_carregadas.iterrows():
            print(f"   {linha['id_linha']}: {linha['utilizacao_media']:.1f}%")
    
    def plotar_comparacao_geradores(self):
        """Comparação simples entre geração de diferentes tipos (incluindo DEF)"""
        print("\n🔧 GERANDO GRÁFICO DE DISTRIBUIÇÃO DE GERAÇÃO")
        
        if 'remocao_geradores' not in self.dados or self.dados['remocao_geradores'].empty:
            print("❌ Dados de geradores não disponíveis")
            return
        
        df_geradores = self.dados['remocao_geradores']
        
        # Agrupar por tipo de gerador (valor absoluto para DEF)
        geracao_por_tipo = df_geradores.groupby('tipo')['geracao_pu'].sum().reset_index()
        
        # Para DEF, usar valor absoluto para visualização
        geracao_por_tipo['geracao_abs'] = geracao_por_tipo['geracao_pu'].abs()
        
        fig = px.pie(
            geracao_por_tipo, 
            values='geracao_abs', 
            names='tipo',
            title='Distribuição de Geração por Tipo de Gerador',
            color='tipo',
            color_discrete_map={'UTH': 'blue', 'UTE': 'red', 'GWD': 'green', 'DEF': 'black'}
        )
        
        fig.show()
        
        print(f"\n📊 DISTRIBUIÇÃO DE GERAÇÃO:")
        for _, tipo in geracao_por_tipo.iterrows():
            sinal = "" if tipo['geracao_pu'] >= 0 else "-"
            print(f"   {tipo['tipo']}: {sinal}{abs(tipo['geracao_pu']):.3f} pu")
    
    def plotar_deficit(self):
        """Gráfico de déficit de potência"""
        print("\n⚠️  GERANDO GRÁFICO DE DÉFICIT")
        
        fig = go.Figure()
        
        # Dados de remoção
        if 'remocao' in self.dados and not self.dados['remocao'].empty:
            df_remocao = self.dados['remocao']
            df_remocao = df_remocao[df_remocao['linha_removida'] != 'CASO_BASE']
            
            if not df_remocao.empty:
                fig.add_trace(go.Bar(
                    name='Déficit Remoção',
                    x=df_remocao['linha_removida'],
                    y=df_remocao['total_deficit_pu'],
                    marker_color='darkred',
                    text=df_remocao['total_deficit_pu'].round(4),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='Déficit de Potência nas Contingências',
            xaxis_title='Linha Removida',
            yaxis_title='Déficit (pu)',
            height=500
        )
        
        fig.show()
        
        # Identificar contingências com déficit
        if 'remocao' in self.dados and not self.dados['remocao'].empty:
            df_com_deficit = self.dados['remocao'][self.dados['remocao']['total_deficit_pu'] > 0.001]
            if not df_com_deficit.empty:
                print(f"\n🚨 CONTINGÊNCIAS COM DÉFICIT:")
                for _, row in df_com_deficit.iterrows():
                    print(f"   {row['linha_removida']}: {row['total_deficit_pu']:.4f} pu")
    
    def plotar_analise_custos_detalhada(self):
        """Gráfico detalhado de análise de custos incluindo custo de operação e custo marginal de fluxo"""
        print("\n💰 GERANDO ANÁLISE DETALHADA DE CUSTOS")
        
        if 'remocao' not in self.dados or self.dados['remocao'].empty:
            print("❌ Dados de contingências não disponíveis")
            return
        
        if 'remocao_linhas' not in self.dados or self.dados['remocao_linhas'].empty:
            print("❌ Dados de linhas não disponíveis")
            return
        
        # Obter dados de contingências
        df_remocao = self._corrigir_custos_negativos(self.dados['remocao'])
        df_remocao = df_remocao[df_remocao['linha_removida'] != 'CASO_BASE']
        
        if df_remocao.empty:
            print("❌ Nenhuma contingência de remoção disponível")
            return
        
        # Calcular custo marginal de fluxo para cada contingência
        custos_marginais_fluxo = []
        
        for _, contingencia in df_remocao.iterrows():
            id_contingencia = contingencia['id_contingencia']
            linha_removida = contingencia['linha_removida']
            
            # Filtrar linhas para esta contingência
            df_linhas_contingencia = self.dados['remocao_linhas'][
                self.dados['remocao_linhas']['id_contingencia'] == id_contingencia
            ]
            
            # Calcular custo marginal total de fluxo (soma dos duais positivos e negativos)
            custo_marginal_fluxo = df_linhas_contingencia['dual_fluxo_pos'].sum() + df_linhas_contingencia['dual_fluxo_neg'].sum()
            
            custos_marginais_fluxo.append({
                'linha_removida': linha_removida,
                'custo_marginal_fluxo': custo_marginal_fluxo,
                'custo_operacao': contingencia['custo_total_usd_h'],  # Usando custo total como proxy
                'total_geracao': contingencia['total_geracao_pu'],
                'total_deficit': contingencia['total_deficit_pu']
            })
        
        df_custos = pd.DataFrame(custos_marginais_fluxo)
        
        # Criar gráfico de barras agrupadas
        fig = go.Figure()
        
        # Custo de Operação
        fig.add_trace(go.Bar(
            name='Custo de Operação',
            x=df_custos['linha_removida'],
            y=df_custos['custo_operacao'],
            marker_color='blue',
            text=df_custos['custo_operacao'].round(0),
            textposition='auto'
        ))
        
        # Custo Marginal de Fluxo (escala diferente)
        fig.add_trace(go.Bar(
            name='Custo Marginal de Fluxo',
            x=df_custos['linha_removida'],
            y=df_custos['custo_marginal_fluxo'],
            marker_color='red',
            text=df_custos['custo_marginal_fluxo'].round(2),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Análise Detalhada de Custos - Operação vs Fluxo Marginal',
            xaxis_title='Linha Removida',
            yaxis_title='Custo (USD/h)',
            barmode='group',
            height=600,
            showlegend=True
        )
        
        fig.show()
        
        # Gráfico de pizza mostrando proporção dos custos
        custo_total_operacao = df_custos['custo_operacao'].sum()
        custo_total_fluxo = df_custos['custo_marginal_fluxo'].sum()
        
        fig_pizza = go.Figure(data=[go.Pie(
            labels=['Custo de Operação', 'Custo Marginal de Fluxo'],
            values=[custo_total_operacao, custo_total_fluxo],
            hole=0.4,
            marker_colors=['blue', 'red']
        )])
        
        fig_pizza.update_layout(
            title='Proporção dos Custos Totais'
        )
        
        fig_pizza.show()
        
        # Análise estatística detalhada
        self._mostrar_analise_custos_detalhada(df_custos)
    
    def _mostrar_analise_custos_detalhada(self, df_custos):
        """Mostra análise detalhada dos custos"""
        print("\n📊 ANÁLISE DETALHADA DE CUSTOS:")
        print("=" * 50)
        
        # Estatísticas básicas
        custo_medio_operacao = df_custos['custo_operacao'].mean()
        custo_medio_fluxo = df_custos['custo_marginal_fluxo'].mean()
        
        print(f"💰 CUSTO MÉDIO POR CONTINGÊNCIA:")
        print(f"   Operação: {custo_medio_operacao:.2f} USD/h")
        print(f"   Fluxo Marginal: {custo_medio_fluxo:.2f} USD/h")
        print(f"   Total: {custo_medio_operacao + custo_medio_fluxo:.2f} USD/h")
        
        # Contingência mais cara
        idx_mais_cara = df_custos['custo_operacao'].idxmax()
        mais_cara = df_custos.loc[idx_mais_cara]
        
        print(f"\n🚨 CONTINGÊNCIA MAIS CRÍTICA:")
        print(f"   Linha: {mais_cara['linha_removida']}")
        print(f"   Custo Operação: {mais_cara['custo_operacao']:.2f} USD/h")
        print(f"   Custo Fluxo: {mais_cara['custo_marginal_fluxo']:.2f} USD/h")
        print(f"   Déficit: {mais_cara['total_deficit']:.4f} pu")
        
        # Correlação entre custos
        correlacao = df_custos['custo_operacao'].corr(df_custos['custo_marginal_fluxo'])
        print(f"\n📈 CORRELAÇÃO ENTRE CUSTOS:")
        print(f"   Operação vs Fluxo: {correlacao:.3f}")
        
        if correlacao > 0.7:
            print("   💡 Alta correlação: custos tendem a aumentar juntos")
        elif correlacao < -0.7:
            print("   💡 Alta correlação negativa: quando um custo aumenta, o outro diminui")
        else:
            print("   💡 Baixa correlação: custos variam independentemente")
    
    def plotar_custos_vs_deficit(self):
        """Gráfico de dispersão: Custo vs Déficit"""
        print("\n📈 GERANDO ANÁLISE: CUSTO vs DÉFICIT")
        
        if 'remocao' not in self.dados or self.dados['remocao'].empty:
            print("❌ Dados de contingências não disponíveis")
            return
        
        df_remocao = self._corrigir_custos_negativos(self.dados['remocao'])
        df_remocao = df_remocao[df_remocao['linha_removida'] != 'CASO_BASE']
        
        if df_remocao.empty:
            return
        
        fig = px.scatter(
            df_remocao,
            x='total_deficit_pu',
            y='custo_total_usd_h',
            size='total_geracao_pu',
            color='linha_removida',
            title='Relação: Custo vs Déficit vs Geração',
            labels={
                'total_deficit_pu': 'Déficit (pu)',
                'custo_total_usd_h': 'Custo Total (USD/h)',
                'total_geracao_pu': 'Geração Total (pu)',
                'linha_removida': 'Linha Removida'
            },
            size_max=20
        )
        
        fig.update_layout(height=600)
        fig.show()
        
        # Análise da relação
        correlacao = df_remocao['total_deficit_pu'].corr(df_remocao['custo_total_usd_h'])
        print(f"\n📊 CORRELAÇÃO DÉFICIT-CUSTO: {correlacao:.3f}")
        
        if correlacao > 0.5:
            print("💡 Forte relação positiva: déficit aumenta o custo significativamente")
        elif correlacao < -0.5:
            print("💡 Forte relação negativa: déficit reduz custos (improvável)")
        else:
            print("💡 Relação fraca: outros fatores influenciam mais os custos")
    
    def plotar_evolucao_custos_marginais(self):
        """Gráfico de evolução dos custos marginais por linha"""
        print("\n📊 GERANDO EVOLUÇÃO DOS CUSTOS MARGINAIS")
        
        if 'remocao_linhas' not in self.dados or self.dados['remocao_linhas'].empty:
            print("❌ Dados de linhas não disponíveis")
            return
        
        df_linhas = self.dados['remocao_linhas']
        
        # Agrupar por linha e calcular custos marginais médios
        custos_por_linha = df_linhas.groupby('id_linha').agg({
            'dual_fluxo_pos': 'mean',
            'dual_fluxo_neg': 'mean',
            'fluxo_pu': 'mean',
            'utilizacao_percentual': 'mean'
        }).reset_index()
        
        # Calcular custo marginal total
        custos_por_linha['custo_marginal_total'] = (
            custos_por_linha['dual_fluxo_pos'] + custos_por_linha['dual_fluxo_neg']
        )
        
        # Ordenar por custo marginal
        custos_por_linha = custos_por_linha.sort_values('custo_marginal_total', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Custo Marginal Total',
            x=custos_por_linha['id_linha'],
            y=custos_por_linha['custo_marginal_total'],
            marker_color='purple',
            text=custos_por_linha['custo_marginal_total'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Custo Marginal de Fluxo por Linha',
            xaxis_title='Linha',
            yaxis_title='Custo Marginal (USD/h)',
            height=500
        )
        
        fig.show()
        
        # Mostrar linhas com maior custo marginal
        print(f"\n🚨 LINHAS COM MAIOR CUSTO MARGINAL:")
        top_linhas = custos_por_linha.head(3)
        for _, linha in top_linhas.iterrows():
            print(f"   {linha['id_linha']}: {linha['custo_marginal_total']:.3f} USD/h "
                  f"(Utilização: {linha['utilizacao_percentual']:.1f}%)")

    
    def analise_completa_custos(self):
        """Executa análise completa de custos"""
        print("💰 INICIANDO ANÁLISE COMPLETA DE CUSTOS")
        print("=" * 60)
        
        # Carregar dados se necessário
        if not self.dados:
            self.carregar_dados()
        
        # Executar todas as análises de custo
        self.plotar_analise_custos_detalhada()
        self.plotar_custos_vs_deficit()
        self.plotar_evolucao_custos_marginais()
        
        print("\n✅ ANÁLISE DE CUSTOS CONCLUÍDA!")

    def analise_rapida(self):
        """Executa uma análise rápida e completa"""
        print("🚀 INICIANDO ANÁLISE RÁPIDA")
        print("=" * 50)
        
        # Carregar dados
        self.carregar_dados()
        self.resumo_dados()
        
        # Gerar gráficos principais
        self.plotar_custos()
        self.plotar_geracao_por_tipo()
        self.plotar_fluxos_linhas()
        self.plotar_comparacao_geradores()
        self.plotar_deficit()
        
        # NOVA ANÁLISE DE CUSTOS DETALHADA
        self.analise_completa_custos()
        
        print("\n✅ ANÁLISE CONCLUÍDA!")

    def analise_rapida(self):
        """Executa uma análise rápida e completa"""
        print("🚀 INICIANDO ANÁLISE RÁPIDA")
        print("=" * 50)
        
        # Carregar dados
        self.carregar_dados()
        self.resumo_dados()
        
        # Gerar gráficos principais
        self.plotar_custos()
        self.plotar_geracao_por_tipo()
        self.plotar_fluxos_linhas()
        self.plotar_comparacao_geradores()
        self.plotar_deficit()
        
        print("\n✅ ANÁLISE CONCLUÍDA!")
    
    def fechar(self):
        """Fecha a conexão com o banco"""
        if self.conn:
            self.conn.close()
            print("🔒 Conexão fechada")
    
    def __enter__(self):
        """Suporte para context manager"""
        self.conectar()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Suporte para context manager"""
        self.fechar()

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal simplificada"""
    try:
        analisador = AnalisadorContingencias()
        analisador.analise_rapida()
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()