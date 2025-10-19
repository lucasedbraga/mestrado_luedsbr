#!/usr/bin/env python3
"""
Sistema de Orquestração Completo para Análise de Contingências em Redes Elétricas
Autor: Sistema de Análise de Contingências
Data: 2024
"""

import os
import sys
import json
import sqlite3
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import argparse
import glob
import time
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orquestracao_opf.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OPFOrchestrator:
    def __init__(self, config_file="config_orquestracao.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.julia_script = "fluxPotContigencia.jl"
        self.results_db = "resultados_opf_contingencias.db"
        self.series_db = "resultados_opf_series.db"
        
    def _load_config(self):
        """Carregar configuração do arquivo JSON"""
        default_config = {
            "num_execucoes": 10,
            "sistemas": ["3barras_BASE.json"],
            "julia_path": "julia",
            "output_dir": "resultados_series",
            "analise_automática": True,
            "gerar_relatorios": True
        }
        
        try:
            with open(self.config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
            logger.info(f"Configuração carregada de {self.config_file}")
        except FileNotFoundError:
            logger.warning(f"Arquivo de configuração {self.config_file} não encontrado. Usando configuração padrão.")
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Arquivo de configuração padrão criado: {self.config_file}")
        
        return default_config
    
    def executar_analise_contingencias(self, sistema, execucao_id):
        """Executar análise de contingências para um sistema específico"""
        logger.info(f"Executando análise para sistema: {sistema}")
        
        # Preparar comando Julia
        cmd = [
            self.config["julia_path"],
            self.julia_script
        ]
        
        # Criar diretório de trabalho temporário se necessário
        work_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Executar comando Julia
            process = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=300)  # 5 minutos timeout
            
            if process.returncode == 0:
                logger.info(f"Análise concluída para {sistema}")
                logger.info(f"Execução ID: {execucao_id}")
                
                # Capturar informações da execução
                if "ANÁLISE DE CONTINGÊNCIAS CONCLUÍDA" in stdout:
                    logger.info("✅ Processo Julia finalizado com sucesso")
                else:
                    logger.warning("Processo Julia finalizado mas sem mensagem de conclusão esperada")
                    
            else:
                logger.error(f"Erro na execução Julia: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout na execução do Julia para {sistema}")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return False
            
        return True
    
    def consolidar_resultados_series(self):
        """Consolidar resultados de múltiplas execuções em banco único"""
        logger.info("Consolidando resultados das séries temporais...")
        
        # Conectar ao banco de séries
        conn_series = sqlite3.connect(self.series_db)
        
        # Listar todos os bancos de resultados
        result_files = glob.glob("resultados_opf_contingencias*.db")
        
        for db_file in result_files:
            try:
                conn_result = sqlite3.connect(db_file)
                
                # Extrair dados de execuções
                execucoes_df = pd.read_sql("SELECT * FROM execucoes", conn_result)
                
                for _, execucao in execucoes_df.iterrows():
                    id_execucao = execucao['id_execucao']
                    
                    # Verificar se execução já existe
                    existing = pd.read_sql(
                        f"SELECT 1 FROM execucoes WHERE id_execucao = '{id_execucao}'", 
                        conn_series
                    )
                    
                    if len(existing) == 0:
                        # Inserir execução
                        execucao.to_sql('execucoes', conn_series, if_exists='append', index=False)
                        
                        # Extrair e inserir dados relacionados
                        tabelas = ['contingencias', 'barras_contingencia', 'geradores_contingencia', 
                                  'linhas_contingencia', 'mvu_mvd_contingencia']
                        
                        for tabela in tabelas:
                            try:
                                df_tabela = pd.read_sql(
                                    f"SELECT * FROM {tabela} WHERE id_execucao = '{id_execucao}'", 
                                    conn_result
                                )
                                if len(df_tabela) > 0:
                                    df_tabela.to_sql(tabela, conn_series, if_exists='append', index=False)
                            except:
                                logger.warning(f"Tabela {tabela} não encontrada em {db_file}")
                
                conn_result.close()
                logger.info(f"Dados de {db_file} consolidados")
                
            except Exception as e:
                logger.error(f"Erro ao processar {db_file}: {e}")
        
        conn_series.close()
        logger.info("Consolidação de resultados concluída")
    
    def analisar_series_temporais(self):
        """Analisar séries temporais dos resultados consolidados"""
        logger.info("Iniciando análise de séries temporais...")
        
        analyzer = OPFAnalyzer(self.series_db)
        analyzer.connect()
        analyzer.load_all_data()
        
        # Gerar análises
        analises = {}
        
        # 1. Análise de cenários
        analises['cenarios'] = analyzer.get_scenario_summary()
        
        # 2. Análise de rampas
        analises['rampas'], analises['dados_evolucao'] = analisar_rampas_simples(analyzer)
        
        # 3. Análise de custos
        analises['custos'] = plot_custos_simples(analyzer)
        
        analyzer.close()
        
        return analises
    
    def gerar_relatorio_execucao(self, execucao_id, sistema):
        """Gerar relatório detalhado para uma execução específica"""
        logger.info(f"Gerando relatório para execução {execucao_id}")
        
        relatorio = {
            "id_execucao": execucao_id,
            "sistema": sistema,
            "data_execucao": datetime.now().isoformat(),
            "metricas_principais": {},
            "contingencias_analisadas": [],
            "alertas": []
        }
        
        # Conectar ao banco de resultados
        conn = sqlite3.connect(self.results_db)
        
        try:
            # Obter estatísticas da execução
            stats_query = """
            SELECT 
                COUNT(DISTINCT id_contingencia) as num_contingencias,
                AVG(total_curtailment_pu) as curtailment_medio,
                AVG(total_deficit_pu) as deficit_medio,
                AVG(custo_total_usd_h) as custo_medio
            FROM contingencias 
            WHERE id_execucao = ?
            """
            stats = pd.read_sql(stats_query, conn, params=[execucao_id]).iloc[0]
            
            relatorio["metricas_principais"] = {
                "numero_contingencias": int(stats['num_contingencias']),
                "curtailment_medio_pu": float(stats['curtailment_medio']),
                "deficit_medio_pu": float(stats['deficit_medio']),
                "custo_medio_usd_h": float(stats['custo_medio'])
            }
            
            # Identificar contingências críticas
            criticas_query = """
            SELECT id_contingencia, total_deficit_pu, total_curtailment_pu
            FROM contingencias 
            WHERE id_execucao = ? AND total_deficit_pu > 0.01
            ORDER BY total_deficit_pu DESC
            """
            criticas = pd.read_sql(criticas_query, conn, params=[execucao_id])
            
            for _, ctg in criticas.iterrows():
                relatorio["contingencias_analisadas"].append({
                    "id": ctg['id_contingencia'],
                    "deficit_pu": float(ctg['total_deficit_pu']),
                    "curtailment_pu": float(ctg['total_curtailment_pu']),
                    "status": "CRITICA"
                })
            
            # Gerar alertas
            if len(criticas) > 0:
                relatorio["alertas"].append(
                    f"⚠️ {len(criticas)} contingências com déficit de carga significativo"
                )
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            relatorio["erro"] = str(e)
        
        finally:
            conn.close()
        
        # Salvar relatório
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        relatorio_file = os.path.join(output_dir, f"relatorio_{execucao_id}.json")
        with open(relatorio_file, 'w') as f:
            json.dump(relatorio, f, indent=4)
        
        logger.info(f"Relatório salvo: {relatorio_file}")
        return relatorio
    
    def executar_fluxo_completo(self):
        """Executar fluxo completo de orquestração"""
        logger.info("Iniciando fluxo completo de orquestração")
        
        # Criar diretório de saída
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        resultados_gerais = {
            "data_inicio": datetime.now().isoformat(),
            "execucoes_realizadas": [],
            "estatisticas_gerais": {}
        }
        
        # Executar múltiplas análises
        for i in range(self.config["num_execucoes"]):
            execucao_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            
            logger.info(f"🎯 Execução {i+1}/{self.config['num_execucoes']} - ID: {execucao_id}")
            
            # Selecionar sistema (alternar entre sistemas configurados)
            sistema_idx = i % len(self.config["sistemas"])
            sistema = self.config["sistemas"][sistema_idx]
            
            # Executar análise
            sucesso = self.executar_analise_contingencias(sistema, execucao_id)
            
            if sucesso:
                # Gerar relatório individual
                relatorio = self.gerar_relatorio_execucao(execucao_id, sistema)
                resultados_gerais["execucoes_realizadas"].append(relatorio)
                
                # Aguardar entre execuções
                time.sleep(2)
            else:
                logger.error(f"Falha na execução {execucao_id}")
        
        # Consolidar resultados
        if self.config["num_execucoes"] > 1:
            self.consolidar_resultados_series()
            
            # Análise de séries temporais
            if self.config["analise_automática"]:
                analises = self.analisar_series_temporais()
                resultados_gerais["analise_series"] = {
                    "numero_cenarios": len(analises.get('dados_evolucao', [])),
                    "rampas_analisadas": len(analises.get('rampas', []))
                }
        
        resultados_gerais["data_fim"] = datetime.now().isoformat()
        resultados_gerais["estatisticas_gerais"] = {
            "total_execucoes": len(resultados_gerais["execucoes_realizadas"]),
            "execucoes_sucesso": len([e for e in resultados_gerais["execucoes_realizadas"] if "erro" not in e])
        }
        
        # Salvar resultados gerais
        resultados_file = os.path.join(self.config["output_dir"], "resultados_gerais.json")
        with open(resultados_file, 'w') as f:
            json.dump(resultados_gerais, f, indent=4)
        
        logger.info(f"Fluxo completo concluído. Resultados em: {resultados_file}")
        return resultados_gerais

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
            
        tables = ['execucoes', 'contingencias', 'barras_contingencia', 
                 'geradores_contingencia', 'linhas_contingencia', 'mvu_mvd_contingencia']
        
        for table in tables:
            try:
                query = f"SELECT * FROM {table}"
                self.dfs[table] = pd.read_sql_query(query, self.conn)
                logger.info(f"Tabela {table} carregada: {len(self.dfs[table])} registros")
            except Exception as e:
                logger.warning(f"Tabela {table} não encontrada: {e}")
                self.dfs[table] = pd.DataFrame()
        
        return self.dfs
    
    def get_scenario_summary(self):
        """Resumo geral dos cenários"""
        if 'contingencias' not in self.dfs:
            self.load_all_data()
            
        df = self.dfs['contingencias']
        print(f"Total de contingências analisadas: {len(df)}")
        
        if len(df) > 0:
            stats = df[['total_geracao_pu', 'total_carga_pu', 'custo_total_usd_h']].describe()
            print(stats.round(4))
            
        return df
    
    def close(self):
        """Fechar conexão"""
        if self.conn:
            self.conn.close()

def analisar_rampas_simples(analyzer):
    """Análise simplificada das rampas"""
    contingencias_df = analyzer.dfs.get('contingencias', pd.DataFrame())
    geradores_df = analyzer.dfs.get('geradores_contingencia', pd.DataFrame())
    
    if len(contingencias_df) < 2:
        print("Mínimo 2 contingências necessárias")
        return pd.DataFrame(), pd.DataFrame()
    
    # Coletar dados por contingência
    dados = []
    for _, ctg in contingencias_df.iterrows():
        ctg_id = ctg['id_contingencia']
        exec_id = ctg['id_execucao']
        
        geradores_ctg = geradores_df[
            (geradores_df['id_contingencia'] == ctg_id) & 
            (geradores_df['id_execucao'] == exec_id)
        ]
        
        geracao_gwd = geradores_ctg[geradores_ctg['tipo'] == 'GWD']['geracao_pu'].sum()
        curtailment = geradores_ctg[geradores_ctg['tipo'] == 'CUR']['geracao_pu'].sum()
        demanda = ctg.get('total_carga_pu', 0)
        
        dados.append({
            'contingencia': ctg_id,
            'execucao': exec_id,
            'geracao_gwd': geracao_gwd,
            'curtailment': curtailment,
            'demanda': demanda,
            'geracao_liquida': geracao_gwd - curtailment
        })
    
    df_dados = pd.DataFrame(dados)
    
    # Calcular rampas entre execuções consecutivas
    rampas = []
    execucoes_unicas = df_dados['execucao'].unique()
    
    for i in range(1, len(execucoes_unicas)):
        exec_anterior = execucoes_unicas[i-1]
        exec_atual = execucoes_unicas[i]
        
        dados_anterior = df_dados[df_dados['execucao'] == exec_anterior].iloc[0]
        dados_atual = df_dados[df_dados['execucao'] == exec_atual].iloc[0]
        
        delta_geracao = dados_atual['geracao_gwd'] - dados_anterior['geracao_gwd']
        delta_curtailment = dados_atual['curtailment'] - dados_anterior['curtailment']
        delta_demanda = dados_atual['demanda'] - dados_anterior['demanda']
        
        rampas.append({
            'transicao': f"{i}→{i+1}",
            'rampa_geracao': abs(delta_geracao),
            'rampa_curtailment': abs(delta_curtailment),
            'rampa_demanda': abs(delta_demanda),
            'variacao_geracao': delta_geracao,
            'variacao_curtailment': delta_curtailment,
            'variacao_demanda': delta_demanda,
            'execucao_anterior': exec_anterior,
            'execucao_atual': exec_atual
        })
    
    df_rampas = pd.DataFrame(rampas)
    
    # Exportar para Excel
    if len(df_rampas) > 0:
        df_rampas.to_excel('analise_rampas_detalhada.xlsx', index=False)
        logger.info("Análise de rampas exportada para analise_rampas_detalhada.xlsx")
    
    print("Rampas da Geração Eólica, Curtailment e Demanda:")
    print(df_rampas.round(4))
    
    return df_rampas, df_dados

def plot_custos_simples(analyzer):
    """Gráfico simples de evolução de custos"""
    contingencias_df = analyzer.dfs.get('contingencias', pd.DataFrame())
    
    if len(contingencias_df) == 0:
        return pd.DataFrame()
    
    # Agrupar por execução
    custos_por_execucao = contingencias_df.groupby('id_execucao')['custo_total_usd_h'].mean().reset_index()
    
    fig = px.line(custos_por_execucao, x='id_execucao', y='custo_total_usd_h',
                 title='Evolução do Custo Total por Execução',
                 markers=True)
    fig.show()
    
    return custos_por_execucao

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Sistema de Orquestração para Análise de Contingências')
    parser.add_argument('--config', default='config_orquestracao.json', help='Arquivo de configuração')
    parser.add_argument('--execucoes', type=int, help='Número de execuções')
    parser.add_argument('--sistema', help='Sistema específico para análise')
    parser.add_argument('--apenas-analise', action='store_true', help='Apenas analisar resultados existentes')
    
    args = parser.parse_args()
    
    # Inicializar orquestrador
    orchestrator = OPFOrchestrator(args.config)
    
    # Sobrescrever configurações se fornecidas via argumentos
    if args.execucoes:
        orchestrator.config["num_execucoes"] = args.execucoes
    if args.sistema:
        orchestrator.config["sistemas"] = [args.sistema]
    
    if args.apenas_analise:
        # Apenas análise de resultados existentes
        logger.info("Executando apenas análise de resultados existentes")
        analises = orchestrator.analisar_series_temporais()
        logger.info("Análise concluída")
    else:
        # Executar fluxo completo
        resultados = orchestrator.executar_fluxo_completo()
        logger.info(f"Orquestração concluída: {resultados['estatisticas_gerais']}")

if __name__ == "__main__":
    main()