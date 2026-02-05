import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

class OPF_DBHandler:
    """Gerenciador de banco de dados SQL para resultados do OPF"""
    
    def __init__(self, db_path: str = 'DATA/SMA/resultados_opf.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Conecta ao banco de dados"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def disconnect(self):
        """Desconecta do banco de dados"""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Cria tabelas necessárias"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Tabela principal
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_opf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tipo_snapshot TEXT,
            sucesso INTEGER,
            custo_total REAL,
            deficit_total REAL,
            curtailment_total REAL,
            perdas_total REAL,
            carga_total REAL,
            eolica_disponivel REAL,
            eolica_utilizada REAL,
            fator_vento REAL,
            iteracoes INTEGER,
            tempo_execucao REAL,
            solver TEXT,
            sistema_base TEXT,
            pg_json TEXT,
            ang_json TEXT,
            fluxos_json TEXT,
            mensagem TEXT
        )
        ''')
        
        # Tabela de detalhes por barra
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detalhes_barras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resultado_id INTEGER,
            barra_id INTEGER,
            tipo_barra TEXT,
            theta_deg REAL,
            carga_p_mw REAL,
            carga_q_mvar REAL,
            geracao_mw REAL,
            deficit_mw REAL,
            cmo REAL,
            FOREIGN KEY (resultado_id) REFERENCES resultados_opf(id)
        )
        ''')
        
        # Tabela de detalhes por gerador
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detalhes_geradores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resultado_id INTEGER,
            gerador_id INTEGER,
            barra_id INTEGER,
            tipo_gerador TEXT,
            p_gerada_mw REAL,
            p_max_mw REAL,
            p_min_mw REAL,
            custo REAL,
            curtailment_mw REAL,
            FOREIGN KEY (resultado_id) REFERENCES resultados_opf(id)
        )
        ''')
        
        # Tabela de detalhes por linha
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detalhes_linhas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resultado_id INTEGER,
            linha_id INTEGER,
            de_barra INTEGER,
            para_barra INTEGER,
            fluxo_mw REAL,
            limite_mw REAL,
            carregamento_percent REAL,
            perdas_mw REAL,
            FOREIGN KEY (resultado_id) REFERENCES resultados_opf(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Tabelas criadas/verificadas")
    
    def save_hourly_result(self, resultado, sistema, hora: int,
                          perfil_carga: float, perfil_eolica: float,
                          solver_name: str = 'glpk') -> int:
        """Salva resultado no banco de dados"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Data/hora atual
        timestamp = datetime.now().isoformat()
        
        # Calcular métricas
        carga_total = np.sum(sistema.PLOAD) * sistema.SB
        capacidade_eolica = sum(sistema.PGMAX_EFETIVO[g] for g in sistema.BAR_GWD) * sistema.SB
        
        # Calcular eólica utilizada
        eolica_utilizada = 0
        for g_idx in sistema.BAR_GWD:
            if g_idx < len(resultado.PG):
                eolica_utilizada += resultado.PG[g_idx] * sistema.SB
        
        # Inserir resultado principal
        cursor.execute('''
        INSERT INTO resultados_opf 
        (timestamp, tipo_snapshot, sucesso, custo_total, deficit_total, 
         curtailment_total, perdas_total, carga_total, eolica_disponivel, 
         eolica_utilizada, fator_vento, iteracoes, tempo_execucao, 
         solver, sistema_base, pg_json, ang_json, fluxos_json, mensagem)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            f"Hora_{hora:02d}",
            int(resultado.sucesso),
            float(resultado.custo_total),
            float(sum(resultado.DEFICIT) * sistema.SB),
            float(sum(resultado.CURTAILMENT) * sistema.SB),
            float(resultado.perdas * sistema.SB),
            float(carga_total),
            float(capacidade_eolica),
            float(eolica_utilizada),
            float(perfil_eolica),
            int(resultado.iteracoes),
            float(getattr(resultado, 'tempo_execucao', 0.0)),
            solver_name,
            sistema.json_file_path if hasattr(sistema, 'json_file_path') else 'unknown',
            json.dumps([float(x) for x in resultado.PG]),
            json.dumps([float(x) for x in resultado.ANG]),
            json.dumps([float(x) for x in resultado.FLUXO]),
            resultado.mensagem
        ))
        
        resultado_id = cursor.lastrowid
        
        # Salvar detalhes por barra
        for i in range(sistema.NBAR):
            barra_id = sistema.indice_para_barra[i]
            
            # Encontrar tipo da barra
            tipo_barra = "PQ"
            for b in sistema.barras:
                if b["ID_Barra"] == barra_id:
                    tipo_barra = b["tipo"]
                    break
            
            # Calcular geração na barra
            geracao_barra = 0
            for g_idx, barra_idx in enumerate(sistema.BARPG):
                if barra_idx == i and g_idx < len(resultado.PG):
                    geracao_barra += resultado.PG[g_idx] * sistema.SB
            
            cursor.execute('''
            INSERT INTO detalhes_barras 
            (resultado_id, barra_id, tipo_barra, theta_deg, 
             carga_p_mw, carga_q_mvar, geracao_mw, deficit_mw, cmo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                resultado_id,
                barra_id,
                tipo_barra,
                float(resultado.ANG[i] * 180 / np.pi) if i < len(resultado.ANG) else 0,
                float(sistema.PLOAD[i] * sistema.SB),
                float(sistema.QLOAD[i] * sistema.SB) if i < len(sistema.QLOAD) else 0,
                float(geracao_barra),
                float(resultado.DEFICIT[i] * sistema.SB) if i < len(resultado.DEFICIT) else 0,
                float(resultado.cmo_total)  # Simplificado - mesmo CMO para todas as barras em DC
            ))
        
        # Salvar detalhes por gerador
        for g_idx in range(len(resultado.PG)):
            if g_idx < len(sistema.BARPG):
                barra_idx = sistema.BARPG[g_idx]
                barra_id = sistema.indice_para_barra[barra_idx]
                
                # Tipo do gerador
                tipo_gerador = sistema.GER_TIPOS_COMBINADO[g_idx] if g_idx < len(sistema.GER_TIPOS_COMBINADO) else "UNKNOWN"
                
                # Curtailment (se for GWD)
                curtailment_mw = 0
                if g_idx in sistema.BAR_GWD:
                    idx_in_gwd = sistema.BAR_GWD.index(g_idx)
                    if idx_in_gwd < len(resultado.CURTAILMENT):
                        curtailment_mw = resultado.CURTAILMENT[idx_in_gwd] * sistema.SB
                
                cursor.execute('''
                INSERT INTO detalhes_geradores 
                (resultado_id, gerador_id, barra_id, tipo_gerador, 
                 p_gerada_mw, p_max_mw, p_min_mw, custo, curtailment_mw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    resultado_id,
                    g_idx,
                    barra_id,
                    tipo_gerador,
                    float(resultado.PG[g_idx] * sistema.SB),
                    float(sistema.PGMAX[g_idx] * sistema.SB) if g_idx < len(sistema.PGMAX) else 0,
                    float(sistema.PGMIN[g_idx] * sistema.SB) if g_idx < len(sistema.PGMIN) else 0,
                    float(sistema.CPG[g_idx]) if g_idx < len(sistema.CPG) else 0,
                    float(curtailment_mw)
                ))
        
        # Salvar detalhes por linha
        for e_idx in range(len(resultado.FLUXO)):
            if e_idx < sistema.NLIN:
                de_barra = sistema.indice_para_barra[sistema.line_fr[e_idx]]
                para_barra = sistema.indice_para_barra[sistema.line_to[e_idx]]
                
                fluxo_mw = resultado.FLUXO[e_idx] * sistema.SB
                limite_mw = sistema.FLIM[e_idx] * sistema.SB
                carregamento = abs(fluxo_mw / limite_mw * 100) if limite_mw > 0 else 0
                
                # Calcular perdas
                r = sistema.r_line[e_idx]
                perdas_mw = r * (resultado.FLUXO[e_idx] ** 2) * sistema.SB
                
                cursor.execute('''
                INSERT INTO detalhes_linhas 
                (resultado_id, linha_id, de_barra, para_barra, 
                 fluxo_mw, limite_mw, carregamento_percent, perdas_mw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    resultado_id,
                    e_idx,
                    de_barra,
                    para_barra,
                    float(fluxo_mw),
                    float(limite_mw),
                    float(carregamento),
                    float(perdas_mw)
                ))
        
        conn.commit()
        conn.close()
        
        return resultado_id
    
    def export_to_csv(self, output_path: str):
        """Exporta todos os resultados para CSV"""
        import os
        
        conn = self.connect()
        
        # Criar diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Lista de tabelas
        tabelas = ['resultados_opf', 'detalhes_barras', 
                  'detalhes_geradores', 'detalhes_linhas']
        
        for tabela in tabelas:
            df = pd.read_sql_query(f"SELECT * FROM {tabela}", conn)
            caminho_arquivo = os.path.join(output_path, f"{tabela}.csv")
            df.to_csv(caminho_arquivo, index=False)
            print(f"   ✓ {tabela}: {len(df)} registros exportados")
        
        conn.close()