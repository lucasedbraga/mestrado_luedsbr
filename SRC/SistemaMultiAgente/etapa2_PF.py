# fluxo_potencia_simplificado.py
import numpy as np
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

@dataclass
class ResultadoFluxoPotencia:
    """Estrutura para armazenar resultados do fluxo de potência simplificado"""
    sucesso: bool
    iteracoes: int
    tensoes: np.ndarray  # Tensões complexas (todas 1.0 pu)
    V_mag: np.ndarray    # Módulos das tensões (1.0 pu)
    V_ang: np.ndarray    # Ângulos das tensões (radianos)
    P_gerado: np.ndarray # Geração ativa por barra (MW)
    Q_gerado: np.ndarray # Geração reativa por barra (MVAr)
    P_carga: np.ndarray  # Carga ativa por barra (MW)
    Q_carga: np.ndarray  # Carga reativa por barra (MVAr)
    fluxos_linhas: List[Dict]
    perdas_ativas: float
    perdas_reativas: float
    hora: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# ============================================================================
# CLASSE DO FLUXO DE POTÊNCIA SIMPLIFICADO (V FIXO)
# ============================================================================

class FluxoPotenciaSimplificado:
    """Fluxo de potência com tensões fixas em 1.0 pu, calcula apenas ângulos"""
    
    def __init__(self, sistema_json_path: str):
        self.sistema_json_path = sistema_json_path
        self.carregar_dados()
        self.inicializar_variaveis()
    
    def carregar_dados(self):
        """Carrega dados do sistema do arquivo JSON"""
        with open(self.sistema_json_path, 'r') as f:
            dados = json.load(f)
        
        # Processar barras
        self.barras = dados["BARRAS"]
        self.n_barras = len(self.barras)
        self.barras_dict = {int(b["ID_Barra"]): idx for idx, b in enumerate(self.barras)}
        
        # Bases do sistema
        self.S_base = dados.get("S_base", 1.0)  # MVA
        self.V_base = dados.get("V_base", 1.0)  # kV
        
        print(f"  Sistema: {self.n_barras} barras")
        print(f"  S_base: {self.S_base} MVA, V_base: {self.V_base} kV")
        
        # Identificar tipos de barras
        self.tipo_barra = np.zeros(self.n_barras, dtype=int)  # 0=PQ, 1=PV, 2=Slack
        
        for idx, barra in enumerate(self.barras):
            barra_tipo = barra["tipo"].upper()
            barra_id = int(barra["ID_Barra"])
            
            if barra_tipo == "SLACK":
                self.barra_slack = idx
                self.tipo_barra[idx] = 2
                print(f"  Barra {barra_id}: Slack")
                
            elif barra_tipo == "PV":
                self.tipo_barra[idx] = 1
                print(f"  Barra {barra_id}: PV")
                
            else:  # PQ
                self.tipo_barra[idx] = 0
                print(f"  Barra {barra_id}: PQ")
        
        # Processar linhas
        self.linhas = dados["LINHAS"]
        self.n_linhas = len(self.linhas)
        print(f"  {self.n_linhas} linhas")
        
        # Processar demandas (cargas) - aqui carregamos apenas a ESTRUTURA da demanda
        self.demandas_json = dados.get("DEMANDAS", [])
        
        # Processar geradores (se existirem no JSON)
        self.geradores_json = dados.get("GERADORES", [])
        
        # Inicializar potências
        self.P_geracao = np.zeros(self.n_barras)  # Geração ativa (pu)
        self.Q_geracao = np.zeros(self.n_barras)  # Geração reativa (pu)
        self.P_carga = np.zeros(self.n_barras)    # Carga ativa (pu)
        self.Q_carga = np.zeros(self.n_barras)    # Carga reativa (pu)
        
        # Carregar estrutura das demandas - valores serão atualizados em cada hora
        # Primeiro, identificar quais barras têm carga
        self.barras_com_carga = []
        for demanda in self.demandas_json:
            barra_id = int(demanda["ID_Barra"])
            if barra_id in self.barras_dict:
                self.barras_com_carga.append(barra_id)
                print(f"  Barra {barra_id} tem carga (será atualizada por hora)")
        
        # Montar matriz admitância
        self.montar_matriz_admitancia()
    
    def montar_matriz_admitancia(self):
        """Monta a matriz admitância Ybus do sistema"""
        # Inicializar matriz Ybus (n_barras x n_barras) com zeros complexos
        self.Ybus = np.zeros((self.n_barras, self.n_barras), dtype=complex)
        
        # Preencher Ybus com as admitâncias das linhas
        for linha in self.linhas:
            de_idx = self.barras_dict[int(linha["ID_Barra_Origem"])]
            para_idx = self.barras_dict[int(linha["ID_Barra_Destino"])]
            
            # Extrair parâmetros da linha
            r = linha.get("R", 0.0)  # resistência (pu)
            x = linha.get("X", 0.0)  # reatância (pu)
            bsh = linha.get("Bsh", 0.0)  # susceptância shunt (pu)
            
            # Calcular admitância série
            z = complex(r, x)
            if z != 0:
                y_serie = 1.0 / z
            else:
                y_serie = 0
                
            # Admitância shunt (metade em cada extremidade)
            y_shunt = complex(0, bsh/2)
            
            # Adicionar à matriz Ybus
            self.Ybus[de_idx, para_idx] -= y_serie
            self.Ybus[para_idx, de_idx] -= y_serie
            self.Ybus[de_idx, de_idx] += y_serie + y_shunt
            self.Ybus[para_idx, para_idx] += y_serie + y_shunt
        
        print(f"  Matriz Ybus montada ({self.n_barras}x{self.n_barras})")
    
    def inicializar_variaveis(self):
        """Inicializa variáveis do fluxo de potência"""
        # Tensões fixas em 1.0 pu para todas as barras
        self.V_mag = np.ones(self.n_barras)  # Todas 1.0 pu
        self.V_ang = np.zeros(self.n_barras)  # Ângulos iniciais 0
        
        # Vetor de tensões complexas
        self.V = self.V_mag * np.exp(1j * self.V_ang)
        
        # Potência líquida injetada (será atualizada com valores reais)
        self.P_esp = self.P_geracao - self.P_carga
        self.Q_esp = -self.Q_carga.copy()  # Q_esp = -Q_carga (Q_geracao = 0 inicialmente)
    
    def atualizar_carga(self, fator_carga: float):
        """Atualiza a carga com base no fator de carga da hora"""
        print(f"  Atualizando carga com fator: {fator_carga:.3f}")
        
        # Zerar carga atual
        self.P_carga = np.zeros(self.n_barras)
        self.Q_carga = np.zeros(self.n_barras)
        
        # Aplicar fator de carga às barras com carga
        total_P_carga = 0
        for demanda in self.demandas_json:
            barra_id = int(demanda["ID_Barra"])
            if barra_id in self.barras_dict:
                idx = self.barras_dict[barra_id]
                
                # Carga base do JSON
                P_base = demanda.get("PLOAD", 0.0) / self.S_base
                Q_base = demanda.get("QLOAD_MW", 0.0) / self.S_base
                
                # Aplicar fator de carga
                self.P_carga[idx] = P_base * fator_carga
                self.Q_carga[idx] = Q_base * fator_carga
                total_P_carga += self.P_carga[idx]
                
                print(f"    Barra {barra_id}: P={self.P_carga[idx]:.3f} pu ({self.P_carga[idx]*self.S_base:.2f} MW)")
        
        print(f"    Carga total: {total_P_carga:.3f} pu ({total_P_carga*self.S_base:.2f} MW)")
        
        # Atualizar potência líquida
        self.P_esp = self.P_geracao - self.P_carga
        self.Q_esp = -self.Q_carga.copy()
        
        print(f"\n  Condições atuais:")
        for i in range(self.n_barras):
            tipo = "PQ" if self.tipo_barra[i] == 0 else ("PV" if self.tipo_barra[i] == 1 else "Slack")
            if abs(self.P_esp[i]) > 0.001 or abs(self.Q_esp[i]) > 0.001:
                print(f"    Barra {i+1} ({tipo}): V=1.000 pu, "
                      f"P_esp={self.P_esp[i]:.3f} pu, "
                      f"Q_esp={self.Q_esp[i]:.3f} pu")
    
    def atualizar_potencia_geracao(self, pg_por_barra: Dict[int, float]):
        """Atualiza a geração ativa com base nos resultados do PL"""
        print(f"  Atualizando geração ativa do PL:")
        
        # Zerar geração atual
        self.P_geracao = np.zeros(self.n_barras)
        
        # Aplicar PGs do PL às barras correspondentes
        total_geracao_pu = 0
        barras_com_geracao = []
        
        for barra_id, pg_mw in pg_por_barra.items():
            if barra_id in self.barras_dict:
                idx = self.barras_dict[barra_id]
                self.P_geracao[idx] = pg_mw / self.S_base
                total_geracao_pu += self.P_geracao[idx]
                barras_com_geracao.append(barra_id)
        
        print(f"    Barras com geração do PL: {barras_com_geracao}")
        print(f"    Geração total do PL: {total_geracao_pu:.3f} pu")
        
        # Recalcular potência líquida
        self.P_esp = self.P_geracao - self.P_carga
        
        # Mostrar balanço
        total_carga_pu = np.sum(self.P_carga)
        print(f"\n    BALANÇO DE POTÊNCIA:")
        print(f"      Carga total: {total_carga_pu:.3f} pu")
        print(f"      Geração total: {total_geracao_pu:.3f} pu")
        print(f"      Diferença (gera - carga): {total_geracao_pu - total_carga_pu:.3f} pu")
        
        if total_geracao_pu < total_carga_pu:
            print(f"      Geração insuficiente! Faltam {total_carga_pu - total_geracao_pu:.3f} pu")
        elif total_geracao_pu > total_carga_pu:
            print(f"      Geração em excesso! Sobram {total_geracao_pu - total_carga_pu:.3f} pu")
    
    def calcular_potencia_injetada(self):
        """Calcula potência injetada a partir das tensões atuais"""
        # Potência complexa injetada: S = V * conj(I) = V * conj(Y * V)
        S_calc = self.V * np.conj(self.Ybus @ self.V)
        return S_calc.real, S_calc.imag
    
    def calcular_jacobiana_aproximada(self):
        """Calcula a Jacobiana aproximada para V=1.0 pu"""
        n = self.n_barras
        # Número de barras não-slack
        n_nao_slack = n - 1
        
        # Inicializar Jacobiana (apenas dP/dθ)
        J = np.zeros((n_nao_slack, n_nao_slack))
        
        # Mapear índices
        idx_map = {}
        counter = 0
        for i in range(n):
            if i != self.barra_slack:
                idx_map[i] = counter
                counter += 1
        
        # Preencher Jacobiana
        for i in range(n):
            if i == self.barra_slack:
                continue
            row = idx_map[i]
            
            for j in range(n):
                if j == self.barra_slack:
                    continue
                col = idx_map[j]
                
                if i == j:
                    # Elemento diagonal
                    sum_val = 0
                    for k in range(n):
                        if k != i:
                            theta_diff = self.V_ang[i] - self.V_ang[k]
                            sum_val += (self.Ybus[i, k].real * np.sin(theta_diff) - 
                                       self.Ybus[i, k].imag * np.cos(theta_diff))
                    J[row, col] = sum_val
                else:
                    # Elemento fora da diagonal
                    theta_diff = self.V_ang[i] - self.V_ang[j]
                    J[row, col] = -(self.Ybus[i, j].real * np.sin(theta_diff) - 
                                   self.Ybus[i, j].imag * np.cos(theta_diff))
        
        return J
    
    def executar_fluxo_linearizado(self):
        """Executa fluxo de potência linearizado (DC Power Flow)"""
        print(f"  Executando fluxo de potência linearizado (DC Power Flow)")
        
        # Para o fluxo DC, consideramos apenas a parte imaginária de Ybus (susceptância)
        n = self.n_barras
        B_prime = np.zeros((n-1, n-1))
        
        # Mapear índices
        idx_map = {}
        counter = 0
        for i in range(n):
            if i != self.barra_slack:
                idx_map[i] = counter
                counter += 1
        
        # Preencher B'
        for i in range(n):
            if i == self.barra_slack:
                continue
            row = idx_map[i]
            
            for j in range(n):
                if j == self.barra_slack:
                    continue
                col = idx_map[j]
                
                if i == j:
                    # Diagonal: soma das susceptâncias incidentes
                    B_prime[row, col] = -np.sum([self.Ybus[i, k].imag for k in range(n) if k != i])
                else:
                    # Fora da diagonal: susceptância da linha
                    B_prime[row, col] = self.Ybus[i, j].imag
        
        # Vetor de injeções de potência ativa (excluindo slack)
        P_vec = np.zeros(n-1)
        for i in range(n):
            if i != self.barra_slack:
                P_vec[idx_map[i]] = self.P_esp[i]
        
        try:
            # Resolver: B' * θ = P  (para θ em radianos)
            theta_vec = np.linalg.solve(B_prime, P_vec)
            
            # Atualizar ângulos (slack tem ângulo 0)
            self.V_ang = np.zeros(n)
            for i in range(n):
                if i != self.barra_slack:
                    self.V_ang[i] = theta_vec[idx_map[i]]
            
            # Atualizar tensões complexas
            self.V = self.V_mag * np.exp(1j * self.V_ang)
            
            return True, 1  # 1 iteração
            
        except np.linalg.LinAlgError:
            print("      Matriz B' singular. Tentando método alternativo...")
            return False, 0
    
    def executar_fluxo_newton_rapido(self, max_iter: int = 10, tol: float = 1e-4):
        """Executa fluxo de potência rápido (V fixo em 1.0 pu)"""
        print(f"  Executando fluxo de potência rápido (V=1.0 pu fixo)")
        
        for iteracao in range(max_iter):
            # Calcular potência injetada
            P_calc, Q_calc = self.calcular_potencia_injetada()
            
            # Calcular mismatches apenas para P (em barras não-slack)
            mismatches = []
            for i in range(self.n_barras):
                if self.tipo_barra[i] != 2:  # Não é slack
                    mismatches.append(self.P_esp[i] - P_calc[i])
            
            mismatches = np.array(mismatches)
            max_mismatch = np.max(np.abs(mismatches))
            
            print(f"    Iter {iteracao}: max mismatch P = {max_mismatch:.6f}")
            
            # Verificar convergência
            if max_mismatch < tol:
                print(f"    Iter {iteracao}: CONVERGIU!")
                return True, iteracao
            
            # Calcular Jacobiana aproximada
            J = self.calcular_jacobiana_aproximada()
            
            # Resolver para Δθ
            try:
                delta_theta = np.linalg.solve(J, mismatches)
            except np.linalg.LinAlgError:
                print(f"    Iter {iteracao}: Jacobiana singular. Adicionando regularização.")
                J_reg = J + np.eye(J.shape[0]) * 1e-6
                delta_theta = np.linalg.solve(J_reg, mismatches)
            
            # Atualizar ângulos (excluindo slack)
            idx = 0
            for i in range(self.n_barras):
                if self.tipo_barra[i] != 2:  # Não é slack
                    self.V_ang[i] += delta_theta[idx]
                    idx += 1
            
            # Atualizar tensões complexas (V magnitude continua 1.0)
            self.V = self.V_mag * np.exp(1j * self.V_ang)
        
        print(f"  NÃO CONVERGIU após {max_iter} iterações")
        return False, max_iter
    
    def calcular_fluxos_linhas(self):
        """Calcula os fluxos de potência nas linhas"""
        fluxos = []
        
        for linha in self.linhas:
            de_idx = self.barras_dict[int(linha["ID_Barra_Origem"])]
            para_idx = self.barras_dict[int(linha["ID_Barra_Destino"])]
            
            # Parâmetros da linha
            r = linha.get("R", 0.0)
            x = linha.get("X", 0.0)
            bsh = linha.get("Bsh", 0.0)
            
            # Calcular admitância
            z = complex(r, x)
            if z != 0:
                y_serie = 1.0 / z
            else:
                y_serie = 0
                
            y_shunt = complex(0, bsh/2)
            
            # Tensões
            V_de = self.V[de_idx]
            V_para = self.V[para_idx]
            
            # Correntes
            I_de_para = (V_de - V_para) * y_serie + V_de * y_shunt
            I_para_de = (V_para - V_de) * y_serie + V_para * y_shunt
            
            # Potências
            S_de_para = V_de * np.conj(I_de_para)
            S_para_de = V_para * np.conj(I_para_de)
            
            # Perdas
            perda_ativa = S_de_para.real + S_para_de.real
            perda_reativa = S_de_para.imag + S_para_de.imag
            
            fluxos.append({
                'id': len(fluxos) + 1,
                'de': int(linha["ID_Barra_Origem"]),
                'para': int(linha["ID_Barra_Destino"]),
                'P_de_para': S_de_para.real * self.S_base,
                'Q_de_para': S_de_para.imag * self.S_base,
                'P_para_de': S_para_de.real * self.S_base,
                'Q_para_de': S_para_de.imag * self.S_base,
                'perda_ativa': perda_ativa * self.S_base,
                'perda_reativa': perda_reativa * self.S_base,
                'limite': linha.get("LIM_Fluxo", 0.0) * self.S_base
            })
        
        return fluxos
    
    def calcular_potencia_reativa(self):
        """Calcula potência reativa injetada em cada barra"""
        # S = V * conj(Y * V)
        S_inj = self.V * np.conj(self.Ybus @ self.V)
        Q_inj = S_inj.imag * self.S_base  # Em MVAr
        
        # Para barras PQ, Q_inj = Q_gerado - Q_carga
        # Para barras PV/Slack, Q_gerado = Q_inj + Q_carga
        Q_gerado = np.zeros(self.n_barras)
        for i in range(self.n_barras):
            Q_gerado[i] = Q_inj[i] + self.Q_carga[i] * self.S_base
        
        return Q_gerado
    
    def executar_fluxo(self):
        """Executa o fluxo de potência simplificado"""
        print(f"  Executando fluxo de potência simplificado (V=1.0 pu)")
        
        # Primeiro tentar fluxo linearizado (DC)
        sucesso, iteracoes = self.executar_fluxo_linearizado()
        
        if not sucesso:
            # Se falhar, tentar Newton rápido
            sucesso, iteracoes = self.executar_fluxo_newton_rapido(max_iter=10, tol=1e-4)
        
        if not sucesso:
            print("    Não foi possível resolver o fluxo de potência")
            return self.criar_resultado_falha()
        
        # Calcular fluxos nas linhas
        fluxos = self.calcular_fluxos_linhas()
        
        # Calcular perdas
        perdas_ativas = sum(f['perda_ativa'] for f in fluxos)
        perdas_reativas = sum(f['perda_reativa'] for f in fluxos)
        
        # Calcular potência reativa gerada
        Q_gerado = self.calcular_potencia_reativa()
        
        return ResultadoFluxoPotencia(
            sucesso=True,
            iteracoes=iteracoes,
            tensoes=self.V.copy(),
            V_mag=self.V_mag.copy(),
            V_ang=self.V_ang.copy(),
            P_gerado=self.P_geracao * self.S_base,
            Q_gerado=Q_gerado,
            P_carga=self.P_carga * self.S_base,
            Q_carga=self.Q_carga * self.S_base,
            fluxos_linhas=fluxos,
            perdas_ativas=perdas_ativas,
            perdas_reativas=perdas_reativas,
            hora=0
        )
    
    def criar_resultado_falha(self):
        """Cria resultado de falha"""
        return ResultadoFluxoPotencia(
            sucesso=False,
            iteracoes=0,
            tensoes=np.ones(self.n_barras, dtype=complex),
            V_mag=np.ones(self.n_barras),
            V_ang=np.zeros(self.n_barras),
            P_gerado=np.zeros(self.n_barras),
            Q_gerado=np.zeros(self.n_barras),
            P_carga=self.P_carga * self.S_base,
            Q_carga=self.Q_carga * self.S_base,
            fluxos_linhas=[],
            perdas_ativas=0,
            perdas_reativas=0,
            hora=0
        )

# ============================================================================
# INTEGRAÇÃO COM RESULTADOS DO PLANEJAMENTO (PL)
# ============================================================================

class FluxoPotenciaComPLSimplificado:
    """Executa fluxo de potência simplificado com dados do PL"""
    
    def __init__(self, sistema_json_path: str, resultados_pl_db: str = 'DATA/SMA/resultados_PL.db'):
        self.sistema_json_path = sistema_json_path
        self.resultados_pl_db = resultados_pl_db
    
    def carregar_resultados_pl(self) -> Dict[int, Tuple[List[float], float]]:
        """Carrega resultados do PL do banco de dados, incluindo a carga total"""
        print(" Carregando resultados do planejamento por PL...")
        
        conn = sqlite3.connect(self.resultados_pl_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT timestamp FROM resultados_PL ORDER BY timestamp DESC LIMIT 1")
        ultimo_timestamp = cursor.fetchone()
        
        if not ultimo_timestamp:
            print(" Nenhum resultado de PL encontrado no banco de dados")
            conn.close()
            return {}
        
        ultimo_timestamp = ultimo_timestamp[0]
        print(f"  Última execução: {ultimo_timestamp}")
        
        # Carregar PGs e carga_total para cada hora
        cursor.execute('''
        SELECT hora, pg_json, carga_total FROM resultados_PL 
        WHERE timestamp = ? AND sucesso = 1 
        ORDER BY hora
        ''', (ultimo_timestamp,))
        
        resultados = {}
        
        for hora, pg_json, carga_total in cursor.fetchall():
            if pg_json:
                pg_valores = json.loads(pg_json)
                resultados[hora] = (pg_valores, carga_total)
        
        conn.close()
        
        print(f"Carregados {len(resultados)} horas com dados de geração e carga")
        return resultados
    
    def mapear_pg_para_barras(self, pg_lista: List[float]) -> Dict[int, float]:
        """Mapeia a lista de PGs para as barras correspondentes"""
        pg_por_barra = {}
        
        with open(self.sistema_json_path, 'r') as f:
            dados_sistema = json.load(f)
        
        geradores = dados_sistema.get("GERADORES", [])
        
        print(f"  Encontrados {len(geradores)} geradores")
        
        for idx, gerador in enumerate(geradores):
            if idx < len(pg_lista):
                barra_id = int(gerador["ID_Barra"])
                pg_valor = pg_lista[idx]
                pg_por_barra[barra_id] = pg_valor
                print(f"    Gerador {gerador['ID_Gerador']} na barra {barra_id}: PG={pg_valor:.3f} MW")
        
        return pg_por_barra
    
    def executar_analise_24h(self) -> Dict[int, ResultadoFluxoPotencia]:
        """Executa fluxo de potência simplificado para 24 horas"""
        print("\n" + "="*70)
        print("FLUXO DE POTÊNCIA SIMPLIFICADO - 24 HORAS (V=1.0 pu)")
        print("="*70)
        
        # Carregar resultados do PL (PGs e carga_total para cada hora)
        resultados_pl = self.carregar_resultados_pl()
        
        if not resultados_pl:
            print(" Nenhum dado de planejamento encontrado")
            return {}
        
        resultados = {}
        horas_sucesso = 0
        
        for hora in sorted(resultados_pl.keys()):
            print(f"\n HORA {hora:02d}:00")
            print("-"*40)
            
            try:
                # Criar novo solver para cada hora
                fluxo_solver = FluxoPotenciaSimplificado(self.sistema_json_path)
                
                # Obter PGs e carga_total para esta hora
                pg_lista, carga_total = resultados_pl[hora]
                
                print(f"  Fator de carga da hora: {carga_total:.3f}")
                
                # Primeiro, atualizar a carga com o fator de carga da hora
                fluxo_solver.atualizar_carga(carga_total)
                
                # Mapear PGs para barras
                pg_por_barra = self.mapear_pg_para_barras(pg_lista)
                
                # Atualizar geração
                fluxo_solver.atualizar_potencia_geracao(pg_por_barra)
                
                # Executar fluxo de potência
                resultado = fluxo_solver.executar_fluxo()
                resultado.hora = hora
                
                resultados[hora] = resultado
                
                if resultado.sucesso:
                    horas_sucesso += 1
                    print(f"\n FLUXO CALCULADO em {resultado.iteracoes} iteração(ões)")
                    print(f"  Perdas ativas: {resultado.perdas_ativas:.3f} MW")
                    print(f"  Perdas reativas: {resultado.perdas_reativas:.3f} MVAr")
                    
                    # Formatar ângulos para exibição
                    angulos_graus = np.degrees(resultado.V_ang)
                    angulos_formatados = [f"{a:.2f}°" for a in angulos_graus]
                    print(f"  Ângulos: {angulos_formatados}")
                    
                    # Mostrar apenas Q gerado significativo
                    q_significativo = []
                    for i, q in enumerate(resultado.Q_gerado):
                        if abs(q) > 0.01:  # Mostrar apenas se > 0.01 MVAr
                            q_significativo.append(f"Barra {i+1}: {q:.3f} MVAr")
                    if q_significativo:
                        print(f"  Q gerado (significativo): {', '.join(q_significativo)}")
                    
                    # Verificar fluxos nas linhas
                    violacoes_limite = []
                    for fluxo in resultado.fluxos_linhas:
                        limite = fluxo.get('limite', 9999)
                        if limite > 0 and abs(fluxo['P_de_para']) > limite * 1.05:
                            violacoes_limite.append(
                                f"Linha {fluxo['de']}-{fluxo['para']}: "
                                f"P={fluxo['P_de_para']:.3f} MW > limite {limite:.3f} MW"
                            )
                    
                    if violacoes_limite:
                        print(f"    Violações de limite:")
                        for violacao in violacoes_limite:
                            print(f"    {violacao}")
                else:
                    print(f"\n FALHA no cálculo do fluxo")
                    
            except Exception as e:
                print(f" Erro na hora {hora}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Estatísticas
        print(f"\n" + "="*70)
        print("RESUMO DA ANÁLISE")
        print("="*70)
        print(f"Horas processadas: {len(resultados)}")
        print(f"Horas com sucesso: {horas_sucesso}")
        
        if horas_sucesso > 0:
            self.salvar_resultados(resultados)
            
            # Calcular estatísticas
            perdas_totais = sum(r.perdas_ativas for r in resultados.values() if r.sucesso)
            perdas_reativas_totais = sum(r.perdas_reativas for r in resultados.values() if r.sucesso)
            print(f"Perdas totais ativas (24h): {perdas_totais:.3f} MW")
            print(f"Perdas totais reativas (24h): {perdas_reativas_totais:.3f} MVAr")
            
            # Mostrar exemplo da última hora
            horas_convergidas = [h for h, r in resultados.items() if r.sucesso]
            if horas_convergidas:
                ultima_hora = max(horas_convergidas)
                r = resultados[ultima_hora]
                
                print(f"\n EXEMPLO - HORA {ultima_hora:02d}:00")
                
                # Formatar ângulos
                angulos_graus = np.degrees(r.V_ang)
                angulos_formatados = [f"{a:.2f}°" for a in angulos_graus]
                print(f"Ângulos: {angulos_formatados}")
                
                print(f"Q gerado: {r.Q_gerado} MVAr")
                print(f"Perdas ativas: {r.perdas_ativas:.3f} MW")
                print(f"Perdas reativas: {r.perdas_reativas:.3f} MVAr")
                
                # Mostrar fluxos na última hora
                if r.fluxos_linhas:
                    print(f"\nFluxos nas linhas (Hora {ultima_hora:02d}:00):")
                    print("Linha | De→Para | P (MW) | Q (MVAr) | Perda P (MW)")
                    print("-"*60)
                    for f in r.fluxos_linhas:
                        print(f"  {f['id']:4d} | {f['de']:3d}→{f['para']:3d} | "
                              f"{f['P_de_para']:7.3f} | {f['Q_de_para']:7.3f} | "
                              f"{f['perda_ativa']:12.3f}")
        
        return resultados
    
    def salvar_resultados(self, resultados: Dict[int, ResultadoFluxoPotencia]):
        """Salva resultados em banco de dados"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        db_nome = f'DATA/SMA/resultados_PF.db'
        
        conn = sqlite3.connect(db_nome)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_fluxo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hora INTEGER,
            sucesso INTEGER,
            iteracoes INTEGER,
            perdas_ativas REAL,
            perdas_reativas REAL,
            tensoes_mag_json TEXT,
            tensoes_ang_json TEXT,
            P_gerado_json TEXT,
            Q_gerado_json TEXT,
            P_carga_json TEXT,
            Q_carga_json TEXT,
            fluxos_json TEXT,
            timestamp TEXT
        )
        ''')
        
        for hora, resultado in resultados.items():
            if resultado.sucesso:
                cursor.execute('''
                INSERT INTO resultados_fluxo 
                (hora, sucesso, iteracoes, perdas_ativas, perdas_reativas,
                 tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
                 P_carga_json, Q_carga_json, fluxos_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    hora, 1, resultado.iteracoes,
                    float(resultado.perdas_ativas), float(resultado.perdas_reativas),
                    json.dumps([float(v) for v in resultado.V_mag]),
                    json.dumps([float(v) for v in resultado.V_ang]),
                    json.dumps([float(v) for v in resultado.P_gerado]),
                    json.dumps([float(v) for v in resultado.Q_gerado]),
                    json.dumps([float(v) for v in resultado.P_carga]),
                    json.dumps([float(v) for v in resultado.Q_carga]),
                    json.dumps(resultado.fluxos_linhas),
                    resultado.timestamp
                ))
        
        conn.commit()
        conn.close()
        
        print(f"\n Resultados salvos em: {db_nome}")

    def plotar_resultados(self, resultados: Dict[int, ResultadoFluxoPotencia]):
        """
        Plota resultados da operação do fluxo de potência para 24 horas
        
        Parâmetros:
        resultados: Dicionário com chave=hora (0-23) e valor=ResultadoFluxoPotencia
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib não está instalado. Não é possível plotar os gráficos.")
            return
        
        if not resultados:
            print("Nenhum resultado para plotar.")
            return
        
        # Filtrar apenas horas com sucesso
        horas_convergidas = [h for h, r in resultados.items() if r.sucesso]
        
        if not horas_convergidas:
            print("Nenhum resultado com sucesso para plotar.")
            return
        
        horas_convergidas.sort()
        
        # Preparar dados para plotagem
        perdas_ativas = [resultados[h].perdas_ativas for h in horas_convergidas]
        perdas_reativas = [resultados[h].perdas_reativas for h in horas_convergidas]
        
        # Calcular totais de geração e carga por hora
        geracao_total = [np.sum(resultados[h].P_gerado) for h in horas_convergidas]
        carga_total = [np.sum(resultados[h].P_carga) for h in horas_convergidas]
        
        # Calcular ângulo médio (em graus) por hora (excluindo slack)
        angulo_medio = []
        for h in horas_convergidas:
            r = resultados[h]
            # Encontrar índice da barra slack (ângulo = 0)
            idx_slack = np.argmin(np.abs(r.V_ang))
            # Calcular média dos ângulos absolutos (excluindo slack)
            angulos_absolutos = np.abs(np.delete(r.V_ang, idx_slack))
            angulo_medio.append(np.mean(np.degrees(angulos_absolutos)))
        
        # Calcular número de linhas com violação de limite (>85% do limite)
        violacoes_limite = []
        for h in horas_convergidas:
            r = resultados[h]
            violacoes = 0
            for fluxo in r.fluxos_linhas:
                limite = fluxo.get('limite', 9999)
                if limite > 0 and abs(fluxo['P_de_para']) > limite * 0.85:
                    violacoes += 1
            violacoes_limite.append(violacoes)
        
        # Criar figura com 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Perdas ativas e reativas por hora
        ax1.plot(horas_convergidas, perdas_ativas, 'ro-', linewidth=2, markersize=8, label='Perdas Ativas (MW)')
        ax1.plot(horas_convergidas, perdas_reativas, 'bs-', linewidth=2, markersize=8, label='Perdas Reativas (MVAr)')
        ax1.set_xlabel('Hora do Dia')
        ax1.set_ylabel('Perdas (MW / MVAr)')
        ax1.set_title('Perdas Ativas e Reativas por Hora')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Gráfico 2: Geração vs Carga total
        ax2.bar(horas_convergidas, carga_total, alpha=0.6, label='Carga Total (MW)', color='orange')
        ax2.plot(horas_convergidas, geracao_total, 'go-', linewidth=2, markersize=8, label='Geração Total (MW)')
        ax2.set_xlabel('Hora do Dia')
        ax2.set_ylabel('Potência (MW)')
        ax2.set_title('Geração Total vs Carga Total')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        
        # Gráfico 3: Ângulo médio das tensões
        ax3.bar(horas_convergidas, angulo_medio, alpha=0.7, color='purple')
        ax3.set_xlabel('Hora do Dia')
        ax3.set_ylabel('Ângulo Médio (graus)')
        ax3.set_title('Ângulo Médio das Tensões (excluindo barra slack)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))
        
        # Gráfico 4: Violações de limite nas linhas
        ax4.bar(horas_convergidas, violacoes_limite, alpha=0.7, color='red')
        ax4.set_xlabel('Hora do Dia')
        ax4.set_ylabel('Número de Linhas')
        ax4.set_title(f'Linhas com Fluxo > 85% do Limite')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 2))
        
        plt.suptitle(f'Resultados do Fluxo de Potência - {len(horas_convergidas)} horas convergentes', fontsize=14)
        plt.tight_layout()
        
        # Salvar figura
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'DATA/SMA/resultados_fluxo_potencia_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        print(f"\nGrágico salvo como: {filename}")
        plt.show()
        
        # Adicional: Plotar tensões para uma hora específica (última hora convergida)
        if horas_convergidas:
            ultima_hora = horas_convergidas[-1]
            r = resultados[ultima_hora]
            
            fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Gráfico de barras das tensões em pu (todas são 1.0 pu no simplificado)
            barras_ids = list(range(1, len(r.V_mag) + 1))
            ax5.bar(barras_ids, r.V_mag, alpha=0.7, color='green')
            ax5.set_xlabel('Barra')
            ax5.set_ylabel('Tensão (pu)')
            ax5.set_title(f'Tensões nas Barras - Hora {ultima_hora:02d}:00')
            ax5.set_xticks(barras_ids)
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1.0 pu')
            
            # Gráfico de ângulos
            angulos_graus = np.degrees(r.V_ang)
            ax6.bar(barras_ids, angulos_graus, alpha=0.7, color='blue')
            ax6.set_xlabel('Barra')
            ax6.set_ylabel('Ângulo (graus)')
            ax6.set_title(f'Ângulos das Tensões - Hora {ultima_hora:02d}:00')
            ax6.set_xticks(barras_ids)
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f'DATA/SMA/detalhes_barras_hora_{ultima_hora:02d}.png', dpi=150)
            plt.show()

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Função principal"""
    import sys
    import os
    
    print("="*70)
    print("FLUXO DE POTÊNCIA SIMPLIFICADO (V=1.0 pu FIXO)")
    print("="*70)
    
    sistema_json = "DATA/input/3barras_BASE.json"
    resultados_pl_db = 'DATA/SMA/resultados_PL.db'
    
    print(f"Sistema elétrico: {sistema_json}")
    print(f"Banco de dados PL: {resultados_pl_db}")
    print("="*70)
    
    try:
        if not os.path.exists(sistema_json):
            print(f" Arquivo do sistema não encontrado: {sistema_json}")
            sys.exit(1)
        
        if not os.path.exists(resultados_pl_db):
            print(f" Banco de dados do PL não encontrado: {resultados_pl_db}")
            print("Execute primeiro a etapa de planejamento (PL).")
            sys.exit(1)
        
        # Executar análise de 24 horas
        analisador = FluxoPotenciaComPLSimplificado(sistema_json, resultados_pl_db)
        resultados = analisador.executar_analise_24h()
        if resultados:
            analisador.plotar_resultados(resultados)
        
    except Exception as e:
        print(f" Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Execução concluída.")
    print("="*70)

if __name__ == "__main__":
    main()