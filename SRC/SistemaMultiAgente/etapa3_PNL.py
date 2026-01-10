import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import math

class ResultadoOPF:
    """Classe para armazenar resultados do OPF"""
    def __init__(self, sucesso: bool, custo_total: float, iteracoes: int = 0):
        self.sucesso = sucesso
        self.custo_total = custo_total
        self.iteracoes = iteracoes
        self.V_mag = []
        self.V_ang = []
        self.P_gerado = []
        self.Q_gerado = []
        self.P_carga = []
        self.Q_carga = []
        self.fluxos_linhas = {}
        self.perdas_ativas = 0.0
        self.perdas_reativas = 0.0
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class OPFNaoLinear:
    """Sistema de Fluxo de Potência Ótimo Não Linear"""
    
    def __init__(self, dados_rede: dict):
        """
        Inicializa o sistema OPF
        
        Args:
            dados_rede: Dicionário com dados da rede
        """
        self.dados = dados_rede
        self.n_barras = len(dados_rede['BARRAS'])
        self.n_linhas = len(dados_rede['LINHAS'])
        self.n_geradores = len(dados_rede['GERADORES'])
        
        # Mapeamento de índices
        self.barra_para_indice = {barra['ID_Barra']: i 
                                 for i, barra in enumerate(dados_rede['BARRAS'])}
        self.indice_para_barra = {i: barra['ID_Barra'] 
                                 for i, barra in enumerate(dados_rede['BARRAS'])}
        
        # Configurações do solver
        self.tolerancia = 1e-6
        self.iter_max = 100
        
        # Construir matriz de admitância uma vez
        self.G, self.B = self.construir_matriz_admitancia()
        
    def carregar_resultados_pf(self, db_path: str, hora: int) -> dict:
        """
        Carrega resultados do Fluxo de Potência da etapa anterior
        
        Args:
            db_path: Caminho para o banco de dados
            hora: Hora específica para carregar
            
        Returns:
            Dicionário com resultados do PF
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
               P_carga_json, Q_carga_json, fluxos_json
        FROM resultados_fluxo 
        WHERE hora = ?
        ''', (hora,))
        
        resultado = cursor.fetchone()
        conn.close()
        
        if not resultado:
            raise ValueError(f"Resultados para hora {hora} não encontrados")
        
        # Converter JSON strings de volta para listas
        resultados = {
            'V_mag': json.loads(resultado[0]),
            'V_ang': json.loads(resultado[1]),
            'P_ger': json.loads(resultado[2]),
            'Q_ger': json.loads(resultado[3]),
            'P_carga': json.loads(resultado[4]),
            'Q_carga': json.loads(resultado[5]),
            'fluxos': json.loads(resultado[6])
        }
        
        return resultados
    
    def construir_matriz_admitancia(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói matriz de admitância nodal Y = G + jB
        
        Returns:
            Tuple contendo matrizes de condutância (G) e susceptância (B)
        """
        G = np.zeros((self.n_barras, self.n_barras), dtype=float)
        B = np.zeros((self.n_barras, self.n_barras), dtype=float)
        
        # Adicionar admitâncias das linhas
        for linha in self.dados['LINHAS']:
            i = self.barra_para_indice[linha['ID_Barra_Origem']]
            j = self.barra_para_indice[linha['ID_Barra_Destino']]
            
            # Admitância série
            r = linha['R']
            x = linha['X']
            if r == 0 and x == 0:
                continue
                
            denom = r**2 + x**2
            g_serie = r / denom  # Condutância série
            b_serie = -x / denom  # Susceptância série (negativa para indutância)
            
            # Admitância shunt (metade de cada lado)
            b_shunt = linha.get('Bsh', 0.0) / 2.0
            
            # Elementos fora da diagonal
            G[i, j] -= g_serie
            G[j, i] -= g_serie
            B[i, j] -= b_serie
            B[j, i] -= b_serie
            
            # Elementos da diagonal
            G[i, i] += g_serie
            G[j, j] += g_serie
            B[i, i] += b_serie + b_shunt
            B[j, j] += b_serie + b_shunt
        
        return G, B
    
    def calcular_fluxo_linha(self, V_mag: list, V_ang: list, 
                           linha_idx: int) -> Tuple[float, float]:
        """
        Calcula fluxo de potência em uma linha
        
        Args:
            V_mag: Lista de magnitudes de tensão
            V_ang: Lista de ângulos de tensão
            linha_idx: Índice da linha
            
        Returns:
            Tuple (P_ij, Q_ij) - fluxo da barra i para j
        """
        linha = self.dados['LINHAS'][linha_idx]
        i = self.barra_para_indice[linha['ID_Barra_Origem']]
        j = self.barra_para_indice[linha['ID_Barra_Destino']]
        
        # Parâmetros da linha
        r = linha['R']
        x = linha['X']
        b_sh = linha.get('Bsh', 0.0) / 2.0
        
        # Diferença de ângulo
        theta_ij = V_ang[i] - V_ang[j]
        
        # Condutância e susceptância série
        denom = r**2 + x**2
        g = r / denom
        b = -x / denom
        
        # Fluxo de potência ativa da barra i para j
        P_ij = (V_mag[i]**2) * g - V_mag[i] * V_mag[j] * (
            g * np.cos(theta_ij) + b * np.sin(theta_ij)
        )
        
        # Fluxo de potência reativa da barra i para j
        Q_ij = -(V_mag[i]**2) * (b + b_sh) - V_mag[i] * V_mag[j] * (
            g * np.sin(theta_ij) - b * np.cos(theta_ij)
        )
        
        return P_ij, Q_ij
    
    def resolver_opf_pyomo(self, resultados_pf: dict) -> ResultadoOPF:
        """
        Resolve o problema de OPF usando Pyomo
        
        Args:
            resultados_pf: Resultados do fluxo de potência da etapa anterior
            
        Returns:
            ResultadoOPF com solução ótima
        """
        # Criar modelo
        model = pyo.ConcreteModel()
        
        # Conjuntos
        model.BARRAS = pyo.RangeSet(0, self.n_barras - 1)
        model.LINHAS = pyo.RangeSet(0, self.n_linhas - 1)
        model.GERADORES = pyo.RangeSet(0, self.n_geradores - 1)
        
        # Variáveis de decisão
        # Tensões
        model.V = pyo.Var(model.BARRAS, bounds=(0.9, 1.1))
        model.theta = pyo.Var(model.BARRAS, bounds=(-math.pi, math.pi))
        
        # Gerações
        model.Pg = pyo.Var(model.GERADORES)
        model.Qg = pyo.Var(model.GERADORES)
        
        # Encontrar barra slack
        slack_idx = None
        slack_barra_id = None
        for i, barra in enumerate(self.dados['BARRAS']):
            if barra['tipo'] == 'Slack':
                slack_idx = i
                slack_barra_id = barra['ID_Barra']
                break
        
        if slack_idx is not None:
            # Fixar ângulo da barra slack em 0
            model.theta[slack_idx].fix(0.0)
            
            # Fixar tensão da barra slack no valor do PF (ou 1.0 se não disponível)
            V_slack = resultados_pf['V_mag'][slack_idx] if slack_idx < len(resultados_pf['V_mag']) else 1.0
            model.V[slack_idx].fix(V_slack)
        
        # Função objetivo: Minimizar desvio quadrático em relação ao PF
        def objetivo_rule(model):
            custo = 0.0
            
            # Penalidade para desvios de geração
            for idx, ger in enumerate(self.dados['GERADORES']):
                barra_idx = self.barra_para_indice[ger['ID_Barra']]
                Pg_ref = resultados_pf['P_ger'][barra_idx]
                Qg_ref = resultados_pf['Q_ger'][barra_idx]
                
                custo += (model.Pg[idx] - Pg_ref)**2
                custo += 0.1 * (model.Qg[idx] - Qg_ref)**2  # Peso menor para Q
            
            # Penalidade para desvios de tensão (exceto slack)
            for i in model.BARRAS:
                if i != slack_idx:  # Não penalizar slack (já está fixa)
                    V_ref = resultados_pf['V_mag'][i]
                    custo += 10.0 * (model.V[i] - V_ref)**2  # Peso maior para tensões
            
            return custo
        
        model.objetivo = pyo.Objective(rule=objetivo_rule, sense=pyo.minimize)
        
        # Restrições de balanço de potência
        def potencia_ativa_rule(model, i):
            if i == slack_idx:
                return pyo.Constraint.Skip  # Slack não tem equação de P
            
            P_ger_total = 0.0
            P_carga_total = 0.0
            
            # Soma gerações na barra i
            for idx, ger in enumerate(self.dados['GERADORES']):
                if self.barra_para_indice[ger['ID_Barra']] == i:
                    P_ger_total += model.Pg[idx]
            
            # Soma cargas na barra i
            for barra in self.dados['BARRAS']:
                if self.barra_para_indice[barra['ID_Barra']] == i:
                    P_carga_total += barra.get('P_carga_MW', 0.0)
            
            # Potência injetada calculada
            P_inj = 0.0
            for j in model.BARRAS:
                theta_ij = model.theta[i] - model.theta[j]
                P_inj += model.V[i] * model.V[j] * (
                    self.G[i, j] * pyo.cos(theta_ij) +
                    self.B[i, j] * pyo.sin(theta_ij)
                )
            
            return P_ger_total - P_carga_total == P_inj
        
        def potencia_reativa_rule(model, i):
            if i == slack_idx:
                return pyo.Constraint.Skip  # Slack não tem equação de Q
            
            Q_ger_total = 0.0
            Q_carga_total = 0.0
            
            # Soma gerações reativas na barra i
            for idx, ger in enumerate(self.dados['GERADORES']):
                if self.barra_para_indice[ger['ID_Barra']] == i:
                    Q_ger_total += model.Qg[idx]
            
            # Soma cargas reativas na barra i
            for barra in self.dados['BARRAS']:
                if self.barra_para_indice[barra['ID_Barra']] == i:
                    Q_carga_total += barra.get('Q_carga_MW', 0.0)
            
            # Potência reativa injetada calculada
            Q_inj = 0.0
            for j in model.BARRAS:
                theta_ij = model.theta[i] - model.theta[j]
                Q_inj += model.V[i] * model.V[j] * (
                    self.G[i, j] * pyo.sin(theta_ij) -
                    self.B[i, j] * pyo.cos(theta_ij)
                )
            
            return Q_ger_total - Q_carga_total == Q_inj
        
        model.balanco_P = pyo.Constraint(model.BARRAS, rule=potencia_ativa_rule)
        model.balanco_Q = pyo.Constraint(model.BARRAS, rule=potencia_reativa_rule)
        
        # Restrições de limites dos geradores
        def limites_geradores_rule(model, idx):
            ger = self.dados['GERADORES'][idx]
            return (ger['PGERmin_MW'], model.Pg[idx], ger['PGERmax_MW'])
        
        def limites_reativo_rule(model, idx):
            ger = self.dados['GERADORES'][idx]
            return (ger['Qmin_MW'], model.Qg[idx], ger['Qmax_MW'])
        
        model.limites_Pg = pyo.Constraint(model.GERADORES, rule=limites_geradores_rule)
        model.limites_Qg = pyo.Constraint(model.GERADORES, rule=limites_reativo_rule)
        
        # Restrições de fluxo nas linhas (apenas magnitude do fluxo ativo)
        def fluxo_linhas_rule(model, linha_idx):
            linha = self.dados['LINHAS'][linha_idx]
            i = self.barra_para_indice[linha['ID_Barra_Origem']]
            j = self.barra_para_indice[linha['ID_Barra_Destino']]
            
            # Cálculo do fluxo de potência ativa
            theta_ij = model.theta[i] - model.theta[j]
            
            # Fluxo ativo da barra i para j (simplificado)
            P_ij = model.V[i] * model.V[j] * (
                self.G[i, j] * pyo.cos(theta_ij) +
                self.B[i, j] * pyo.sin(theta_ij)
            ) - self.G[i, j] * model.V[i]**2
            
            # Limite de fluxo
            limite = linha.get('LIM_Fluxo', 2.0)
            
            return pyo.inequality(-limite, P_ij, limite)
        
        model.fluxo_linhas = pyo.Constraint(model.LINHAS, rule=fluxo_linhas_rule)
        
        # Valores iniciais baseados no PF
        for i in model.BARRAS:
            if i != slack_idx:  # Não inicializar slack (já está fixa)
                if i < len(resultados_pf['V_mag']):
                    model.V[i] = resultados_pf['V_mag'][i]
                if i < len(resultados_pf['V_ang']):
                    model.theta[i] = resultados_pf['V_ang'][i]
        
        for idx, ger in enumerate(self.dados['GERADORES']):
            barra_idx = self.barra_para_indice[ger['ID_Barra']]
            if barra_idx < len(resultados_pf['P_ger']):
                model.Pg[idx] = resultados_pf['P_ger'][barra_idx]
            if barra_idx < len(resultados_pf['Q_ger']):
                model.Qg[idx] = resultados_pf['Q_ger'][barra_idx]
        
        # Resolver o problema
        solver = pyo.SolverFactory('couenne', executable='../../../scripts/couenne/couenne')
        solver.options['tol'] = self.tolerancia
        solver.options['max_iter'] = self.iter_max
        solver.options['print_level'] = 0  # Reduzir output do IPOPT
        
        try:
            results = solver.solve(model, tee=False)
            
            # Verificar se a solução é ótima
            sucesso = str(results.solver.termination_condition) in ['optimal', 'locallyOptimal']
            
            resultado = ResultadoOPF(
                sucesso=sucesso,
                custo_total=pyo.value(model.objetivo) if sucesso else 0.0,
                iteracoes=results.solver.iterations if hasattr(results.solver, 'iterations') else 0
            )
            
            if sucesso:
                # Preencher resultados
                resultado.V_mag = [pyo.value(model.V[i]) for i in model.BARRAS]
                resultado.V_ang = [pyo.value(model.theta[i]) for i in model.BARRAS]
                
                # Mapear gerações para barras
                resultado.P_gerado = [0.0] * self.n_barras
                resultado.Q_gerado = [0.0] * self.n_barras
                
                for idx, ger in enumerate(self.dados['GERADORES']):
                    barra_idx = self.barra_para_indice[ger['ID_Barra']]
                    resultado.P_gerado[barra_idx] = pyo.value(model.Pg[idx])
                    resultado.Q_gerado[barra_idx] = pyo.value(model.Qg[idx])
                
                # Carregar dados de carga
                for barra in self.dados['BARRAS']:
                    idx = self.barra_para_indice[barra['ID_Barra']]
                    resultado.P_carga.append(barra.get('P_carga_MW', 0.0))
                    resultado.Q_carga.append(barra.get('Q_carga_MW', 0.0))
                
                # Calcular fluxos nas linhas e perdas
                resultado.perdas_ativas = 0.0
                resultado.perdas_reativas = 0.0
                
                for linha_idx in range(self.n_linhas):
                    linha = self.dados['LINHAS'][linha_idx]
                    i = self.barra_para_indice[linha['ID_Barra_Origem']]
                    j = self.barra_para_indice[linha['ID_Barra_Destino']]
                    
                    # Calcular fluxos
                    P_ij, Q_ij = self.calcular_fluxo_linha(
                        resultado.V_mag, resultado.V_ang, linha_idx
                    )
                    
                    linha_id = f"{linha['ID_Barra_Origem']}-{linha['ID_Barra_Destino']}"
                    resultado.fluxos_linhas[linha_id] = {
                        'P_ij': P_ij,
                        'Q_ij': Q_ij,
                        'S_ij': math.sqrt(P_ij**2 + Q_ij**2)
                    }
                    
                    # Calcular perdas (aproximação)
                    resultado.perdas_ativas += abs(P_ij)
                    resultado.perdas_reativas += abs(Q_ij)
            
            return resultado
            
        except Exception as e:
            print(f"Erro ao resolver OPF: {e}")
            return ResultadoOPF(sucesso=False, custo_total=0.0)
    
    def resolver_opf_multiplas_horas(self, db_path: str, horas: List[int]) -> Dict[int, ResultadoOPF]:
        """
        Resolve OPF para múltiplas horas
        
        Args:
            db_path: Caminho para banco de dados com resultados PF
            horas: Lista de horas para processar
            
        Returns:
            Dicionário com resultados por hora
        """
        resultados = {}
        
        for hora in horas:
            print(f"\n=== Processando OPF para hora {hora} ===")
            
            try:
                # Carregar resultados do PF para esta hora
                resultados_pf = self.carregar_resultados_pf(db_path, hora)
                
                # Resolver OPF
                resultado_opf = self.resolver_opf_pyomo(resultados_pf)
                
                if resultado_opf.sucesso:
                    resultados[hora] = resultado_opf
                    print(f"OPF para hora {hora} resolvido com sucesso")
                    print(f"Custo objetivo: {resultado_opf.custo_total:.6f}")
                    print(f"Perdas ativas: {resultado_opf.perdas_ativas:.4f} pu")
                else:
                    print(f"Falha ao resolver OPF para hora {hora}")
                    
            except Exception as e:
                print(f"Erro processando hora {hora}: {e}")
                import traceback
                traceback.print_exc()
        
        return resultados
    
    def salvar_resultados_opf(self, resultados: Dict[int, ResultadoOPF], db_path: str):
        """
        Salva resultados do OPF em banco de dados
        
        Args:
            resultados: Dicionário com resultados por hora
            db_path: Caminho para o banco de dados
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Criar tabela para resultados OPF
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_OPF (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hora INTEGER,
            sucesso INTEGER,
            custo_total REAL,
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
                INSERT INTO resultados_OPF 
                (hora, sucesso, custo_total, iteracoes, perdas_ativas, perdas_reativas,
                 tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
                 P_carga_json, Q_carga_json, fluxos_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    hora, 1, float(resultado.custo_total), resultado.iteracoes,
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
        
        print(f"\nResultados OPF salvos em: {db_path}")
        
    def gerar_relatorio(self, resultado: ResultadoOPF):
        """
        Gera relatório detalhado dos resultados
        
        Args:
            resultado: Resultado do OPF
        """
        if not resultado.sucesso:
            print("\nOPF não convergiu!")
            return
            
        print("\n" + "="*60)
        print("RELATÓRIO DO FLUXO DE POTÊNCIA ÓTIMO (OPF)")
        print("="*60)
        
        print(f"\nStatus: {'SUCESSO' if resultado.sucesso else 'FALHA'}")
        print(f"Custo objetivo: {resultado.custo_total:.6f}")
        print(f"Iterações: {resultado.iteracoes}")
        print(f"Perdas ativas: {resultado.perdas_ativas:.4f} pu")
        print(f"Perdas reativas: {resultado.perdas_reativas:.4f} pu")
        
        print("\n" + "-"*60)
        print("TENSÕES NAS BARRAS")
        print("-"*60)
        print(f"{'Barra':<6} {'|V| (pu)':<12} {'Ângulo (°)':<12} {'Tipo':<8}")
        print("-"*60)
        
        for i, (V, theta) in enumerate(zip(resultado.V_mag, resultado.V_ang)):
            barra_id = self.indice_para_barra[i]
            barra_tipo = next(b['tipo'] for b in self.dados['BARRAS'] 
                            if b['ID_Barra'] == barra_id)
            print(f"{barra_id:<6} {V:<12.4f} {np.degrees(theta):<12.4f} {barra_tipo:<8}")
        
        print("\n" + "-"*60)
        print("GERAÇÃO POR BARRA")
        print("-"*60)
        print(f"{'Barra':<6} {'P_ger (pu)':<12} {'Q_ger (pu)':<12} {'Gerador':<10}")
        print("-"*60)
        
        for i, (P, Q) in enumerate(zip(resultado.P_gerado, resultado.Q_gerado)):
            if P != 0 or Q != 0:
                barra_id = self.indice_para_barra[i]
                # Encontrar gerador nesta barra
                geradores_na_barra = []
                for ger in self.dados['GERADORES']:
                    if ger['ID_Barra'] == barra_id:
                        geradores_na_barra.append(ger['ID_Gerador'])
                
                geradores_str = ', '.join(geradores_na_barra)
                print(f"{barra_id:<6} {P:<12.4f} {Q:<12.4f} {geradores_str:<10}")
        
        print("\n" + "-"*60)
        print("FLUXOS NAS LINHAS")
        print("-"*60)
        print(f"{'Linha':<10} {'P_ij (pu)':<12} {'Q_ij (pu)':<12} {'|S_ij| (pu)':<12} {'% Carga':<10}")
        print("-"*60)
        
        for linha_id, fluxo in resultado.fluxos_linhas.items():
            # Encontrar limite da linha
            linha_info = None
            for l in self.dados['LINHAS']:
                if f"{l['ID_Barra_Origem']}-{l['ID_Barra_Destino']}" == linha_id:
                    linha_info = l
                    break
            
            if linha_info:
                limite = linha_info.get('LIM_Fluxo', 1.0)
                S_mag = fluxo['S_ij']
                percentual = (abs(S_mag) / limite) * 100 if limite > 0 else 0
                
                print(f"{linha_id:<10} {fluxo['P_ij']:<12.4f} {fluxo['Q_ij']:<12.4f} "
                      f"{S_mag:<12.4f} {percentual:<10.1f}%")
        
        print("="*60)

def executar_etapa_3():
    """Função principal para executar a etapa 3"""
    
    # 1. Carregar dados da rede
    with open('DATA/input/3barras_BASE.json', 'r') as f:
        dados_rede = json.load(f)
    
    # 2. Inicializar sistema OPF
    opf_system = OPFNaoLinear(dados_rede)
    
    # 3. Especificar banco de dados com resultados PF e horas a processar
    db_pf = 'resultados_PF.db' 
    horas = list(range(1, 25))  # Processar 24 horas
    
    # 4. Resolver OPF para múltiplas horas
    resultados_opf = opf_system.resolver_opf_multiplas_horas(db_pf, horas[:2])  # Apenas 2 horas para exemplo
    
    # 5. Salvar resultados
    opf_system.salvar_resultados_opf(resultados_opf, 'resultados_OPF.db')
    
    # 6. Gerar relatório para primeira hora (se disponível)
    if 1 in resultados_opf:
        opf_system.gerar_relatorio(resultados_opf[1])
    
    return resultados_opf

# Executar a etapa 3
if __name__ == "__main__":
    print("Iniciando Etapa 3: Fluxo de Potência Ótimo Não Linear")
    print("-" * 50)
    
    try:
        resultados = executar_etapa_3()
        print("\nEtapa 3 concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro na execução da Etapa 3: {e}")
        import traceback
        traceback.print_exc()