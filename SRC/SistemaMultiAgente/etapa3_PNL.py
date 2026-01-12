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
        self.custo_geracao = 0.0
        self.custo_perdas = 0.0
        self.custo_desvio = 0.0
        self.custo_curtailment = 0.0
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
        self.curtailment_total = 0.0
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.detalhes_geradores = {}
        self.geradores_eolicos = {}

class OPFNaoLinear:
    """Sistema de Fluxo de Potência Ótimo Não Linear com GWD e Curtailment"""
    
    def __init__(self, dados_rede: dict):
        self.dados = dados_rede
        self.n_barras = len(dados_rede['BARRAS'])
        self.n_linhas = len(dados_rede['LINHAS'])
        self.n_geradores = len(dados_rede['GERADORES'])
        
        # Mapeamento de índices
        self.barra_para_indice = {barra['ID_Barra']: i 
                                 for i, barra in enumerate(dados_rede['BARRAS'])}
        self.indice_para_barra = {i: barra['ID_Barra'] 
                                 for i, barra in enumerate(dados_rede['BARRAS'])}
        
        # Identificar geradores eólicos (GWD)
        self.geradores_eolicos_idx = []
        self.geradores_convencionais_idx = []
        for idx, ger in enumerate(dados_rede['GERADORES']):
            if ger.get('Tipo', '').upper() == 'GWD':
                self.geradores_eolicos_idx.append(idx)
            else:
                self.geradores_convencionais_idx.append(idx)
        
        print(f"Geradores eólicos (GWD): {self.geradores_eolicos_idx}")
        print(f"Geradores convencionais: {self.geradores_convencionais_idx}")
        
        # Configurações
        self.tolerancia = 1e-3
        self.iter_max = 100
        
        # Pesos para função objetivo
        self.peso_desvio = 10.0
        self.peso_custo_ger = 1.0
        self.peso_perdas = 0.5
        self.peso_curtailment = 1000.0
        
        # Construir matriz de admitância
        self.G, self.B = self.construir_matriz_admitancia()
        
    def carregar_resultados_pf(self, db_path: str, hora: int) -> dict:
        """Carrega resultados do Fluxo de Potência da etapa anterior"""
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
        
        return {
            'V_mag': json.loads(resultado[0]),
            'V_ang': json.loads(resultado[0]),
            'P_ger': json.loads(resultado[2]),
            'Q_ger': json.loads(resultado[3]),
            'P_carga': json.loads(resultado[4]),
            'Q_carga': json.loads(resultado[5]),
            'fluxos': json.loads(resultado[6])
        }
    
    def construir_matriz_admitancia(self) -> Tuple[np.ndarray, np.ndarray]:
        """Constrói matriz de admitância nodal Y = G + jB"""
        G = np.zeros((self.n_barras, self.n_barras), dtype=float)
        B = np.zeros((self.n_barras, self.n_barras), dtype=float)
        
        for linha in self.dados['LINHAS']:
            i = self.barra_para_indice[linha['ID_Barra_Origem']]
            j = self.barra_para_indice[linha['ID_Barra_Destino']]
            
            r = linha['R']
            x = linha['X']
            if r == 0 and x == 0:
                continue
                
            denom = r**2 + x**2
            g_serie = r / denom
            b_serie = -x / denom
            b_shunt = linha.get('Bsh', 0.0) / 2.0
            
            G[i, j] -= g_serie
            G[j, i] -= g_serie
            B[i, j] -= b_serie
            B[j, i] -= b_serie
            
            G[i, i] += g_serie
            G[j, j] += g_serie
            B[i, i] += b_serie + b_shunt
            B[j, j] += b_serie + b_shunt
        
        return G, B
    
    def calcular_fluxo_linha(self, V_mag: list, V_ang: list, 
                           linha_idx: int) -> Tuple[float, float, float]:
        """Calcula fluxo de potência em uma linha e perdas"""
        linha = self.dados['LINHAS'][linha_idx]
        i = self.barra_para_indice[linha['ID_Barra_Origem']]
        j = self.barra_para_indice[linha['ID_Barra_Destino']]
        
        r = linha['R']
        x = linha['X']
        b_sh = linha.get('Bsh', 0.0) / 2.0
        
        theta_ij = V_ang[i] - V_ang[j]
        
        denom = r**2 + x**2
        g = r / denom
        b = -x / denom
        
        # Fluxo da barra i para j
        P_ij = (V_mag[i]**2) * g - V_mag[i] * V_mag[j] * (
            g * np.cos(theta_ij) + b * np.sin(theta_ij)
        )
        
        Q_ij = -(V_mag[i]**2) * (b + b_sh) - V_mag[i] * V_mag[j] * (
            g * np.sin(theta_ij) - b * np.cos(theta_ij)
        )
        
        # Fluxo da barra j para i
        P_ji = (V_mag[j]**2) * g - V_mag[j] * V_mag[i] * (
            g * np.cos(-theta_ij) + b * np.sin(-theta_ij)
        )
        
        P_perda = P_ij + P_ji
        
        return P_ij, Q_ij, P_perda
    
    def resolver_opf_pyomo(self, resultados_pf: dict) -> ResultadoOPF:
        """Resolve o problema de OPF com GWD e curtailment usando Pyomo"""
        model = pyo.ConcreteModel()
        
        # Conjuntos
        model.BARRAS = pyo.RangeSet(0, self.n_barras - 1)
        model.LINHAS = pyo.RangeSet(0, self.n_linhas - 1)
        model.GERADORES = pyo.RangeSet(0, self.n_geradores - 1)
        model.GWD = pyo.Set(initialize=self.geradores_eolicos_idx)
        model.G_CONV = pyo.Set(initialize=self.geradores_convencionais_idx)
        
        # Variáveis de decisão
        model.V = pyo.Var(model.BARRAS, bounds=(0.85, 1.5))
        model.theta = pyo.Var(model.BARRAS, bounds=(-math.pi, math.pi))
        
        # VARIÁVEIS DE DÉFICIT (FOLGAS) COM LIMITES
        # Déficit de potência ativa (positivo = falta de geração)
        model.DEFICIT_P = pyo.Var(model.BARRAS, within=pyo.NonNegativeReals, bounds=(0, 2.0))
        # Déficit de potência reativa (positivo = falta de reativo)
        model.DEFICIT_Q = pyo.Var(model.BARRAS, within=pyo.NonNegativeReals, bounds=(0, 2.0))
        
        model.Pg = pyo.Var(model.GERADORES)
        model.Pg_util = pyo.Var(model.GERADORES)
        model.Qg = pyo.Var(model.GERADORES)
        model.curtailment = pyo.Var(model.GWD, within=pyo.NonNegativeReals)
        model.P_perda = pyo.Var(model.LINHAS, within=pyo.NonNegativeReals)
        
        # Encontrar barra slack
        slack_idx = None
        for i, barra in enumerate(self.dados['BARRAS']):
            if barra['tipo'] == 'Slack':
                slack_idx = i
                break
        
        if slack_idx is not None:
            model.theta[slack_idx].fix(0.0)
            V_slack = resultados_pf['V_mag'][slack_idx] if slack_idx < len(resultados_pf['V_mag']) else 1.0
            model.V[slack_idx].fix(V_slack)
        
        # Restrições de geração disponível
        def geracao_disponivel_rule(model, g):
            ger = self.dados['GERADORES'][g]
            barra_idx = self.barra_para_indice[ger['ID_Barra']]
            Pg_ref_pf = resultados_pf['P_ger'][barra_idx] if barra_idx < len(resultados_pf['P_ger']) else 0.0
            
            if g in model.GWD:
                return model.Pg[g] == Pg_ref_pf
            else:
                return (ger['PGERmin_MW'], model.Pg[g], ger['PGERmax_MW'])
        
        model.geracao_disponivel = pyo.Constraint(model.GERADORES, rule=geracao_disponivel_rule)
        
        # Relação entre geração disponível e utilizada
        def geracao_utilizada_rule(model, g):
            if g in model.GWD:
                return model.Pg_util[g] == model.Pg[g] - model.curtailment[g]
            else:
                return model.Pg_util[g] == model.Pg[g]
        
        model.geracao_utilizada = pyo.Constraint(model.GERADORES, rule=geracao_utilizada_rule)
        
        # Limites da geração utilizada
        def limites_utilizada_rule(model, g):
            ger = self.dados['GERADORES'][g]
            barra_idx = self.barra_para_indice[ger['ID_Barra']]
            Pg_ref_pf = resultados_pf['P_ger'][barra_idx] if barra_idx < len(resultados_pf['P_ger']) else 0.0
            
            if g in model.GWD:
                return (0.0, model.Pg_util[g], Pg_ref_pf)
            else:
                return (ger['PGERmin_MW'], model.Pg_util[g], ger['PGERmax_MW'])
        
        model.limites_utilizada = pyo.Constraint(model.GERADORES, rule=limites_utilizada_rule)
        
        # Limites de geração reativa
        def limites_reativo_rule(model, g):
            ger = self.dados['GERADORES'][g]
            return (ger['Qmin_MW'], model.Qg[g], ger['Qmax_MW'])
        
        model.limites_Qg = pyo.Constraint(model.GERADORES, rule=limites_reativo_rule)
        
        # Função objetivo MULTI-OBJETIVO COM PENALIDADES ALTÍSSIMAS PARA DÉFICIT
        def objetivo_multiobjetivo_rule(model):
            custo_total = 0.0
            
            # 0. COMPONENTE: PENALIDADE ALTÍSSIMA PARA DÉFICIT (MAIOR PRIORIDADE)
            custo_deficit = 0.0
            peso_deficit = 1000000.0  # Penalidade MUITO alta para déficit
            for i in model.BARRAS:
                custo_deficit += peso_deficit * model.DEFICIT_P[i]
                custo_deficit += peso_deficit * model.DEFICIT_Q[i]
            
            # 1. Componente: Desvio quadrático em relação ao PF
            custo_desvio = 0.0
            for idx, ger in enumerate(self.dados['GERADORES']):
                barra_idx = self.barra_para_indice[ger['ID_Barra']]
                Pg_ref = resultados_pf['P_ger'][barra_idx]
                Qg_ref = resultados_pf['Q_ger'][barra_idx]
                
                desvio_P = model.Pg_util[idx] - Pg_ref
                
                if idx in model.GWD:
                    custo_desvio += 1000 * desvio_P**2
                else:
                    custo_desvio += desvio_P**2
                
                custo_desvio += 0.1 * (model.Qg[idx] - Qg_ref)**2
            
            for i in model.BARRAS:
                V_ref = resultados_pf['V_mag'][i]
                custo_desvio += (model.V[i] - V_ref)**2
            
            # 2. Componente: Custo de geração (apenas convencionais)
            custo_geracao = 0.0
            for idx in model.G_CONV:
                ger = self.dados['GERADORES'][idx]
                custo_coef = ger.get('custo_var_USD_MW', 1.0)
                custo_geracao += custo_coef * model.Pg_util[idx]
            
            # 3. Componente: Minimização de perdas
            custo_perdas = 0.0
            for linha_idx in model.LINHAS:
                custo_perdas += model.P_perda[linha_idx]
            
            # 4. Componente: Penalidade por curtailment
            custo_curtailment = 0.0
            for idx in model.GWD:
                custo_curtailment += model.curtailment[idx]
            
            # Soma ponderada (déficit tem prioridade máxima)
            custo_total = (custo_deficit +  # PRIMEIRO: garantir que não haja déficit
                        self.peso_desvio * custo_desvio + 
                        self.peso_custo_ger * custo_geracao + 
                        self.peso_perdas * custo_perdas +
                        self.peso_curtailment * custo_curtailment)
            
            return custo_total
        
        model.objetivo = pyo.Objective(rule=objetivo_multiobjetivo_rule, sense=pyo.minimize)
        
        # Restrições de balanço de potência COM DÉFICIT
        def potencia_ativa_rule(model, i):
            P_ger_total = 0.0
            P_carga_total = 0.0
            
            # Soma gerações utilizadas na barra i
            for idx, ger in enumerate(self.dados['GERADORES']):
                if self.barra_para_indice[ger['ID_Barra']] == i:
                    P_ger_total += model.Pg_util[idx]
            
            # Soma cargas na barra i - USANDO resultado PF
            # Verificar se o valor é negativo (carga) e converter para positivo
            carga_pf = resultados_pf['P_carga'][i]
            if carga_pf < 0:
                P_carga_total = -carga_pf  # Converte para positivo
            else:
                P_carga_total = carga_pf
            
            # Potência injetada calculada
            P_inj = 0.0
            for j in model.BARRAS:
                theta_ij = model.theta[i] - model.theta[j]
                P_inj += model.V[i] * model.V[j] * (
                    self.G[i, j] * pyo.cos(theta_ij) +
                    self.B[i, j] * pyo.sin(theta_ij)
                )
            
            # Balanço com déficit (DEFICIT_P é positivo quando falta geração)
            return P_ger_total + model.DEFICIT_P[i] - P_carga_total == P_inj

        def potencia_reativa_rule(model, i):
            Q_ger_total = 0.0
            Q_carga_total = 0.0
            
            # Soma gerações reativas na barra i
            for idx, ger in enumerate(self.dados['GERADORES']):
                if self.barra_para_indice[ger['ID_Barra']] == i:
                    Q_ger_total += model.Qg[idx]
            
            # Soma cargas reativas na barra i - USANDO resultado PF
            # Verificar se o valor é negativo (carga) e converter para positivo
            carga_q_pf = resultados_pf['Q_carga'][i]
            if carga_q_pf < 0:
                Q_carga_total = -carga_q_pf  # Converte para positivo
            else:
                Q_carga_total = carga_q_pf
            
            # Potência reativa injetada calculada
            Q_inj = 0.0
            for j in model.BARRAS:
                theta_ij = model.theta[i] - model.theta[j]
                Q_inj += model.V[i] * model.V[j] * (
                    self.G[i, j] * pyo.sin(theta_ij) -
                    self.B[i, j] * pyo.cos(theta_ij)
                )
            
            # Balanço com déficit (DEFICIT_Q é positivo quando falta reativo)
            return Q_ger_total + model.DEFICIT_Q[i] - Q_carga_total == Q_inj

        model.balanco_P = pyo.Constraint(model.BARRAS, rule=potencia_ativa_rule)
        model.balanco_Q = pyo.Constraint(model.BARRAS, rule=potencia_reativa_rule)
        
        # Restrições de fluxo nas linhas e cálculo de perdas
        def fluxo_linhas_rule(model, linha_idx):
            linha = self.dados['LINHAS'][linha_idx]
            i = self.barra_para_indice[linha['ID_Barra_Origem']]
            j = self.barra_para_indice[linha['ID_Barra_Destino']]
            
            theta_ij = model.theta[i] - model.theta[j]
            
            P_ij = model.V[i] * model.V[j] * (
                self.G[i, j] * pyo.cos(theta_ij) +
                self.B[i, j] * pyo.sin(theta_ij)
            ) - self.G[i, j] * model.V[i]**2
            
            limite = linha.get('LIM_Fluxo', 2.0)
            
            return pyo.inequality(-limite, P_ij, limite)
        
        def perdas_linhas_rule(model, linha_idx):
            linha = self.dados['LINHAS'][linha_idx]
            i = self.barra_para_indice[linha['ID_Barra_Origem']]
            j = self.barra_para_indice[linha['ID_Barra_Destino']]
            
            theta_ij = model.theta[i] - model.theta[j]
            g_ij = -self.G[i, j] if i != j else 0
            
            perda_calc = g_ij * (model.V[i]**2 + model.V[j]**2 - 
                            2 * model.V[i] * model.V[j] * pyo.cos(theta_ij))
            
            return model.P_perda[linha_idx] == perda_calc
        
        model.fluxo_linhas = pyo.Constraint(model.LINHAS, rule=fluxo_linhas_rule)
        model.perdas_linhas = pyo.Constraint(model.LINHAS, rule=perdas_linhas_rule)
        
        # DESCOMENTAR: Valores iniciais baseados no PF (ajuda na convergência)
        for i in model.BARRAS:
            if i != slack_idx:
                if i < len(resultados_pf['V_mag']):
                    model.V[i] = resultados_pf['V_mag'][i]
                if i < len(resultados_pf['V_ang']):
                    model.theta[i] = resultados_pf['V_ang'][i]
        
        for idx, ger in enumerate(self.dados['GERADORES']):
            barra_idx = self.barra_para_indice[ger['ID_Barra']]
            if barra_idx < len(resultados_pf['P_ger']):
                model.Pg_util[idx] = resultados_pf['P_ger'][barra_idx]
                model.Pg[idx] = resultados_pf['P_ger'][barra_idx]
            if barra_idx < len(resultados_pf['Q_ger']):
                model.Qg[idx] = resultados_pf['Q_ger'][barra_idx]
        
        # Inicializar variáveis de déficit com zero
        for i in model.BARRAS:
            model.DEFICIT_P[i] = 0.0
            model.DEFICIT_Q[i] = 0.0

        # Resolver o problema
        solver = pyo.SolverFactory('ipopt')
        solver.options['tol'] = self.tolerancia
        solver.options['max_iter'] = self.iter_max
        solver.options['print_level'] = 5  # Aumentar para ver mais detalhes
        solver.options['max_cpu_time'] = 30.0
        
        # Configurações para melhor convergência
        solver.options['mu_strategy'] = 'adaptive'
        solver.options['linear_solver'] = 'mumps'
        solver.options['acceptable_tol'] = 1e-4
        
        try:
            results = solver.solve(model, tee=True)  # tee=True para ver o log
            
            sucesso = str(results.solver.termination_condition) in ['optimal', 'locallyOptimal', 'userInterrupt']
            
            # Calcular componentes da função objetivo
            custo_deficit = 0.0
            custo_desvio = 0.0
            custo_geracao = 0.0
            custo_perdas = 0.0
            custo_curtailment = 0.0
            
            resultado = ResultadoOPF(
                sucesso=sucesso,
                custo_total=pyo.value(model.objetivo) if sucesso else 0.0,
                iteracoes=results.solver.iterations if hasattr(results.solver, 'iterations') else 0
            )
            
            if sucesso:
                # Calcular déficit total
                deficit_p_total = 0.0
                deficit_q_total = 0.0
                for i in model.BARRAS:
                    deficit_p_total += pyo.value(model.DEFICIT_P[i])
                    deficit_q_total += pyo.value(model.DEFICIT_Q[i])
                    custo_deficit += 1000000.0 * (pyo.value(model.DEFICIT_P[i]) + pyo.value(model.DEFICIT_Q[i]))
                
                print(f"\nDéficit P total: {deficit_p_total:.6f} pu")
                print(f"Déficit Q total: {deficit_q_total:.6f} pu")
                
                for idx, ger in enumerate(self.dados['GERADORES']):
                    barra_idx = self.barra_para_indice[ger['ID_Barra']]
                    Pg_ref = resultados_pf['P_ger'][barra_idx]
                    Qg_ref = resultados_pf['Q_ger'][barra_idx]
                    
                    Pg_util_val = pyo.value(model.Pg_util[idx])
                    Qg_val = pyo.value(model.Qg[idx])
                    
                    if idx in self.geradores_eolicos_idx:
                        desvio = Pg_util_val - Pg_ref
                        custo_desvio += 1000 * (desvio)**2
                    else:
                        custo_desvio += (Pg_util_val - Pg_ref)**2
                    
                    custo_desvio += 0.1 * (Qg_val - Qg_ref)**2
                    
                    if idx in self.geradores_convencionais_idx:
                        custo_coef = ger.get('custo_var_USD_MW', 1.0)
                        custo_geracao += custo_coef * Pg_util_val
                
                for i in model.BARRAS:
                    V_ref = resultados_pf['V_mag'][i]
                    V_val = pyo.value(model.V[i])
                    custo_desvio += (V_val - V_ref)**2
                
                for linha_idx in model.LINHAS:
                    custo_perdas += pyo.value(model.P_perda[linha_idx])
                
                for idx in self.geradores_eolicos_idx:
                    custo_curtailment += pyo.value(model.curtailment[idx])
            
            # Armazenar componentes (incluindo déficit)
            resultado.custo_deficit = custo_deficit
            resultado.custo_desvio = custo_desvio * self.peso_desvio
            resultado.custo_geracao = custo_geracao * self.peso_custo_ger
            resultado.custo_perdas = custo_perdas * self.peso_perdas
            resultado.custo_curtailment = custo_curtailment * self.peso_curtailment
            
            if sucesso:
                # Preencher resultados
                resultado.V_mag = [pyo.value(model.V[i]) for i in model.BARRAS]
                resultado.V_ang = [pyo.value(model.theta[i]) for i in model.BARRAS]
                
                resultado.P_gerado = [0.0] * self.n_barras
                resultado.Q_gerado = [0.0] * self.n_barras
                resultado.DEFICIT_P = [pyo.value(model.DEFICIT_P[i]) for i in model.BARRAS]
                resultado.DEFICIT_Q = [pyo.value(model.DEFICIT_Q[i]) for i in model.BARRAS]
                
                for idx, ger in enumerate(self.dados['GERADORES']):
                    barra_idx = self.barra_para_indice[ger['ID_Barra']]
                    Pg_util_val = pyo.value(model.Pg_util[idx])
                    Qg_val = pyo.value(model.Qg[idx])
                    Pg_disponivel_val = pyo.value(model.Pg[idx])
                    
                    resultado.P_gerado[barra_idx] = Pg_util_val
                    resultado.Q_gerado[barra_idx] = Qg_val
                    
                    detalhes = {
                        'barra': ger['ID_Barra'],
                        'P_ger': Pg_util_val,
                        'Q_ger': Qg_val,
                        'P_disponivel': Pg_disponivel_val,
                        'P_min': ger['PGERmin_MW'],
                        'P_max': ger['PGERmax_MW'],
                        'tipo': ger.get('Tipo', 'CONV')
                    }
                    
                    if idx in self.geradores_eolicos_idx:
                        curtailment_val = pyo.value(model.curtailment[idx])
                        detalhes['curtailment'] = curtailment_val
                        pf_ger = resultados_pf['P_ger'][barra_idx] if barra_idx < len(resultados_pf['P_ger']) else 0.0
                        detalhes['curtailment_percent'] = (curtailment_val / pf_ger * 100 
                                                        if pf_ger > 0 else 0)
                        resultado.curtailment_total += curtailment_val
                        resultado.geradores_eolicos[ger['ID_Gerador']] = detalhes
                    else:
                        custo_coef = ger.get('custo_var_USD_MW', 1.0)
                        detalhes['custo_coef'] = custo_coef
                        detalhes['custo_total'] = custo_coef * Pg_util_val
                        detalhes['percentual_uso'] = ((Pg_util_val - ger['PGERmin_MW']) / 
                                                    (ger['PGERmax_MW'] - ger['PGERmin_MW']) * 100 
                                                    if ger['PGERmax_MW'] > ger['PGERmin_MW'] else 0)
                    
                    resultado.detalhes_geradores[ger['ID_Gerador']] = detalhes
                
                # Carregar dados de carga do PF (com correção de sinal)
                for i in range(self.n_barras):
                    carga_p = resultados_pf['P_carga'][i]
                    if carga_p < 0:
                        resultado.P_carga.append(-carga_p)
                    else:
                        resultado.P_carga.append(carga_p)
                        
                    carga_q = resultados_pf['Q_carga'][i]
                    if carga_q < 0:
                        resultado.Q_carga.append(-carga_q)
                    else:
                        resultado.Q_carga.append(carga_q)
                
                # Calcular fluxos nas linhas e perdas
                for linha_idx in range(self.n_linhas):
                    linha = self.dados['LINHAS'][linha_idx]
                    i = self.barra_para_indice[linha['ID_Barra_Origem']]
                    j = self.barra_para_indice[linha['ID_Barra_Destino']]
                    
                    P_ij, Q_ij, P_perda = self.calcular_fluxo_linha(
                        resultado.V_mag, resultado.V_ang, linha_idx
                    )
                    
                    linha_id = f"{linha['ID_Barra_Origem']}-{linha['ID_Barra_Destino']}"
                    resultado.fluxos_linhas[linha_id] = {
                        'P_ij': P_ij,
                        'Q_ij': Q_ij,
                        'P_perda': P_perda,
                        'S_ij': math.sqrt(P_ij**2 + Q_ij**2),
                        'limite': linha.get('LIM_Fluxo', 1.0),
                        'percentual_uso': abs(P_ij) / linha.get('LIM_Fluxo', 1.0) * 100 
                                        if linha.get('LIM_Fluxo', 1.0) > 0 else 0
                    }
                    
                    resultado.perdas_ativas += abs(P_perda)
                    resultado.perdas_reativas += abs(Q_ij)
            
            return resultado
            
        except Exception as e:
            print(f"Erro ao resolver OPF: {e}")
            import traceback
            traceback.print_exc()
            return ResultadoOPF(sucesso=False, custo_total=0.0)

    def resolver_opf_multiplas_horas(self, db_path: str, horas: List[int]) -> Dict[int, ResultadoOPF]:
        """
        Resolve OPF para múltiplas horas
        """
        resultados = {}
        
        for hora in horas:
            print(f"\n=== Processando OPF para hora {hora} ===")
            
            try:
                resultados_pf = self.carregar_resultados_pf(db_path, hora)
                resultado_opf = self.resolver_opf_pyomo(resultados_pf)
                
                if resultado_opf.sucesso:
                    resultados[hora] = resultado_opf
                    print(f"OPF para hora {hora} resolvido com sucesso")
                    print(f"Custo total: {resultado_opf.custo_total:.6f}")
                    print(f"  - Custo desvio: {resultado_opf.custo_desvio:.6f}")
                    print(f"  - Custo geração: {resultado_opf.custo_geracao:.6f}")
                    print(f"  - Custo perdas: {resultado_opf.custo_perdas:.6f}")
                    print(f"  - Custo curtailment: {resultado_opf.custo_curtailment:.6f}")
                    print(f"Perdas ativas: {resultado_opf.perdas_ativas:.4f} pu")
                    print(f"Perdas reativas: {resultado_opf.perdas_reativas:.4f} pu")
                    print(f"Curtailment total: {resultado_opf.curtailment_total:.4f} pu")
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
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_OPF (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hora INTEGER,
            sucesso INTEGER,
            custo_total REAL,
            custo_desvio REAL,
            custo_geracao REAL,
            custo_perdas REAL,
            custo_curtailment REAL,
            iteracoes INTEGER,
            perdas_ativas REAL,
            perdas_reativas REAL,
            curtailment_total REAL,
            tensoes_mag_json TEXT,
            tensoes_ang_json TEXT,
            P_gerado_json TEXT,
            Q_gerado_json TEXT,
            P_carga_json TEXT,
            Q_carga_json TEXT,
            fluxos_json TEXT,
            geradores_json TEXT,
            eolicos_json TEXT,
            timestamp TEXT
        )
        ''')
        
        for hora, resultado in resultados.items():
            if resultado.sucesso:
                cursor.execute('''
                INSERT INTO resultados_OPF 
                (hora, sucesso, custo_total, custo_desvio, custo_geracao, custo_perdas, custo_curtailment,
                 iteracoes, perdas_ativas, perdas_reativas, curtailment_total,
                 tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
                 P_carga_json, Q_carga_json, fluxos_json, geradores_json, eolicos_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    hora, 1, 
                    float(resultado.custo_total), 
                    float(resultado.custo_desvio),
                    float(resultado.custo_geracao),
                    float(resultado.custo_perdas),
                    float(resultado.custo_curtailment),
                    resultado.iteracoes,
                    float(resultado.perdas_ativas), 
                    float(resultado.perdas_reativas),
                    float(resultado.curtailment_total),
                    json.dumps([float(v) for v in resultado.V_mag]),
                    json.dumps([float(v) for v in resultado.V_ang]),
                    json.dumps([float(v) for v in resultado.P_gerado]),
                    json.dumps([float(v) for v in resultado.Q_gerado]),
                    json.dumps([float(v) for v in resultado.P_carga]),
                    json.dumps([float(v) for v in resultado.Q_carga]),
                    json.dumps(resultado.fluxos_linhas),
                    json.dumps(resultado.detalhes_geradores),
                    json.dumps(resultado.geradores_eolicos),
                    resultado.timestamp
                ))
        
        conn.commit()
        conn.close()
        
        print(f"\nResultados OPF salvos em: {db_path}")
    
    def gerar_tabela_resultados(self, resultados: Dict[int, ResultadoOPF]):
        """
        Gera tabela com resultados por hora no formato solicitado
        
        Colunas: Hora | Carga Total | G1 | G2 | GWD | Curtailment | Perdas Ativas | Perdas Reativas
        """
        print("\n" + "="*90)
        print("TABELA DE RESULTADOS OPF - 24 HORAS")
        print("="*90)
        print(f"{'Hora':<6} | {'Carga Total':<12} | {'G1':<10} | {'G2':<10} | {'GWD':<10} | "
              f"{'Curtailment':<12} | {'Perdas Ativas':<14} | {'Perdas Reativas':<15}")
        print("-"*90)
        
        horas_ordenadas = sorted(resultados.keys())
        
        for hora in horas_ordenadas:
            if hora in resultados and resultados[hora].sucesso:
                r = resultados[hora]
                
                # Calcular carga total
                carga_total = sum(r.P_carga)
                
                # Extrair valores dos geradores
                geradores_por_tipo = {'G1': 0.0, 'G2': 0.0, 'GWD': 0.0}
                
                for ger_id, detalhes in r.detalhes_geradores.items():
                    if detalhes['tipo'] == 'GWD':
                        geradores_por_tipo['GWD'] += detalhes['P_ger']
                    else:
                        if geradores_por_tipo['G1'] == 0:
                            geradores_por_tipo['G1'] = detalhes['P_ger']
                        else:
                            geradores_por_tipo['G2'] = detalhes['P_ger']
                
                print(f"{hora:02d}:00 | {carga_total:<12.4f} | {geradores_por_tipo['G1']:<10.4f} | "
                      f"{geradores_por_tipo['G2']:<10.4f} | {geradores_por_tipo['GWD']:<10.4f} | "
                      f"{r.curtailment_total:<12.4f} | {r.perdas_ativas:<14.4f} | {r.perdas_reativas:<15.4f}")
        
        print("="*90)
        
        # Estatísticas totais
        if resultados:
            horas_sucesso = [h for h, r in resultados.items() if r.sucesso]
            if horas_sucesso:
                print("\nRESUMO GERAL:")
                print(f"Horas processadas com sucesso: {len(horas_sucesso)}")
                
                perdas_ativas_total = sum(resultados[h].perdas_ativas for h in horas_sucesso)
                perdas_reativas_total = sum(resultados[h].perdas_reativas for h in horas_sucesso)
                curtailment_total = sum(resultados[h].curtailment_total for h in horas_sucesso)
                
                print(f"Perdas ativas totais (24h): {perdas_ativas_total:.4f} pu")
                print(f"Perdas reativas totais (24h): {perdas_reativas_total:.4f} pu")
                print(f"Curtailment total (24h): {curtailment_total:.4f} pu")
    
    def gerar_relatorio(self, resultado: ResultadoOPF):
        """
        Gera relatório detalhado dos resultados
        """
        if not resultado.sucesso:
            print("\nOPF não convergiu!")
            return
            
        print("\n" + "="*80)
        print("RELATÓRIO DO FLUXO DE POTÊNCIA ÓTIMO (OPF) COM GWD")
        print("="*80)
        
        print(f"\nStatus: {'SUCESSO' if resultado.sucesso else 'FALHA'}")
        print(f"Custo total: {resultado.custo_total:.6f}")
        print(f"Componentes do custo:")
        print(f"  - Desvio: {resultado.custo_desvio:.6f} (peso: {self.peso_desvio})")
        print(f"  - Geração: {resultado.custo_geracao:.6f} (peso: {self.peso_custo_ger})")
        print(f"  - Perdas: {resultado.custo_perdas:.6f} (peso: {self.peso_perdas})")
        print(f"  - Curtailment: {resultado.custo_curtailment:.6f} (peso: {self.peso_curtailment})")
        print(f"Iterações: {resultado.iteracoes}")
        print(f"Perdas ativas: {resultado.perdas_ativas:.6f} pu")
        print(f"Perdas reativas: {resultado.perdas_reativas:.6f} pu")
        print(f"Curtailment total: {resultado.curtailment_total:.6f} pu")
        
        print("\n" + "-"*80)
        print("GERADORES EÓLICOS (GWD)")
        print("-"*80)
        print(f"{'Gerador':<10} {'Barra':<6} {'Disponível':<12} {'Utilizado':<12} {'Curtailment':<12} {'Curt%':<10}")
        print("-"*80)
        
        for ger_id, detalhes in resultado.geradores_eolicos.items():
            print(f"{ger_id:<10} {detalhes['barra']:<6} {detalhes['P_disponivel']:<12.4f} "
                  f"{detalhes['P_ger']:<12.4f} {detalhes.get('curtailment', 0):<12.4f} "
                  f"{detalhes.get('curtailment_percent', 0):<10.1f}")
        
        print("\n" + "-"*80)
        print("GERADORES CONVENCIONAIS")
        print("-"*80)
        print(f"{'Gerador':<10} {'Barra':<6} {'P (pu)':<12} {'Q (pu)':<12} {'Min':<8} {'Max':<8} {'Uso %':<10}")
        print("-"*80)
        
        for ger_id, detalhes in resultado.detalhes_geradores.items():
            if detalhes['tipo'] != 'GWD':
                print(f"{ger_id:<10} {detalhes['barra']:<6} {detalhes['P_ger']:<12.4f} "
                      f"{detalhes['Q_ger']:<12.4f} {detalhes['P_min']:<8.2f} "
                      f"{detalhes['P_max']:<8.2f} {detalhes.get('percentual_uso', 0):<10.1f}")
        
        print("\n" + "-"*80)
        print("FLUXOS NAS LINHAS")
        print("-"*80)
        print(f"{'Linha':<12} {'P_ij (pu)':<12} {'Q_ij (pu)':<12} {'Perda (pu)':<12} {'Uso %':<10} {'Status':<10}")
        print("-"*80)
        
        for linha_id, fluxo in resultado.fluxos_linhas.items():
            status = "OK"
            if fluxo['percentual_uso'] > 100:
                status = "SOBRECARGA"
            elif fluxo['percentual_uso'] > 85:
                status = "ALERTA"
            
            print(f"{linha_id:<12} {fluxo['P_ij']:<12.4f} {fluxo['Q_ij']:<12.4f} "
                  f"{fluxo['P_perda']:<12.4f} {fluxo['percentual_uso']:<10.1f} {status:<10}")
        
        print("="*80)

def executar_etapa_3():
    """Função principal para executar a etapa 3"""
    
    # 1. Carregar dados da rede
    with open('DATA/input/3barras_BASE.json', 'r') as f:
        dados_rede = json.load(f)
    
    # 2. Inicializar sistema OPF
    opf_system = OPFNaoLinear(dados_rede)
    
    # 3. Especificar banco de dados com resultados PF e horas a processar
    db_pf = 'DATA/SMA/resultados_PF.db' 
    horas = [0]  # Corrigido: começa em 1
    
    # 4. Resolver OPF para múltiplas horas
    resultados_opf = opf_system.resolver_opf_multiplas_horas(db_pf, horas)
    
    # 5. Salvar resultados
    opf_system.salvar_resultados_opf(resultados_opf, 'DATA/SMA/resultados_OPF.db')
    
    # 6. Gerar tabela de resultados
    if resultados_opf:
        opf_system.gerar_tabela_resultados(resultados_opf)
    
    # 7. Gerar relatório para primeira hora (se disponível)
    if 1 in resultados_opf and resultados_opf[0].sucesso:
        opf_system.gerar_relatorio(resultados_opf[0])
    
    return resultados_opf

# Executar a etapa 3
if __name__ == "__main__":
    print("=" * 80)
    print("ETAPA 3: FLUXO DE POTÊNCIA ÓTIMO NÃO LINEAR COM GWD E CURTAILMENT")
    print("=" * 80)
    print("Função objetivo: Minimizar [desvio + custo geração + perdas + curtailment]")
    print(f"Pesos: Desvio={10.0}, Custo Geração={1.0}, Perdas={0.5}, Curtailment={1000.0}")
    print("=" * 80)
    
    try:
        resultados = executar_etapa_3()
        print("\nEtapa 3 concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro na execução da Etapa 3: {e}")
        import traceback
        traceback.print_exc()