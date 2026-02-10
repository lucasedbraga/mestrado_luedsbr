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
        self.deficit_total = 0.0
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.detalhes_geradores = {}
        self.geradores_eolicos = {}

class OPFNaoLinear:
    """Sistema de Fluxo de Potência Ótimo Não Linear com GWD e Curtailment em PU"""
    
    def __init__(self, dados_rede: dict):
        self.dados = dados_rede
        self.S_base = dados_rede['S_base']  # Potência base (geralmente 100 MVA)
        
        # Converter dados para PU
        self.converter_para_pu()
        
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
        self.peso_deficit = 10000.0  # Penalidade MUITO alta para déficit
        
        # Construir matriz de admitância
        self.G, self.B = self.construir_matriz_admitancia()
    
    def converter_para_pu(self):
        """Converte todos os valores para PU usando S_base"""
        S_base = self.S_base
        
        # Converter cargas das barras
        for barra in self.dados['BARRAS']:
            barra['P_carga_pu'] = barra.get('P_carga_MW', 0.0) / S_base
            barra['Q_carga_pu'] = barra.get('Q_carga_MVAr', 0.0) / S_base
        
        # Converter geradores
        for ger in self.dados['GERADORES']:
            # Potências
            ger['PGERmin_pu'] = ger.get('PGERmin_MW', 0.0) / S_base
            ger['PGERmax_pu'] = ger.get('PGERmax_MW', 0.0) / S_base
            ger['Qmin_pu'] = ger.get('Qmin_MW', 0.0) / S_base
            ger['Qmax_pu'] = ger.get('Qmax_MW', 0.0) / S_base
            
            # Rampas (MW/h -> pu/h)
            ger['ramp_up_pu_h'] = ger.get('ramp_up_MW_h', 0.0) / S_base
            ger['ramp_down_pu_h'] = ger.get('ramp_down_MW_h', 0.0) / S_base
            
            # Custos (USD/MW -> USD/pu)
            ger['custo_var_USD_pu'] = ger.get('custo_var_USD_MW', 0.0) * S_base
            ger['custo_curtailment_USD_pu'] = ger.get('custo_curtailment_USD_MW', 0.0) * S_base
            
            # Emissões (tCO2/MWh -> tCO2/pu)
            ger['emissao_tCO2_pu'] = ger.get('emissao_tCO2_MWh', 0.0) * S_base
        
        # Converter demandas
        for demanda in self.dados['DEMANDAS']:
            demanda['PLOAD_pu'] = demanda.get('PLOAD', 0.0) / S_base
            demanda['QLOAD_pu'] = demanda.get('QLOAD_MW', 0.0) / S_base
        
        # Converter limites de fluxo
        for linha in self.dados['LINHAS']:
            if linha.get('LIM_Fluxo_Unidade', 'MW') == 'MW':
                linha['LIM_Fluxo_pu'] = linha.get('LIM_Fluxo', 0.0) / S_base
        
        print(f"Conversão para PU concluída (S_base = {S_base} MVA)")
    
    def carregar_resultados_pf(self, db_path: str, hora: int) -> dict:
        """Carrega resultados do Fluxo de Potência da etapa anterior e converte para PU"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Primeiro, verificar quais horas estão disponíveis
        cursor.execute('SELECT DISTINCT hora FROM resultados_fluxo ORDER BY hora')
        horas_disponiveis = [row[0] for row in cursor.fetchall()]
        print(f"Horas disponíveis no banco de dados: {horas_disponiveis}")
        
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
        
        # Converter resultados do PF para PU
        P_ger_json = json.loads(resultado[2])
        Q_ger_json = json.loads(resultado[3])
        P_carga_json = json.loads(resultado[4])
        Q_carga_json = json.loads(resultado[5])
        
        # Converter MW para pu
        P_ger_pu = [p / self.S_base for p in P_ger_json]
        Q_ger_pu = [q / self.S_base for q in Q_ger_json]
        P_carga_pu = [p / self.S_base for p in P_carga_json]
        Q_carga_pu = [q / self.S_base for q in Q_carga_json]
        
        return {
            'V_mag': json.loads(resultado[0]),
            'V_ang': json.loads(resultado[1]),
            'P_ger': P_ger_pu,
            'Q_ger': Q_ger_pu,
            'P_carga': P_carga_pu,
            'Q_carga': Q_carga_pu,
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
        """Calcula fluxo de potência em uma linha e perdas (em pu)"""
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
        
        # Fluxo da barra i para j (em pu)
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

    def resolver_opf_pyomo(self, resultados_pf: dict, hora: int) -> ResultadoOPF:
        """Resolve o problema de OPF com GWD, curtailment e déficit usando Pyomo (tudo em PU)"""
        print(f"\n{'='*60}")
        print(f"Resolvendo OPF para hora {hora}")
        print(f"{'='*60}")
        
        model = pyo.ConcreteModel()
        
        # Conjuntos
        model.BARRAS = pyo.RangeSet(0, self.n_barras - 1)
        model.LINHAS = pyo.RangeSet(0, self.n_linhas - 1)
        model.GERADORES = pyo.RangeSet(0, self.n_geradores - 1)
        model.GWD = pyo.Set(initialize=self.geradores_eolicos_idx)
        model.G_CONV = pyo.Set(initialize=self.geradores_convencionais_idx)
        
        # Variáveis de decisão
        model.V = pyo.Var(model.BARRAS, bounds=(0.85, 1.15))
        model.theta = pyo.Var(model.BARRAS, bounds=(-math.pi, math.pi))
        
        # VARIÁVEIS DE DÉFICIT (FOLGAS) COM LIMITES (em pu)
        model.DEFICIT_P = pyo.Var(model.BARRAS, within=pyo.NonNegativeReals, bounds=(0, 1.0))  
        model.DEFICIT_Q = pyo.Var(model.BARRAS, within=pyo.NonNegativeReals, bounds=(0, 1.0))  
        
        model.Pg = pyo.Var(model.GERADORES)  # Geração disponível (pu)
        model.Pg_util = pyo.Var(model.GERADORES)  # Geração utilizada (pu)
        model.Qg = pyo.Var(model.GERADORES)  # Reativo gerado (pu)
        model.curtailment = pyo.Var(model.GWD, within=pyo.NonNegativeReals)  # Curtailment (pu)
        model.P_perda = pyo.Var(model.LINHAS, within=pyo.NonNegativeReals)  # Perdas (pu)
        
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
        
        # Restrições de geração disponível (em pu)
        def geracao_disponivel_rule(model, g):
            ger = self.dados['GERADORES'][g]
            
            if g in model.GWD:
                # Para GWD, geração disponível é fixa (valor do PF)
                barra_idx = self.barra_para_indice[ger['ID_Barra']]
                Pg_ref_pf = resultados_pf['P_ger'][barra_idx] if barra_idx < len(resultados_pf['P_ger']) else 0.0
                return model.Pg[g] == Pg_ref_pf
            else:
                # Para geradores convencionais, limites normais
                return model.Pg[g] == model.Pg_util[g]
        
        model.geracao_disponivel = pyo.Constraint(model.GERADORES, rule=geracao_disponivel_rule)
        
        # Relação entre geração disponível e utilizada para GWD
        def geracao_utilizada_gwd_rule(model, g):
            if g in model.GWD:
                return model.Pg_util[g] == model.Pg[g] - model.curtailment[g]
            else:
                return pyo.Constraint.Skip
        
        model.geracao_utilizada_gwd = pyo.Constraint(model.GWD, rule=geracao_utilizada_gwd_rule)
        
        # Para geradores convencionais, utilizada = disponível
        def geracao_utilizada_conv_rule(model, g):
            if g not in model.GWD:
                return model.Pg_util[g] == model.Pg[g]
            else:
                return pyo.Constraint.Skip
        
        model.geracao_utilizada_conv = pyo.Constraint(model.G_CONV, rule=geracao_utilizada_conv_rule)
        
        # Limites inferiores para geração utilizada
        def limites_inferiores_utilizada_rule(model, g):
            ger = self.dados['GERADORES'][g]
            return model.Pg_util[g] >= ger['PGERmin_pu']
        
        model.limites_inf_utilizada = pyo.Constraint(model.GERADORES, 
                                                    rule=limites_inferiores_utilizada_rule)
        
        # Limites superiores para geração utilizada
        def limites_superiores_utilizada_rule(model, g):
            ger = self.dados['GERADORES'][g]
            
            if g in model.GWD:
                # Para GWD, limite superior é a geração disponível (valor do PF)
                return model.Pg_util[g] <= model.Pg[g]
            else:
                # Para geradores convencionais, limite fixo
                return model.Pg_util[g] <= ger['PGERmax_pu']
        
        model.limites_sup_utilizada = pyo.Constraint(model.GERADORES, 
                                                    rule=limites_superiores_utilizada_rule)
        
        # Limites de geração disponível (convencionais)
        def limites_disponivel_conv_rule(model, g):
            if g not in model.GWD:
                ger = self.dados['GERADORES'][g]
                return (ger['PGERmin_pu'], model.Pg[g], ger['PGERmax_pu'])
            else:
                return pyo.Constraint.Skip
        
        model.limites_disponivel_conv = pyo.Constraint(model.G_CONV, rule=limites_disponivel_conv_rule)
        
        # Limites de geração reativa (em pu)
        def limites_reativo_inf_rule(model, g):
            ger = self.dados['GERADORES'][g]
            return model.Qg[g] >= ger['Qmin_pu']
        
        def limites_reativo_sup_rule(model, g):
            ger = self.dados['GERADORES'][g]
            return model.Qg[g] <= ger['Qmax_pu']
        
        model.limites_Qg_inf = pyo.Constraint(model.GERADORES, rule=limites_reativo_inf_rule)
        model.limites_Qg_sup = pyo.Constraint(model.GERADORES, rule=limites_reativo_sup_rule)
        
        # Função objetivo MULTI-OBJETIVO (tudo em pu) COM DÉFICIT
        def objetivo_multiobjetivo_rule(model):
            custo_total = 0.0
            
            # 0. CUSTO DE DÉFICIT (prioridade máxima - penalidade MUITO alta)
            custo_deficit = 0.0
            for i in model.BARRAS:
                custo_deficit += self.peso_deficit * model.DEFICIT_P[i]
                custo_deficit += self.peso_deficit * 0.5 * model.DEFICIT_Q[i]
            
            # 1. Desvio quadrático em relação ao PF (em pu)
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
            
            # 2. Custo de geração (convencionais, em USD/pu)
            custo_geracao = 0.0
            for idx in model.G_CONV:
                ger = self.dados['GERADORES'][idx]
                custo_coef = ger.get('custo_var_USD_pu', 1.0)
                custo_geracao += custo_coef * model.Pg_util[idx]
            
            # 3. Minimização de perdas (em pu)
            custo_perdas = 0.0
            for linha_idx in model.LINHAS:
                custo_perdas += model.P_perda[linha_idx]
            
            # 4. Penalidade por curtailment (em USD/pu)
            custo_curtailment = 0.0
            for idx in model.GWD:
                ger = self.dados['GERADORES'][idx]
                custo_curtailment_coef = ger.get('custo_curtailment_USD_pu', 1000.0)
                custo_curtailment += custo_curtailment_coef * model.curtailment[idx]
            
            # Soma ponderada - déficit tem a MAIOR prioridade
            custo_total = (custo_deficit + 
                        self.peso_desvio * custo_desvio + 
                        self.peso_custo_ger * custo_geracao + 
                        self.peso_perdas * custo_perdas +
                        custo_curtailment)
            
            return custo_total
        
        model.objetivo = pyo.Objective(rule=objetivo_multiobjetivo_rule, sense=pyo.minimize)
        
        # Restrições de balanço de potência COM DÉFICIT (em pu)
        def potencia_ativa_rule(model, i):
            P_ger_total = 0.0
            P_carga_total = 0.0
            
            # Soma gerações utilizadas na barra i
            for idx, ger in enumerate(self.dados['GERADORES']):
                if self.barra_para_indice[ger['ID_Barra']] == i:
                    P_ger_total += model.Pg_util[idx]
            
            # Soma cargas na barra i (já em pu)
            P_carga_total = resultados_pf['P_carga'][i]
            
            # Potência injetada calculada
            P_inj = 0.0
            for j in model.BARRAS:
                theta_ij = model.theta[i] - model.theta[j]
                P_inj += model.V[i] * model.V[j] * (
                    self.G[i, j] * pyo.cos(theta_ij) +
                    self.B[i, j] * pyo.sin(theta_ij)
                )
            
            # Balanço COM DÉFICIT: Geração + Déficit = Carga + Injeção + Perdas
            return P_ger_total + model.DEFICIT_P[i] == P_carga_total + P_inj

        def potencia_reativa_rule(model, i):
            Q_ger_total = 0.0
            Q_carga_total = 0.0
            
            # Soma gerações reativas na barra i
            for idx, ger in enumerate(self.dados['GERADORES']):
                if self.barra_para_indice[ger['ID_Barra']] == i:
                    Q_ger_total += model.Qg[idx]
            
            # Soma cargas reativas na barra i (já em pu)
            Q_carga_total = resultados_pf['Q_carga'][i]
            
            # Potência reativa injetada calculada
            Q_inj = 0.0
            for j in model.BARRAS:
                theta_ij = model.theta[i] - model.theta[j]
                Q_inj += model.V[i] * model.V[j] * (
                    self.G[i, j] * pyo.sin(theta_ij) -
                    self.B[i, j] * pyo.cos(theta_ij)
                )
            
            # Balanço COM DÉFICIT
            return Q_ger_total + model.DEFICIT_Q[i] == Q_carga_total + Q_inj

        model.balanco_P = pyo.Constraint(model.BARRAS, rule=potencia_ativa_rule)
        model.balanco_Q = pyo.Constraint(model.BARRAS, rule=potencia_reativa_rule)
        
        def fluxo_linhas_rule(model, linha_idx):
            linha = self.dados['LINHAS'][linha_idx]
            i = self.barra_para_indice[linha['ID_Barra_Origem']]
            j = self.barra_para_indice[linha['ID_Barra_Destino']]
            
            theta_ij = model.theta[i] - model.theta[j]
            
            P_ij = model.V[i] * model.V[j] * (
                self.G[i, j] * pyo.cos(theta_ij) +
                self.B[i, j] * pyo.sin(theta_ij)
            ) - self.G[i, j] * model.V[i]**2
            
            limite = linha.get('LIM_Fluxo_pu', 1.0)
            
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
        
        # Valores iniciais baseados no PF
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
        
        # Inicializar curtailment com zero
        for idx in model.GWD:
            model.curtailment[idx] = 0.0

        # RESOLVER COM CONFIGURAÇÕES PARA MELHOR CONVERGÊNCIA
        solver = pyo.SolverFactory('ipopt')
        
        # Configurações agressivas para evitar infactibilidade
        solver.options['tol'] = self.tolerancia
        solver.options['max_iter'] = self.iter_max
        solver.options['print_level'] = 5
        solver.options['max_cpu_time'] = 60.0
        solver.options['acceptable_tol'] = 1e-4
        solver.options['acceptable_iter'] = 5
        
        # Configurações para melhor convergência
        solver.options['mu_strategy'] = 'adaptive'
        solver.options['mu_init'] = 1e-2
        solver.options['linear_solver'] = 'mumps'
        solver.options['nlp_scaling_method'] = 'gradient-based'
        solver.options['bound_relax_factor'] = 1e-6
        
        # Opções para lidar com infactibilidade
        solver.options['bound_push'] = 1e-6
        solver.options['bound_frac'] = 1e-6
        solver.options['slack_bound_push'] = 1e-6
        solver.options['slack_bound_frac'] = 1e-6
        solver.options['constr_viol_tol'] = 1e-4
        solver.options['acceptable_constr_viol_tol'] = 1e-3
        
        try:
            print(f"\nResolvendo modelo OPF para hora {hora} com IPOPT...")
            results = solver.solve(model, tee=True)
            
            sucesso = str(results.solver.termination_condition) in ['optimal', 'locallyOptimal', 'userInterrupt', 'acceptable']
            
            if not sucesso:
                print(f"Status do solver: {results.solver.termination_condition}")
                print("Tentando resolver com configurações relaxadas...")
                
                # Tentar com configurações mais relaxadas
                solver.options['acceptable_tol'] = 1e-3
                solver.options['constr_viol_tol'] = 1e-3
                solver.options['acceptable_constr_viol_tol'] = 1e-2
                solver.options['max_iter'] = 200
                
                results = solver.solve(model, tee=True)
                sucesso = str(results.solver.termination_condition) in ['optimal', 'locallyOptimal', 'userInterrupt', 'acceptable']
            
            resultado = ResultadoOPF(
                sucesso=sucesso,
                custo_total=pyo.value(model.objetivo) if sucesso else 0.0,
                iteracoes=results.solver.iterations if hasattr(results.solver, 'iterations') else 0
            )
            
            if sucesso:
                # Calcular déficit total
                deficit_p_total = 0.0
                deficit_q_total = 0.0
                custo_deficit = 0.0
                custo_desvio = 0.0
                custo_geracao = 0.0
                custo_perdas = 0.0
                custo_curtailment = 0.0
                
                for i in model.BARRAS:
                    deficit_p_val = pyo.value(model.DEFICIT_P[i])
                    deficit_q_val = pyo.value(model.DEFICIT_Q[i])
                    deficit_p_total += deficit_p_val
                    deficit_q_total += deficit_q_val
                    custo_deficit += self.peso_deficit * deficit_p_val
                    custo_deficit += self.peso_deficit * 0.5 * deficit_q_val
                
                print(f"\n=== RESULTADOS DO OPF para hora {hora} ===")
                print(f"Déficit P total: {deficit_p_total:.6f} pu ({deficit_p_total * self.S_base:.2f} MW)")
                print(f"Déficit Q total: {deficit_q_total:.6f} pu ({deficit_q_total * self.S_base:.2f} MVAr)")
                
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
                        custo_coef = ger.get('custo_var_USD_pu', 1.0)
                        custo_geracao += custo_coef * Pg_util_val
                
                for i in model.BARRAS:
                    V_ref = resultados_pf['V_mag'][i]
                    V_val = pyo.value(model.V[i])
                    custo_desvio += (V_val - V_ref)**2
                
                for linha_idx in model.LINHAS:
                    custo_perdas += pyo.value(model.P_perda[linha_idx])
                
                for idx in self.geradores_eolicos_idx:
                    ger = self.dados['GERADORES'][idx]
                    custo_curtailment_coef = ger.get('custo_curtailment_USD_pu', 1000.0)
                    custo_curtailment += custo_curtailment_coef * pyo.value(model.curtailment[idx])
            
                # Armazenar componentes (incluindo déficit)
                resultado.custo_deficit = custo_deficit
                resultado.custo_desvio = custo_desvio * self.peso_desvio
                resultado.custo_geracao = custo_geracao * self.peso_custo_ger
                resultado.custo_perdas = custo_perdas * self.peso_perdas
                resultado.custo_curtailment = custo_curtailment
                
                # Preencher resultados
                resultado.V_mag = [pyo.value(model.V[i]) for i in model.BARRAS]
                resultado.V_ang = [pyo.value(model.theta[i]) for i in model.BARRAS]
                
                resultado.P_gerado = [0.0] * self.n_barras
                resultado.Q_gerado = [0.0] * self.n_barras
                
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
                        'P_min': ger['PGERmin_pu'],
                        'P_max': ger['PGERmax_pu'],
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
                        custo_coef = ger.get('custo_var_USD_pu', 1.0)
                        detalhes['custo_coef'] = custo_coef
                        detalhes['custo_total'] = custo_coef * Pg_util_val
                        detalhes['percentual_uso'] = ((Pg_util_val - ger['PGERmin_pu']) / 
                                                    (ger['PGERmax_pu'] - ger['PGERmin_pu']) * 100 
                                                    if ger['PGERmax_pu'] > ger['PGERmin_pu'] else 0)
                    
                    resultado.detalhes_geradores[ger['ID_Gerador']] = detalhes
                
                # Carregar dados de carga do PF (já em pu)
                resultado.P_carga = resultados_pf['P_carga']
                resultado.Q_carga = resultados_pf['Q_carga']
                
                # Calcular déficit total
                resultado.deficit_total = deficit_p_total
                
                # Calcular fluxos nas linhas e perdas (em pu)
                for linha_idx in range(self.n_linhas):
                    linha = self.dados['LINHAS'][linha_idx]
                    i = self.barra_para_indice[linha['ID_Barra_Origem']]
                    j = self.barra_para_indice[linha['ID_Barra_Destino']]
                    
                    P_ij, Q_ij, P_perda = self.calcular_fluxo_linha(
                        resultado.V_mag, resultado.V_ang, linha_idx
                    )
                    
                    linha_id = f"{linha['ID_Barra_Origem']}-{linha['ID_Barra_Destino']}"
                    limite_pu = linha.get('LIM_Fluxo_pu', 1.0)
                    resultado.fluxos_linhas[linha_id] = {
                        'P_ij': P_ij,
                        'Q_ij': Q_ij,
                        'P_perda': P_perda,
                        'S_ij': math.sqrt(P_ij**2 + Q_ij**2),
                        'limite': limite_pu,
                        'percentual_uso': abs(P_ij) / limite_pu * 100 
                                        if limite_pu > 0 else 0
                    }
                    
                    resultado.perdas_ativas += abs(P_perda)
                    resultado.perdas_reativas += abs(Q_ij)
            
            return resultado
            
        except Exception as e:
            print(f"Erro ao resolver OPF para hora {hora}: {e}")
            import traceback
            traceback.print_exc()
            return ResultadoOPF(sucesso=False, custo_total=0.0)

    def resolver_opf_multiplas_horas(self, db_path: str) -> Dict[int, ResultadoOPF]:
        """
        Resolve OPF para todas as horas disponíveis no banco de dados
        """
        resultados = {}
        
        # Primeiro, descobrir quantas horas estão disponíveis
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT hora FROM resultados_fluxo ORDER BY hora')
        horas_disponiveis = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"\nHoras disponíveis para processamento: {horas_disponiveis}")
        
        for hora in horas_disponiveis:
            print(f"\n{'='*80}")
            print(f"=== Processando OPF para hora {hora} ===")
            print(f"{'='*80}")
            
            try:
                resultados_pf = self.carregar_resultados_pf(db_path, hora)
                resultado_opf = self.resolver_opf_pyomo(resultados_pf, hora)
                
                if resultado_opf.sucesso:
                    resultados[hora] = resultado_opf
                    print(f"\n✓ OPF para hora {hora} resolvido com sucesso")
                    print(f"  Custo total: {resultado_opf.custo_total:.6f}")
                    print(f"    - Custo déficit: {resultado_opf.custo_deficit:.6f}")
                    print(f"    - Custo desvio: {resultado_opf.custo_desvio:.6f}")
                    print(f"    - Custo geração: {resultado_opf.custo_geracao:.6f}")
                    print(f"    - Custo perdas: {resultado_opf.custo_perdas:.6f}")
                    print(f"    - Custo curtailment: {resultado_opf.custo_curtailment:.6f}")
                    print(f"  Perdas ativas: {resultado_opf.perdas_ativas:.4f} pu ({resultado_opf.perdas_ativas * self.S_base:.2f} MW)")
                    print(f"  Perdas reativas: {resultado_opf.perdas_reativas:.4f} pu ({resultado_opf.perdas_reativas * self.S_base:.2f} MVAr)")
                    print(f"  Curtailment total: {resultado_opf.curtailment_total:.4f} pu ({resultado_opf.curtailment_total * self.S_base:.2f} MW)")
                    print(f"  Déficit total: {resultado_opf.deficit_total:.4f} pu ({resultado_opf.deficit_total * self.S_base:.2f} MW)")
                else:
                    print(f"\n✗ Falha ao resolver OPF para hora {hora}")
                    
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
            deficit_total REAL,
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
                (hora, sucesso, custo_total, custo_desvio, custo_geracao, custo_perdas, 
                 custo_curtailment, iteracoes, perdas_ativas, perdas_reativas,
                 curtailment_total, deficit_total,
                 tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
                 P_carga_json, Q_carga_json, fluxos_json, geradores_json, eolicos_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    float(resultado.deficit_total),
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
        
        print(f"\n✅ Resultados OPF salvos em: {db_path}")
    
    def gerar_tabela_resultados(self, resultados: Dict[int, ResultadoOPF]):
        """
        Gera tabela com resultados por hora
        """
        print("\n" + "="*120)
        print("TABELA DE RESULTADOS OPF")
        print("="*120)
        print(f"{'Hora':<6} | {'Carga Total':<12} | {'G Conv':<10} | {'GWD':<10} | "
              f"{'Curtailment':<12} | {'Déficit':<10} | {'Perdas Ativas':<14} | {'Perdas Reativas':<15}")
        print("-"*120)
        
        horas_ordenadas = sorted(resultados.keys())
        
        for hora in horas_ordenadas:
            if hora in resultados and resultados[hora].sucesso:
                r = resultados[hora]
                
                # Calcular carga total
                carga_total = sum(r.P_carga)
                
                # Extrair valores dos geradores
                g_conv_total = 0.0
                gwd_total = 0.0
                
                for ger_id, detalhes in r.detalhes_geradores.items():
                    if detalhes['tipo'] == 'GWD':
                        gwd_total += detalhes['P_ger']
                    else:
                        g_conv_total += detalhes['P_ger']
                
                print(f"{hora:02d}:00 | {carga_total:<12.4f} | {g_conv_total:<10.4f} | "
                      f"{gwd_total:<10.4f} | {r.curtailment_total:<12.4f} | {r.deficit_total:<10.4f} | "
                      f"{r.perdas_ativas:<14.4f} | {r.perdas_reativas:<15.4f}")
        
        print("="*120)
    
    def gerar_relatorio(self, resultado: ResultadoOPF, hora: int):
        """
        Gera relatório detalhado dos resultados para uma hora específica
        """
        if not resultado.sucesso:
            print(f"\nOPF para hora {hora} não convergiu!")
            return
            
        print(f"\n{'='*80}")
        print(f"RELATÓRIO DO FLUXO DE POTÊNCIA ÓTIMO (OPF) - Hora {hora}")
        print(f"{'='*80}")
        
        print(f"\nStatus: {'SUCESSO' if resultado.sucesso else 'FALHA'}")
        print(f"Custo total: {resultado.custo_total:.6f}")
        print(f"Componentes do custo:")
        print(f"  - Déficit: {resultado.custo_deficit:.6f} (peso: {self.peso_deficit})")
        print(f"  - Desvio: {resultado.custo_desvio:.6f} (peso: {self.peso_desvio})")
        print(f"  - Geração: {resultado.custo_geracao:.6f} (peso: {self.peso_custo_ger})")
        print(f"  - Perdas: {resultado.custo_perdas:.6f} (peso: {self.peso_perdas})")
        print(f"  - Curtailment: {resultado.custo_curtailment:.6f} USD")
        print(f"Iterações: {resultado.iteracoes}")
        print(f"Perdas ativas: {resultado.perdas_ativas:.6f} pu ({resultado.perdas_ativas * self.S_base:.2f} MW)")
        print(f"Perdas reativas: {resultado.perdas_reativas:.6f} pu ({resultado.perdas_reativas * self.S_base:.2f} MVAr)")
        print(f"Curtailment total: {resultado.curtailment_total:.6f} pu ({resultado.curtailment_total * self.S_base:.2f} MW)")
        print(f"Déficit total: {resultado.deficit_total:.6f} pu ({resultado.deficit_total * self.S_base:.2f} MW)")
        
        print(f"\n{'-'*80}")
        print("GERADORES EÓLICOS (GWD)")
        print("-"*80)
        print(f"{'Gerador':<10} {'Barra':<6} {'Disponível':<12} {'Utilizado':<12} {'Curtailment':<12} {'Curt%':<10}")
        print("-"*80)
        
        for ger_id, detalhes in resultado.geradores_eolicos.items():
            print(f"{ger_id:<10} {detalhes['barra']:<6} {detalhes['P_disponivel']:<12.4f} "
                  f"{detalhes['P_ger']:<12.4f} {detalhes.get('curtailment', 0):<12.4f} "
                  f"{detalhes.get('curtailment_percent', 0):<10.1f}")
        
        print(f"\n{'-'*80}")
        print("GERADORES CONVENCIONAIS")
        print("-"*80)
        print(f"{'Gerador':<10} {'Barra':<6} {'P (pu)':<12} {'Q (pu)':<12} {'Min':<8} {'Max':<8} {'Uso %':<10}")
        print("-"*80)
        
        for ger_id, detalhes in resultado.detalhes_geradores.items():
            if detalhes['tipo'] != 'GWD':
                print(f"{ger_id:<10} {detalhes['barra']:<6} {detalhes['P_ger']:<12.4f} "
                      f"{detalhes['Q_ger']:<12.4f} {detalhes['P_min']:<8.4f} "
                      f"{detalhes['P_max']:<8.4f} {detalhes.get('percentual_uso', 0):<10.1f}")
        
        print(f"\n{'-'*80}")
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
    with open('DATA/input/B6L8_BASE.json', 'r') as f:
        dados_rede = json.load(f)
    
    # 2. Inicializar sistema OPF
    opf_system = OPFNaoLinear(dados_rede)
    
    # 3. Especificar banco de dados com resultados PF
    db_pf = 'DATA/SMA/resultados_PF.db' 
    
    # 4. Resolver OPF para múltiplas horas (todas disponíveis)
    resultados_opf = opf_system.resolver_opf_multiplas_horas(db_pf)
    
    # 5. Salvar resultados
    opf_system.salvar_resultados_opf(resultados_opf, 'DATA/SMA/resultados_OPF.db')
    
    # 6. Gerar tabela de resultados
    if resultados_opf:
        opf_system.gerar_tabela_resultados(resultados_opf)
    
    # 7. Gerar relatórios para as primeiras 3 horas (se disponível)
    for hora in sorted(resultados_opf.keys())[:3]:
        if resultados_opf[hora].sucesso:
            opf_system.gerar_relatorio(resultados_opf[hora], hora)
    
    return resultados_opf

# Executar a etapa 3
if __name__ == "__main__":
    print("=" * 80)
    print("ETAPA 3: FLUXO DE POTÊNCIA ÓTIMO NÃO LINEAR COM GWD E CURTAILMENT")
    print("=" * 80)
    print("Função objetivo: Minimizar [déficit + desvio + custo geração + perdas + curtailment]")
    print(f"Pesos: Déficit={10000.0}, Desvio={10.0}, Custo Geração={1.0}, Perdas={0.5}, Curtailment={1000.0}")
    print("=" * 80)
    
    try:
        resultados = executar_etapa_3()
        
        if resultados:
            horas_com_sucesso = sum(1 for r in resultados.values() if r.sucesso)
            print(f"\n{'='*80}")
            print(f"ETAPA 3 CONCLUÍDA: {horas_com_sucesso}/{len(resultados)} horas processadas com sucesso")
            print(f"{'='*80}")
        else:
            print("\n✗ Nenhum resultado obtido na Etapa 3")
        
    except Exception as e:
        print(f"\nErro na execução da Etapa 3: {e}")
        import traceback
        traceback.print_exc()