from pyomo.environ import *
import numpy as np
import time
from typing import Dict, Optional
from datetime import datetime

from SOLVER.models.DC_OPF_Model import OPF_Model, OPFResult

class OPF_Solver:
    """Solver genérico para OPF"""
    
    def __init__(self, sistema):
        """
        Args:
            sistema: Objeto com dados do sistema processados
                     (deve ter os atributos definidos em SistemaLoader)
        """
        self.sistema = sistema
        
    def build_model(self, considerar_perdas=False) -> OPF_Model:
        """Constrói modelo físico"""
        modelo = OPF_Model()
        modelo.build(self.sistema, considerar_perdas)
        return modelo
    
    def add_objective(self, modelo: OPF_Model):
        """Adiciona função objetivo ao modelo"""
        m = modelo.model
        
        # Custo de geração convencional
        custo_geracao = sum(
            m.PG[g] * self.sistema.CPG[g] 
            for g in m.GENERATORS 
            if g not in m.GWD_GENERATORS
        )
        
        # Custo de curtailment (penalidade alta)
        custo_curtailment = sum(
            m.CURTAILMENT[g] * 1000.0
            for g in m.GWD_GENERATORS
        )
        
        # Custo de déficit (penalidade muito alta)
        custo_deficit = sum(
            m.DEFICIT[b] * 5000.0
            for b in m.BUSES
        )
        
        # Custo de operação das baterias (se houver)
        custo_bateria = 0.0
        if hasattr(m, 'BATTERIES') and len(m.BATTERIES) > 0:
            custo_bateria = sum(
                (m.CHARGE[b] + m.DISCHARGE[b]) * getattr(self.sistema, 'BATTERY_COST', [0])[b]
                for b in m.BATTERIES
            )
        
        # Função objetivo total
        m.obj = Objective(
            expr=custo_geracao + custo_curtailment + custo_deficit + custo_bateria, 
            sense=minimize
        )
        
        return m.obj
    
    def solve_simple(self, considerar_perdas=False, solver_name='glpk', 
                    verbose=False) -> OPFResult:
        """Resolve OPF uma vez (sem iterações de perdas)"""
        start_time = time.time()
        
        try:
            # Construir modelo
            modelo = self.build_model(considerar_perdas)
            
            # Adicionar objetivo
            self.add_objective(modelo)
            
            # Configurar solver
            solver = SolverFactory(solver_name)
            
            # Resolver
            results = solver.solve(modelo.model, tee=verbose)
            
            # Extrair resultados
            resultado = modelo.extract_results(results, iteracoes=1)
            resultado.tempo_execucao = time.time() - start_time
            
            return resultado
            
        except Exception as e:
            print(f"Erro no OPF: {e}")
            import traceback
            traceback.print_exc()
            
            return OPFResult(
                sucesso=False,
                PG=[0.0] * self.sistema.NGER,
                ANG=[0.0] * self.sistema.NBAR,
                FLUXO=[0.0] * self.sistema.NLIN,
                DEFICIT=[0.0] * self.sistema.NBAR,
                CURTAILMENT=[0.0] * len(self.sistema.BAR_GWD),
                deficit_total=0.0,
                curtailment_total=0.0,
                custo_total=0.0,
                cmo_total=0.0,
                perdas=0.0,
                iteracoes=0,
                timestamp=datetime.now()
            )
    
    def solve_with_losses(self,
                          max_iter=20,
                          tol=1e-5,
                          solver_name='glpk',
                          verbose=False) -> OPFResult:
        """Resolve OPF com iterações para convergência de perdas"""
        start_time = time.time()
        
        try:
            # Construir modelo (com perdas)
            modelo = self.build_model(considerar_perdas=True)
            
            # Adicionar objetivo
            self.add_objective(modelo)
            
            # Configurar solver
            solver = SolverFactory(solver_name)
            
            # Habilitar duais se possível
            if solver_name == 'ipopt':
                modelo.model.dual = Suffix(direction=Suffix.IMPORT)
            
            # Inicializar perdas
            perdas_anteriores = np.zeros(self.sistema.NBAR)
            perdas_atual = np.zeros(self.sistema.NBAR)
            
            # Iterar para convergência
            convergiu = False
            iteracao = 0
            
            for iteracao in range(1, max_iter + 1):
                # Resolver
                results = solver.solve(modelo.model, tee=False)
                
                # Verificar se solução foi ótima
                if results.solver.termination_condition != TerminationCondition.optimal:
                    if verbose:
                        print(f"Iteração {iteracao}: Solver falhou")
                    break
                
                # Calcular novas perdas
                perdas_novas = modelo.calculate_losses()
                
                # Verificar convergência
                diferenca = np.max(np.abs(perdas_novas - perdas_atual))
                
                if verbose:
                    custo = value(modelo.model.obj)
                    print(f"Iteração {iteracao}: Custo=${custo:.2f} | "
                          f"Perdas={np.sum(perdas_novas)*self.sistema.SB:.2f} MW | "
                          f"Diferença={diferenca:.2e}")
                
                if diferenca < tol:
                    convergiu = True
                    perdas_atual = perdas_novas
                    if verbose:
                        print(f">> Convergência atingida na iteração {iteracao}")
                    break
                
                # Atualizar perdas no modelo
                perdas_anteriores = perdas_atual.copy()
                perdas_atual = perdas_novas
                modelo.update_losses(perdas_atual)
            
            # Extrair resultados
            if convergiu and results.solver.termination_condition == TerminationCondition.optimal:
                resultado = modelo.extract_results(results, iteracao)
                resultado.perdas = np.sum(perdas_atual)
                resultado.tempo_execucao = time.time() - start_time
                return resultado
            else:
                return OPFResult(
                    sucesso=False,
                    PG=[0.0] * self.sistema.NGER,
                    ANG=[0.0] * self.sistema.NBAR,
                    FLUXO=[0.0] * self.sistema.NLIN,
                    DEFICIT=[0.0] * self.sistema.NBAR,
                    CURTAILMENT=[0.0] * len(self.sistema.BAR_GWD),
                    custo_total=0.0,
                    cmo_total=0.0,
                    perdas=0.0,
                    iteracoes=iteracao,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            print(f"Erro no OPF com perdas: {e}")
            import traceback
            traceback.print_exc()
            
            return OPFResult(
                sucesso=False,
                PG=[0.0] * self.sistema.NGER,
                ANG=[0.0] * self.sistema.NBAR,
                FLUXO=[0.0] * self.sistema.NLIN,
                DEFICIT=[0.0] * self.sistema.NBAR,
                CURTAILMENT=[0.0] * len(self.sistema.BAR_GWD),
                custo_total=0.0,
                cmo_total=0.0,
                perdas=0.0,
                iteracoes=0,
                timestamp=datetime.now()
            )