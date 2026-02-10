from pyomo.environ import ConcreteModel, Block, Var, Constraint, NonNegativeReals, Objective, minimize, ConstraintList
from SOLVER.models.DC_OPF_Model import OPF_Model
import numpy as np
from SOLVER.FOB.economic_dispatch import OPF_Solver
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class MultiDayOPFSnapshotResult:
    dia: int
    hora: int
    sucesso: bool
    PG: List[float]
    ANG: List[float]
    FLUXO: List[float]
    DEFICIT: List[float]
    CURTAILMENT: List[float]
    custo_total: float
    deficit_total: float
    curtailment_total: float
    cmo_total: float
    perdas: float
    SOC: List[float] = field(default_factory=list)
    BATTERY_OPERATION: List[str] = field(default_factory=list)
    BATTERY_POWER: List[float] = field(default_factory=list)
    mensagem: str = ""
    timestamp: Optional[datetime] = None
    iteracoes: int = 1
    tempo_execucao: float = 0.0

@dataclass
class MultiDayOPFResult:
    snapshots: List[MultiDayOPFSnapshotResult] = field(default_factory=list)
    sucesso_global: bool = True
    mensagem_global: str = ""

class MultiDayOPFModel:
    def __init__(self, sistema, n_horas=24, n_dias=1):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.model = None  # Só será criado em build()
        self._solved = False
        self._raw_results = None

    def build(self):
        """Constrói o modelo multi-dia/hora com acoplamento temporal"""
        self.model = ConcreteModel()
        self.model.snapshots = Block(range(self.n_dias), rule=self._snapshot_block_rule)
        self._add_temporal_coupling()
        self._add_objective()
        self._solved = False
        self._raw_results = None

    def _snapshot_block_rule(self, b, dia):
        b.hours = Block(range(self.n_horas), rule=lambda bh, h: self._build_hour_block(bh, h, dia))

    def _build_hour_block(self, bh, h, dia):
        # Instancia o modelo DC_OPF para cada hora
        opf = OPF_Model()
        opf.build(self.sistema, considerar_perdas=True)
        # Adiciona função objetivo via economic_dispatch
        solver = OPF_Solver(self.sistema)
        solver.add_objective(opf)
        # Desativa objetivo do submodelo para evitar múltiplos objetivos ativos
        if hasattr(opf.model, 'obj'):
            opf.model.obj.deactivate()
        bh.opf = opf.model
        # Adiciona variáveis de SOC para acoplamento
        bh.SOC = Var(range(len(self.sistema.BARRAS_COM_BATERIA)), within=NonNegativeReals)

    def _add_temporal_coupling(self):
        m = self.model
        # Cria restrições de acoplamento temporal para todas as baterias, horas e dias
        m.soc_coupling = ConstraintList()
        for dia in range(self.n_dias):
            for h in range(self.n_horas-1):
                for b_idx in range(len(self.sistema.BARRAS_COM_BATERIA)):
                    block_curr = m.snapshots[dia].hours[h]
                    block_next = m.snapshots[dia].hours[h+1]
                    m.soc_coupling.add(block_next.SOC[b_idx] == block_curr.SOC[b_idx])
            if dia < self.n_dias-1:
                for b_idx in range(len(self.sistema.BARRAS_COM_BATERIA)):
                    block_last = m.snapshots[dia].hours[self.n_horas-1]
                    block_first_next = m.snapshots[dia+1].hours[0]
                    m.soc_coupling.add(block_first_next.SOC[b_idx] == block_last.SOC[b_idx])

    def _add_objective(self):
        m = self.model
        # Soma os objetivos de cada bloco (hora/dia)
        def total_obj_rule(_m):
            return sum(
                getattr(m.snapshots[dia].hours[h].opf, 'obj', 0)
                for dia in range(self.n_dias)
                for h in range(self.n_horas)
            )
        m.total_obj = Objective(rule=total_obj_rule, sense=minimize)

    def solve(self, solver_name='glpk'):
        """Resolve o modelo multi-dia/hora"""
        from pyomo.opt import SolverFactory
        if self.model is None:
            raise RuntimeError("O modelo deve ser construído com build() antes de resolver.")
        solver = SolverFactory(solver_name)
        self._raw_results = solver.solve(self.model, tee=False)
        self._solved = True
        return self._raw_results

    def solve_multiday_sequencial(self, solver_name='glpk',  fator_carga=None, fator_vento=None,soc_inicial=None):
        """
        Resolve cada hora/dia separadamente, conectando via SOC da bateria.
        Para cada simulação, sorteia fator de vento e demanda.
        df_wind: DataFrame com coluna 'wind_factor' (para sorteio)
        demanda_mu: valor médio da demanda (pu)
        demanda_sigma: desvio padrão da demanda (pu)
        """
        import numpy as np
        import pandas as pd
        resultados = []
        n_bat = len(self.sistema.BARRAS_COM_BATERIA)
        soc_atual = soc_inicial if soc_inicial is not None else [0.0]*n_bat
        # Salva os valores originais para restaurar depois
        PLOAD_original = self.sistema.PLOAD.copy()
        PGMAX_EFETIVO_original = self.sistema.PGMAX_EFETIVO.copy() if hasattr(self.sistema, 'PGMAX_EFETIVO') else None
        for dia in range(self.n_dias):
            for h in range(self.n_horas):
                # Atualiza carga para o valor sorteado dessa hora/dia
                self.sistema.PLOAD = PLOAD_original * fator_carga[dia, h] if fator_carga is not None else PLOAD_original
                # Atualiza vento para o valor sorteado dessa hora/dia
                if fator_vento is not None:
                    for g_idx in self.sistema.BAR_GWD:
                        self.sistema.PGMAX_EFETIVO[g_idx] = self.sistema.PGMAX[g_idx] * fator_vento[dia, h] if fator_vento is not None else self.sistema.PGMAX[g_idx]
                # Atualiza SOC inicial
                self.sistema.BATTERY_INITIAL_SOC = soc_atual

                opf = OPF_Model()
                opf.build(self.sistema, considerar_perdas=True)
                solver = OPF_Solver(self.sistema)
                solver.add_objective(opf)
                from pyomo.opt import SolverFactory
                solver_pyomo = SolverFactory(solver_name)
                results_pyomo = solver_pyomo.solve(opf.model, tee=False)
                res = opf.extract_results(results_pyomo)
                soc_atual = res.SOC.copy() if res.SOC else 0.0
                resultados.append(MultiDayOPFSnapshotResult(
                    dia=dia,
                    hora=h,
                    sucesso=res.sucesso,
                    PG=res.PG,
                    ANG=res.ANG,
                    FLUXO=res.FLUXO,
                    DEFICIT=res.DEFICIT,
                    CURTAILMENT=res.CURTAILMENT,
                    custo_total=res.custo_total,
                    deficit_total=sum(res.DEFICIT) if res.DEFICIT else 0.0,
                    curtailment_total=res.curtailment_total,
                    cmo_total=res.cmo_total,
                    perdas=res.perdas,
                    SOC=res.SOC,
                    BATTERY_OPERATION=res.BATTERY_OPERATION,
                    BATTERY_POWER=res.BATTERY_POWER,
                    mensagem=res.mensagem,
                    timestamp=res.timestamp,
                    iteracoes=getattr(res, 'iteracoes', 1),
                    tempo_execucao=getattr(res, 'tempo_execucao', 0.0)
                ))
        # Restaura os valores originais do sistema
        self.sistema.PLOAD = PLOAD_original
        if PGMAX_EFETIVO_original is not None:
            self.sistema.PGMAX_EFETIVO = PGMAX_EFETIVO_original
        sucesso_global = all(r.sucesso for r in resultados)
        mensagem_global = "OK" if sucesso_global else "Alguns snapshots falharam"
        return MultiDayOPFResult(snapshots=resultados, sucesso_global=sucesso_global, mensagem_global=mensagem_global)

    def extract_results(self) -> MultiDayOPFResult:
        """Extrai resultados estruturados de todos os snapshots (dia/hora)"""
        if not self._solved or self.model is None:
            raise RuntimeError("O modelo deve ser resolvido antes de extrair resultados.")
        resultados = []
        for dia in range(self.n_dias):
            for h in range(self.n_horas):
                block = self.model.snapshots[dia].hours[h]
                opf_model = block.opf
                try:
                    PG = [float(opf_model.PG[g].value) for g in opf_model.GENERATORS] if hasattr(opf_model, 'PG') else []
                    ANG = [float(opf_model.ANG[b].value) for b in opf_model.BUSES] if hasattr(opf_model, 'ANG') else []
                    FLUXO = [float(opf_model.FLUXO[e].value) for e in opf_model.LINES] if hasattr(opf_model, 'FLUXO') else []
                    DEFICIT = [float(opf_model.DEFICIT[b].value) for b in opf_model.BUSES] if hasattr(opf_model, 'DEFICIT') else []
                    CURTAILMENT = [float(opf_model.CURTAILMENT[g].value) for g in opf_model.GWD_GENERATORS] if hasattr(opf_model, 'CURTAILMENT') else []
                    custo_total = float(opf_model.obj()) if hasattr(opf_model, 'obj') else 0.0
                    deficit_total = sum(DEFICIT)
                    curtailment_total = sum(CURTAILMENT)
                    cmo_total = 0.0  # Pode ser ajustado se dual disponível
                    perdas = 0.0  # Pode ser ajustado se perdas disponíveis
                    SOC = [block.SOC[idx].value if hasattr(block, 'SOC') else None for idx in range(len(self.sistema.BARRAS_COM_BATERIA))]
                    operacao = []
                    potencia_bess_mw = []
                    for idx in range(len(self.sistema.BARRAS_COM_BATERIA)):
                        charge = opf_model.CHARGE[idx].value if hasattr(opf_model, 'CHARGE') else 0.0
                        discharge = opf_model.DISCHARGE[idx].value if hasattr(opf_model, 'DISCHARGE') else 0.0
                        if charge > 0.01:
                            operacao.append('charge')
                        elif discharge > 0.01:
                            operacao.append('discharge')
                        else:
                            operacao.append('idle')
                        potencia_bess_mw.append(discharge - charge)
                    sucesso = True
                    mensagem = ""
                except Exception as e:
                    PG = ANG = FLUXO = DEFICIT = CURTAILMENT = SOC = operacao = potencia_bess_mw = []
                    custo_total = deficit_total = curtailment_total = cmo_total = perdas = 0.0
                    sucesso = False
                    mensagem = str(e)
                resultados.append(MultiDayOPFSnapshotResult(
                    dia=dia,
                    hora=h,
                    sucesso=sucesso,
                    PG=PG,
                    ANG=ANG,
                    FLUXO=FLUXO,
                    DEFICIT=DEFICIT,
                    CURTAILMENT=CURTAILMENT,
                    custo_total=custo_total,
                    deficit_total=deficit_total,
                    curtailment_total=curtailment_total,
                    cmo_total=cmo_total,
                    perdas=perdas,
                    SOC=SOC,
                    BATTERY_OPERATION=operacao,
                    BATTERY_POWER=potencia_bess_mw,
                    mensagem=mensagem,
                    timestamp=datetime.now(),
                    iteracoes=1,
                    tempo_execucao=0.0
                ))
        sucesso_global = all(r.sucesso for r in resultados)
        mensagem_global = "OK" if sucesso_global else "Alguns snapshots falharam"
        return MultiDayOPFResult(snapshots=resultados, sucesso_global=sucesso_global, mensagem_global=mensagem_global)