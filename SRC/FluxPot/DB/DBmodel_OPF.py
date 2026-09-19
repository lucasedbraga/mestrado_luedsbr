from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class MultiDayOPFSnapshotResult:
    dia: int
    hora: int
    sucesso: bool

    PGER: List[float] = field(default_factory=list)
    PGWIND_disponivel: List[float] = field(default_factory=list)
    PGWIND: List[float] = field(default_factory=list)
    CURTAILMENT: List[float] = field(default_factory=list)
    SOC_init: List[float] = field(default_factory=list)
    BESS_operation: List[float] = field(default_factory=list)
    SOC_atual: List[float] = field(default_factory=list)
    DEFICIT: List[float] = field(default_factory=list)

    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)

    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS_BARRA: List[float] = field(default_factory=list)

    mensagem: str = ""
    timestamp: Optional[datetime] = None
    tempo_execucao: float = 0.0

@dataclass
class MultiDayOPFSnapshotResult:
    dia: int
    hora: int
    sucesso: bool
    PGER: List[float] = field(default_factory=list)
    PGWIND_disponivel: List[float] = field(default_factory=list)
    PGWIND: List[float] = field(default_factory=list)
    CURTAILMENT: List[float] = field(default_factory=list)
    SOC_init: List[float] = field(default_factory=list)
    BESS_operation: List[float] = field(default_factory=list)
    SOC_atual: List[float] = field(default_factory=list)
    DEFICIT: List[float] = field(default_factory=list)
    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)
    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS_BARRA: List[float] = field(default_factory=list)
    mensagem: str = ""
    timestamp: Optional[datetime] = None
    tempo_execucao: float = 0.0

@dataclass
class MultiDayOPFResult:
    snapshots: List[MultiDayOPFSnapshotResult] = field(default_factory=list)
    sucesso_global: bool = True
    mensagem_global: str = ""

@dataclass
class MultiDayOPFResult:
    snapshots: List[MultiDayOPFSnapshotResult] = field(default_factory=list)
    sucesso_global: bool = True
    mensagem_global: str = ""

@dataclass
class OPF_SnapshotResult:
    """Resultado de um único instante (hora) da simulação multi-período."""
    dia: int
    hora: int
    sucesso: bool

    # Dia da semana
    dia_semana: int = 0
    dia_semana_nome: str = ""

    PLOAD: List[float] = field(default_factory=list)
    QLOAD: List[float] = field(default_factory=list)
    # Geração térmica (lista por gerador)
    PGER: List[float] = field(default_factory=list)
    QGER: List[float] = field(default_factory=list)

    # Geração eólica e corte (listas por gerador eólico)
    PGWIND_disponivel: List[float] = field(default_factory=list)  # disponível no período
    PGWIND: List[float] = field(default_factory=list)             # gerado
    CURTAILMENT: List[float] = field(default_factory=list)        # cortado

    # Baterias (listas por bateria, na ordem das barras com bateria)
    SOC_init: List[float] = field(default_factory=list)      # SOC no início do período
    SOC_atual: List[float] = field(default_factory=list)     # SOC no final do período
    BESS_operation: List[float] = field(default_factory=list)  # potência líquida (descarga - carga)

    # Déficit (por barra)
    DEFICIT: List[float] = field(default_factory=list)

    # Tensão, ângulo, fluxo (por barra/linha)
    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)
    REATIVO_LIN: List[float] = field(default_factory=list)

    # Custos e perdas
    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS_TOTAIS: List[float] = field(default_factory=list)

    # Metadados
    mensagem: str = ""
    timestamp: Optional[datetime] = None
    tempo_execucao: float = 0.0

@dataclass
class TimeCoupled_OPF_Result:
    """Conjunto de snapshots para toda a simulação multi-período."""
    snapshots: List[OPF_SnapshotResult] = field(default_factory=list)
    sucesso_global: bool = True
    mensagem_global: str = ""
