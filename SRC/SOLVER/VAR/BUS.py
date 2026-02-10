from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Bus:
    """Classe representando uma barra do sistema"""
    id: int
    tipo: str  # "Slack", "PV", "PQ"
    v_base: float
    p_carga_pu: float = 0.0
    q_carga_pu: float = 0.0
    v_min: float = 0.95
    v_max: float = 1.05
    ang_min: float = -np.pi
    ang_max: float = np.pi
    shunt_g: float = 0.0
    shunt_b: float = 0.0

@dataclass
class BusCollection:
    """Coleção de barras do sistema"""
    buses: List[Bus] = field(default_factory=list)
    bus_dict: Dict[int, Bus] = field(default_factory=dict, init=False)
    idx_map: Dict[int, int] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.bus_dict = {bus.id: bus for bus in self.buses}
        self.idx_map = {bus.id: idx for idx, bus in enumerate(self.buses)}
    
    def get_bus_ids(self) -> List[int]:
        return [bus.id for bus in self.buses]
    
    def get_bus_by_id(self, bus_id: int) -> Optional[Bus]:
        return self.bus_dict.get(bus_id)
    
    def get_pd_vector(self) -> np.ndarray:
        return np.array([bus.p_carga_pu for bus in self.buses])
    
    def get_qd_vector(self) -> np.ndarray:
        return np.array([bus.q_carga_pu for bus in self.buses])
    
    def get_slack_bus(self) -> Optional[Bus]:
        for bus in self.buses:
            if bus.tipo == "Slack":
                return bus
        return None