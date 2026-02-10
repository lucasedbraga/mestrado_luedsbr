from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Line:
    """Classe representando uma linha de transmissão"""
    id: int
    de_barra_id: int
    para_barra_id: int
    r_pu: float
    x_pu: float
    b_shunt_pu: float = 0.0
    limite_fluxo_pu: float = 9999.0
    custo_investimento: float = 0.0
    status: int = 1
    
    def get_admittance(self) -> complex:
        """Retorna admitância série da linha"""
        return 1 / complex(self.r_pu, self.x_pu)

@dataclass
class LineCollection:
    """Coleção de linhas de transmissão"""
    lines: List[Line] = field(default_factory=list)
    line_dict: Dict[int, Line] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.line_dict = {line.id: line for line in self.lines}
    
    def get_line_ids(self) -> List[int]:
        return [line.id for line in self.lines]
    
    def get_line_by_id(self, line_id: int) -> Optional[Line]:
        return self.line_dict.get(line_id)
    
    def get_r_vector(self) -> np.ndarray:
        return np.array([line.r_pu for line in self.lines])
    
    def get_x_vector(self) -> np.ndarray:
        return np.array([line.x_pu for line in self.lines])
    
    def get_b_shunt_vector(self) -> np.ndarray:
        return np.array([line.b_shunt_pu for line in self.lines])
    
    def get_limite_fluxo_vector(self) -> np.ndarray:
        return np.array([line.limite_fluxo_pu for line in self.lines])
    
    def get_incidence_matrix(self, bus_ids: List[int]) -> np.ndarray:
        """Retorna matriz de incidência n_bus x n_line"""
        n_bus = len(bus_ids)
        n_line = len(self.lines)
        incidence = np.zeros((n_bus, n_line))
        
        bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        
        for j, line in enumerate(self.lines):
            if line.de_barra_id in bus_id_to_idx:
                i = bus_id_to_idx[line.de_barra_id]
                incidence[i, j] = 1
            if line.para_barra_id in bus_id_to_idx:
                i = bus_id_to_idx[line.para_barra_id]
                incidence[i, j] = -1
        
        return incidence