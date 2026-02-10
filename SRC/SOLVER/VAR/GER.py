from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Generator:
    """Classe representando um gerador"""
    id: int
    barra_id: int
    tipo: str  # "TERMICA", "GWD", "CURTAILMENT", "DEFICIT"
    p_min_pu: float
    p_max_pu: float
    q_min_pu: float = 0.0
    q_max_pu: float = 0.0
    custo_var_pu: float = 0.0
    custo_curtailment_pu: float = 1000.0
    fator_vento: float = 1.0
    status: int = 1
    
    def get_capacidade_efetiva(self) -> float:
        if self.tipo == "GWD":
            return self.p_max_pu * self.fator_vento
        return self.p_max_pu

@dataclass
class GeneratorCollection:
    """Coleção de geradores do sistema"""
    generators: List[Generator] = field(default_factory=list)
    gen_dict: Dict[int, Generator] = field(default_factory=dict, init=False)
    gen_by_bus: Dict[int, List[Generator]] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.gen_dict = {gen.id: gen for gen in self.generators}
        self.gen_by_bus = {}
        for gen in self.generators:
            self.gen_by_bus.setdefault(gen.barra_id, []).append(gen)
    
    def get_generator_ids(self) -> List[int]:
        return [gen.id for gen in self.generators]
    
    def get_generators_by_type(self, tipo: str) -> List[Generator]:
        return [gen for gen in self.generators if gen.tipo == tipo]
    
    def get_generators_by_bus(self, bus_id: int) -> List[Generator]:
        return self.gen_by_bus.get(bus_id, [])
    
    def update_wind_factors(self, fatores_vento: Dict[int, float]):
        for gen in self.generators:
            if gen.tipo == "GWD" and gen.id in fatores_vento:
                gen.fator_vento = fatores_vento[gen.id]
    
    def get_pg_min_vector(self) -> np.ndarray:
        return np.array([gen.p_min_pu for gen in self.generators])
    
    def get_pg_max_vector(self) -> np.ndarray:
        return np.array([gen.get_capacidade_efetiva() for gen in self.generators])
    
    def get_custo_vector(self) -> np.ndarray:
        return np.array([gen.custo_var_pu for gen in self.generators])
    
    def get_barra_ids_vector(self) -> np.ndarray:
        return np.array([gen.barra_id for gen in self.generators])