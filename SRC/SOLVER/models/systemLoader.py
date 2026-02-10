import json
import numpy as np
from typing import Dict, List, Any

class SistemaLoader:
    """Carrega e processa dados do sistema a partir de JSON"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = None
        self.barras = []
        self.geradores_data = []
        self.demandas_data = []
        self.linhas = []
        self.baterias_data = []
        
        self.SB = 100.0  # Potência base (será atualizada)
        self.VB = 230.0  # Tensão base
        self.f_base = 60.0  # Frequência base
        self.ZB = 0.0  # Impedância base
        
        # Estruturas processadas
        self.bus_ids = []
        self.idx_map = {}
        self.indice_para_barra = {}
        self.NBAR = 0
        self.NLIN = 0
        self.slack_idx = 0
        
        # Arrays
        self.line_fr = []
        self.line_to = []
        self.x_line = np.array([])
        self.r_line = np.array([])
        self.FLIM = np.array([])
        
        self.NGER_ORIGINAL = 0
        self.BARPG_ORIGINAL = []
        self.BAR_GWD = []
        self.GER_TIPOS = []
        self.PGMIN_ORIGINAL = np.array([])
        self.PGMAX_ORIGINAL = np.array([])
        self.PGMIN_EFETIVO = np.array([])
        self.PGMAX_EFETIVO = np.array([])
        self.CPG_ORIGINAL = np.array([])
        
        self.PLOAD = np.array([])
        self.QLOAD = np.array([])
        
        # Processa o sistema
        self.carrega_sistema()
        self.processa_sistema()
    
    def carrega_sistema(self):
        """Carrega dados do JSON"""
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.barras = self.data["BARRAS"]
        self.geradores_data = self.data["GERADORES"]
        self.demandas_data = self.data["DEMANDAS"]
        self.linhas = self.data["LINHAS"]
        self.baterias_data = self.data.get("BATERIAS", [])
    
    def processa_sistema(self):
        """Processa todos os dados do sistema"""
        # Parâmetros base
        self.SB = self.data["S_base"]
        self.VB = self.data["V_base"]
        self.f_base = self.data["f_base"]
        self.ZB = (self.VB ** 2) / self.SB
        
        # Processa em PU
        self.processa_pu()
        
        # Processa barras
        self.processa_barras()
        
        # Processa linhas
        self.processa_linhas()
        
        # Processa geradores
        self.processa_geradores()
        
        # Processa carga
        self.processa_carga()
        
        # Processa DEF/GWD (curtailment e déficit)
        self.processa_def_gwd()

        # Processa baterias (se houver)
        if len(self.baterias_data) > 0:
            self.processa_baterias()
        else:
            # Inicializar atributos vazios mesmo sem baterias
            self.BARRAS_COM_BATERIA = []
            self.BATTERIES = []
            self.BATTERY_POWER_LIMIT = np.zeros(self.NBAR)
            self.BATTERY_POWER_OUT = np.zeros(self.NBAR)
            self.BATTERY_CAPACITY = np.zeros(self.NBAR)
            self.BATTERY_INITIAL_SOC = np.zeros(self.NBAR)
            self.BATTERY_COST = np.zeros(self.NBAR)


    
    def processa_pu(self):
        """Converte todos os valores para PU"""
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
            b["Q_carga_pu"] = b.get("Q_carga_MVAr", 0.0) / self.SB
        
        # Geradores
        for g in self.geradores_data:
            g["PGERmin_pu"] = g.get("PGERmin_MW", 0.0) / self.SB
            g["PGERmax_pu"] = g.get("PGERmax_MW", 0.0) / self.SB
            g["Qmin_pu"] = g.get("Qmin_MW", 0.0) / self.SB
            g["Qmax_pu"] = g.get("Qmax_MW", 0.0) / self.SB
            
            # Custos em USD/pu
            g["custo_var_pu"] = g.get("custo_var_USD_MW", 0.0) * self.SB
            g["custo_curtailment_pu"] = g.get("custo_curtailment_USD_MW", 100.0) * self.SB
        
        # Demandas
        for d in self.demandas_data:
            d["PLOAD_pu"] = d.get("PLOAD", 0.0) / self.SB
            d["QLOAD_pu"] = d.get("QLOAD_MW", 0.0) / self.SB
        
        # Linhas
        for l in self.linhas:
            if l.get("R_Unidade", "pu") != "pu":
                l["R"] = l["R"] / self.ZB
                l["X"] = l["X"] / self.ZB
            
            if "Bsh" in l and l.get("Bsh_Unidade", "pu") != "pu":
                l["Bsh"] = l["Bsh"] * self.ZB
            
            if l.get("LIM_Fluxo_Unidade", "MW") == "MW":
                l["Fmax_pu"] = l["LIM_Fluxo"] / self.SB
            else:
                l["Fmax_pu"] = l["LIM_Fluxo"]
    
    def processa_barras(self):
        """Processa dados das barras"""
        self.bus_ids = [b["ID_Barra"] for b in self.barras]
        self.idx_map = {id: i for i, id in enumerate(self.bus_ids)}
        self.indice_para_barra = {i: id for i, id in enumerate(self.bus_ids)}
        self.NBAR = len(self.bus_ids)
        
        # Identifica barra slack
        slack_list = [b for b in self.barras if b["tipo"] == "Slack"]
        if len(slack_list) != 1:
            raise ValueError("Deve haver exatamente 1 barra Slack")
        slack_id = slack_list[0]["ID_Barra"]
        self.slack_idx = self.idx_map[slack_id]
    
    def processa_linhas(self):
        """Processa dados das linhas"""
        self.NLIN = len(self.linhas)
        
        self.line_fr = []
        self.line_to = []
        self.x_line = np.zeros(self.NLIN)
        self.r_line = np.zeros(self.NLIN)
        self.FLIM = np.zeros(self.NLIN)
        
        for e, ln in enumerate(self.linhas):
            fr = ln["ID_Barra_Origem"]
            to = ln["ID_Barra_Destino"]
            self.line_fr.append(self.idx_map[fr])
            self.line_to.append(self.idx_map[to])
            
            self.x_line[e] = ln.get("X", 0.01)
            self.r_line[e] = ln.get("R", 0.001)
            self.FLIM[e] = ln.get("Fmax_pu", 1.0)
    
    def processa_geradores(self):
        """Processa dados dos geradores"""
        self.NGER_ORIGINAL = len(self.geradores_data)
        self.BARPG_ORIGINAL = []
        self.BAR_GWD = []
        self.GER_TIPOS = []
        
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            barra_idx = self.idx_map[id_barra]
            self.BARPG_ORIGINAL.append(barra_idx)
            tipo_ger = g.get("Tipo", "CONV")
            self.GER_TIPOS.append(tipo_ger)
            if tipo_ger == "GWD":
                self.BAR_GWD.append(i)
        
        # Inicializar arrays
        self.PGMIN_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        self.PGMIN_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.PGMAX_EFETIVO = np.zeros(self.NGER_ORIGINAL)
        self.CPG_ORIGINAL = np.zeros(self.NGER_ORIGINAL)
        
        for i, g in enumerate(self.geradores_data):
            self.PGMIN_ORIGINAL[i] = g.get("PGERmin_pu", 0.0)
            self.PGMAX_ORIGINAL[i] = g.get("PGERmax_pu", 1.0)
            self.CPG_ORIGINAL[i] = g.get("custo_var_pu", 50.0)
            self.PGMAX_EFETIVO[i] = self.PGMAX_ORIGINAL[i]
            self.PGMIN_EFETIVO[i] = self.PGMIN_ORIGINAL[i]
    
    def processa_carga(self):
        """Processa dados de carga"""
        self.PLOAD = np.zeros(self.NBAR)
        self.QLOAD = np.zeros(self.NBAR)
        
        # Carga das demandas
        for d in self.demandas_data:
            idx = self.idx_map[d["ID_Barra"]]
            self.PLOAD[idx] += d.get("PLOAD_pu", 0.0)
            self.QLOAD[idx] += d.get("QLOAD_pu", 0.0)
    
    def processa_def_gwd(self):
        """Processa deficit e curtailment"""
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        barras_com_gerador = set(self.BARPG_ORIGINAL)
        self.barras_PQ_sem_gerador = [b for b in barras_PQ 
                                    if self.idx_map[b["ID_Barra"]] not in barras_com_gerador]
        
        # Curtailment (todos os GWD)
        self.NGER_CURTAILMENT = len(self.BAR_GWD)
        self.BARPG_CURTAILMENT = []
        self.PGMIN_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        self.PGMAX_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        self.CPG_CURTAILMENT = np.zeros(self.NGER_CURTAILMENT)
        
        for i, gwd_idx in enumerate(self.BAR_GWD):
            if gwd_idx < len(self.geradores_data):
                ger = self.geradores_data[gwd_idx]
                barra_idx = self.idx_map[ger["ID_Barra"]]
                self.BARPG_CURTAILMENT.append(barra_idx)
                self.PGMIN_CURTAILMENT[i] = 0.0
                self.PGMAX_CURTAILMENT[i] = self.PGMAX_EFETIVO[gwd_idx]
                self.CPG_CURTAILMENT[i] = ger.get("custo_curtailment_pu", 1000.0)
        
        # Déficit (barras PQ sem geradores)
        self.NGER_DEFICIT = len(self.barras_PQ_sem_gerador)
        self.BARPG_DEFICIT = []
        self.PGMIN_DEFICIT = np.zeros(self.NGER_DEFICIT)
        self.PGMAX_DEFICIT = np.zeros(self.NGER_DEFICIT)
        self.CPG_DEFICIT = np.zeros(self.NGER_DEFICIT)
        
        for i, b in enumerate(self.barras_PQ_sem_gerador):
            idx = self.idx_map[b["ID_Barra"]]
            self.BARPG_DEFICIT.append(idx)
            self.PGMIN_DEFICIT[i] = 0.0
            self.PGMAX_DEFICIT[i] = self.PLOAD[idx]
            self.CPG_DEFICIT[i] = 5000.0 * self.SB
        
        # Combinar todos os geradores
        self.NGER = self.NGER_ORIGINAL + self.NGER_CURTAILMENT + self.NGER_DEFICIT
        self.BARPG = self.BARPG_ORIGINAL + self.BARPG_CURTAILMENT + self.BARPG_DEFICIT
        self.PGMIN = np.concatenate([self.PGMIN_EFETIVO, self.PGMIN_CURTAILMENT, self.PGMIN_DEFICIT])
        self.PGMAX = np.concatenate([self.PGMAX_EFETIVO, self.PGMAX_CURTAILMENT, self.PGMAX_DEFICIT])
        self.CPG = np.concatenate([self.CPG_ORIGINAL, self.CPG_CURTAILMENT, self.CPG_DEFICIT])
        
        self.GER_TIPOS_COMBINADO = self.GER_TIPOS + ["CURTAILMENT"] * self.NGER_CURTAILMENT + ["DEFICIT"] * self.NGER_DEFICIT
    
    def processa_baterias(self):
        """Processa dados de baterias"""
        # Inicializar atributos ANTES de usá-los
        self.BARRAS_COM_BATERIA = []  # Inicializar lista vazia
        
        # Inicializar arrays
        BATmax_in_base = np.zeros(self.NBAR)  # Valores BASE (sem dimensionamento)
        BATmax_out_base = np.zeros(self.NBAR)
        BATcapacidade_base = np.zeros(self.NBAR)
        BATarm_inicial_base = np.zeros(self.NBAR)
        
        # Inicializar atributos do sistema
        self.BATTERY_POWER_LIMIT = np.zeros(self.NBAR)
        self.BATTERY_POWER_OUT = np.zeros(self.NBAR)
        self.BATTERY_CAPACITY = np.zeros(self.NBAR)
        self.BATTERY_INITIAL_SOC = np.zeros(self.NBAR)
        self.BATTERY_COST = np.zeros(self.NBAR)  # Custo de operação das baterias

        for bat in self.baterias_data:
            id_barra = str(bat["ID_Barra"])
            if id_barra not in self.idx_map:
                print(f"  ⚠️  Bateria em barra {id_barra} não encontrada - ignorando")
                continue
            
            idx = self.idx_map[id_barra]
            self.BARRAS_COM_BATERIA.append(idx)
            
            # Converter para PU se necessário
            if "Pmax_carga_base_pu" in bat:
                p_max_carga = bat["Pmax_carga_base_pu"]
            else:
                p_max_carga = bat.get("Pmax_carga_MW", 0.0) / self.SB
            
            if "capacidade_base_pu" in bat:
                capacidade = bat["capacidade_base_pu"]
            else:
                capacidade = bat.get("capacidade_armazenamento_MWh", 0.0) / self.SB
            
            # Armazenar valores
            BATmax_in_base[idx] = p_max_carga
            BATmax_out_base[idx] = bat.get("Pmax_descarga_pu", p_max_carga)  # Usar carga como padrão
            BATcapacidade_base[idx] = capacidade
            BATarm_inicial_base[idx] = bat.get("SOC_inicial_pu", 0.5) * capacidade  # SOC 50% por padrão
            
            # Custo da bateria (penalidade por uso)
            self.BATTERY_COST[idx] = bat.get("custo_operacao_pu", 10.0)  # USD/pu

        # Atribuir arrays
        self.BATTERY_POWER_LIMIT = BATmax_in_base.copy()
        self.BATTERY_POWER_OUT = BATmax_out_base.copy()
        self.BATTERY_CAPACITY = BATcapacidade_base.copy()
        self.BATTERY_INITIAL_SOC = BATarm_inicial_base.copy()
        
        print(f"  ✓ Baterias processadas: {len(self.BARRAS_COM_BATERIA)} bateria(s) em {len(set(self.BARRAS_COM_BATERIA))} barra(s)")
        
        # Criar conjunto de baterias para o modelo
        self.BATTERIES = self.BARRAS_COM_BATERIA.copy()

    def atualizar_perfil_eolico(self, fator_vento: float):
        """Atualiza capacidade eólica com fator de vento"""
        for idx in self.BAR_GWD:
            if idx < len(self.PGMAX_EFETIVO):
                self.PGMAX_EFETIVO[idx] = self.PGMAX_ORIGINAL[idx] * fator_vento
        
        # Atualizar também no PGMAX_CURTAILMENT
        for i, gwd_idx in enumerate(self.BAR_GWD):
            if gwd_idx < len(self.PGMAX_EFETIVO):
                self.PGMAX_CURTAILMENT[i] = self.PGMAX_EFETIVO[gwd_idx]
    
    def get_sistema_dict(self) -> Dict:
        """Retorna dicionário com todos os dados do sistema processados"""
        return {
            # Parâmetros base
            'SB': self.SB,
            'VB': self.VB,
            'f_base': self.f_base,
            'ZB': self.ZB,
            
            # Barras
            'bus_ids': self.bus_ids,
            'idx_map': self.idx_map,
            'indice_para_barra': self.indice_para_barra,
            'NBAR': self.NBAR,
            'slack_idx': self.slack_idx,
            
            # Linhas
            'NLIN': self.NLIN,
            'line_fr': self.line_fr,
            'line_to': self.line_to,
            'x_line': self.x_line,
            'r_line': self.r_line,
            'FLIM': self.FLIM,
            
            # Geradores
            'NGER': self.NGER,
            'NGER_ORIGINAL': self.NGER_ORIGINAL,
            'BARPG': self.BARPG,
            'BARPG_ORIGINAL': self.BARPG_ORIGINAL,
            'BAR_GWD': self.BAR_GWD,
            'GER_TIPOS': self.GER_TIPOS,
            'GER_TIPOS_COMBINADO': self.GER_TIPOS_COMBINADO,
            'PGMIN': self.PGMIN,
            'PGMAX': self.PGMAX,
            'PGMIN_EFETIVO': self.PGMIN_EFETIVO,
            'PGMAX_EFETIVO': self.PGMAX_EFETIVO,
            'CPG': self.CPG,
            'CPG_ORIGINAL': self.CPG_ORIGINAL,
            
            # Carga
            'PLOAD': self.PLOAD,
            'QLOAD': self.QLOAD,
            
            # Dados originais
            'barras': self.barras,
            'geradores_data': self.geradores_data,
            'demandas_data': self.demandas_data,
            'linhas': self.linhas,
            'json_file_path': self.json_file_path
        }