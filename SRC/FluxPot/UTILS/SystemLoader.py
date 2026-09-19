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

        self.SB = 1.0  # Potência base
        self.f_base = 1.0  # Frequência base
        self.VB = 1.0  # Tensão base
        self.ZB = 1.0  # Impedância base

        # Estruturas processadas
        self.bus_ids = []
        self.idx_map = {}
        self.indice_para_barra = {}
        self.NBAR = 0
        self.NLIN = 0
        self.slack_idx = 0

        # Arrays de linhas
        self.line_fr = []
        self.line_to = []
        self.x_line = np.array([])
        self.r_line = np.array([])
        self.FLIM = np.array([])

        # --- Geradores convencionais (UTE, UTH) ---
        self.NGER_UTE = 0
        self.BAR_PGER_UTE = []          # índices das barras dos convencionais
        self.GER_TIPO = []      # tipos ("UTE", "UTH", etc.)
        self.PGER_MIN = np.array([])
        self.PGER_MAX = np.array([])
        self.QGER_MIN = np.array([])
        self.QGER_MAX = np.array([])
        self.custo_GER = np.array([])
        self.RAMP_UP = np.array([])
        self.RAMP_DOWN = np.array([])
        self.PGER_INICIAL_UTE = np.array([])   # geração inicial (pu)

        # --- Geradores eólicos (GWD) ---
        self.NGER_GWD = 0
        self.BARPG_EOL = []            # índices das barras dos eólicos
        self.PGWD_MAX_ORIGINAL = np.array([])   # capacidade instalada (pu)
        self.PGWD_MAX_EFETIVO = np.array([])    # após fator de vento (pu)
        self.PGWIND_disponivel = np.array([])    # disponibilidade atual (pu) – igual ao efetivo
        self.custo_curtailment = np.array([])      # custo de curtailment por eólico (USD/pu)

        # Mapeamento auxiliar: índice global do gerador no JSON -> posição na lista de eólicos
        self.gwd_idx_to_pos = {}

        # --- Déficit (por barra) ---
        self.custo_DEFICIT  = 5000.0 * self.SB   # custo do déficit (USD/pu), pode ser escalar

        # Carga
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
        #self.processa_pu()

        # Processa barras
        self.processa_BARRAS()

        # Processa linhas
        self.processa_LINHAS()

        # Processa geradores
        self.processa_GER()

        # Processa carga
        self.processa_LOAD()

        # Processa déficit
        self.processa_DEFICT()

        # Processa gwd e curtailment
        self.processa_GWD()

        # Processa baterias (se houver)
        if len(self.baterias_data) > 0:
            self.processa_BESS()
        else:
            self.BARRAS_COM_BATERIA = []
            self.BATTERIES = []
            self.BATTERY_POWER_LIMIT = np.zeros(self.NBAR)
            self.BATTERY_POWER_OUT = np.zeros(self.NBAR)
            self.BATTERY_CAPACITY = np.zeros(self.NBAR)
            self.BATTERY_MIN_SOC = np.zeros(self.NBAR)
            self.BATTERY_INITIAL_SOC = np.zeros(self.NBAR)
            self.BATTERY_COST_CHARGE = np.zeros(self.NBAR)
            self.BATTERY_COST_DISCHARGE = np.zeros(self.NBAR)

    def processa_pu(self):
        """Converte todos os valores para PU"""
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
            b["Q_carga_pu"] = b.get("Q_carga_MVAr", 0.0) / self.SB

        # Geradores
        for g in self.geradores_data:
            g["PGER_MIN"] = g.get("PGER_MIN", 0.0) / self.SB
            g["PGER_MAX"] = g.get("PGER_MAX", 0.0) / self.SB
            g["QGER_MIN"] = g.get("QGER_MIN", 0.0) / self.SB
            g["QGER_MAX"] = g.get("QGER_MAX", 0.0) / self.SB

            # Custos em USD/pu
            g["CustoGeracao"] = g.get("custo_GER", 0.0) * self.SB
            g["CustoCurtailment"] = g.get("custo_curtailment", 100.0) * self.SB

        # Demandas
        for d in self.demandas_data:
            d["PLOAD"] = d.get("PLOAD", 0.0) / self.SB
            d["QLOAD"] = d.get("QLOAD", 0.0) / self.SB

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

    def processa_BARRAS(self):
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

    def processa_LINHAS(self):
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
            self.FLIM[e] = ln.get("LIM_Fluxo", 1.0)

    def processa_GER(self):
        """
        Separa os geradores em convencionais (UTE, UTH)
        Preenche os arrays correspondentes.
        """
        # Listas temporárias para convencionais
        BAR_PGER_UTE = []
        tipos_conv = []
        PGER_MIN_UTE = []
        PGER_MAX_UTE = []
        QGER_MIN_UTE = []
        QGER_MAX_UTE = []
        V_ESP_BUS = []
        custo_GER = []
        P_ramp_up = []
        P_ramp_down = []
        pg_inicial_conv = []  # geração inicial (pu)

        # Itera sobre todos os geradores do JSON
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            barra_idx = self.idx_map[id_barra]
            tipo = g.get("Tipo", "CONV")

            if tipo != "GWD":
                # Gerador convencional
                BAR_PGER_UTE.append(barra_idx)
                tipos_conv.append(tipo)
                PGER_MIN_UTE.append(g.get("PGER_MIN", 0.0))
                PGER_MAX_UTE.append(g.get("PGER_MAX", 1.0))
                QGER_MIN_UTE.append(g.get("QGER_MIN", 0.0))
                QGER_MAX_UTE.append(g.get("QGER_MAX", 0.0))
                custo_GER.append(g.get("CustoGeracao", 50.0))
                P_ramp_up.append(g.get("P_ramp_up", 100))
                P_ramp_down.append(g.get("P_ramp_down", 100))
                # Geração inicial: campo opcional, se não existir, assume 0.0
                pg_ini_mw = g.get("PGER_inicial", 0.0)
                pg_inicial_conv.append(pg_ini_mw / self.SB)
                if tipo == "SINC":
                    V_ESP_BUS.append((barra_idx,1))
                else:
                    V_ESP_BUS.append(0)

        # Converte para arrays numpy
        self.NGER_UTE = len(BAR_PGER_UTE)
        self.BAR_PGER_UTE = BAR_PGER_UTE
        self.GER_TIPO = tipos_conv
        self.PGER_MIN_UTE = np.array(PGER_MIN_UTE)
        self.PGER_MAX_UTE = np.array(PGER_MAX_UTE)
        self.QGER_MIN_UTE = np.array(QGER_MIN_UTE)
        self.QGER_MAX_UTE = np.array(QGER_MAX_UTE)
        self.V_ESP_BUS = V_ESP_BUS
        self.custo_GER = np.array(custo_GER)
        self.RAMP_UP = np.array(P_ramp_up)
        self.RAMP_DOWN = np.array(P_ramp_down)
        self.PGER_INICIAL_UTE = np.array(pg_inicial_conv)

        print(f"  ✓ Geradores processados: {self.NGER_UTE} UTE")

    def processa_LOAD(self):
        """Processa dados de carga"""
        self.PLOAD = np.zeros(self.NBAR)
        self.QLOAD = np.zeros(self.NBAR)

        for d in self.demandas_data:
            idx = self.idx_map[d["ID_Barra"]]
            self.PLOAD[idx] += d.get("PLOAD", 0.0)
            self.QLOAD[idx] += d.get("QLOAD", 0.0)
    
    def processa_DEFICT(self):
        """
        Define as barras que podem ter déficit (PQ sem gerador convencional)
        e o custo do déficit. Não cria geradores artificiais.
        O déficit será modelado como variável por barra no OPF.
        """
        # Barras PQ
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        # Barras que possuem gerador convencional
        barras_com_gerador_conv = set(self.BAR_PGER_UTE)
        # Barras PQ sem gerador convencional (passíveis de déficit)
        self.barras_PQ_sem_gerador = []
        for b in barras_PQ:
            idx = self.idx_map[b["ID_Barra"]]
            if idx not in barras_com_gerador_conv:
                self.barras_PQ_sem_gerador.append(b)

        # Custo do déficit 
        self.custo_DEFICIT  = 5000.0 * self.SB  # USD/pu

        print(f"  ✓ Déficit: {len(self.barras_PQ_sem_gerador)} barras PQ sem gerador convencional")

    def processa_GWD(self):
        # Listas temporárias para eólicos
        barpg_eol = []
        pgmax_eol_orig = []
        custo_curtail = []

        # Itera sobre todos os geradores do JSON
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            barra_idx = self.idx_map[id_barra]
            tipo = g.get("Tipo", "CONV")

            if tipo == "GWD":
                # Gerador eólico
                pos = len(barpg_eol)
                barpg_eol.append(barra_idx)
                pgmax_orig = g.get("PGER_MAX", 0.0)
                pgmax_eol_orig.append(pgmax_orig)
                custo_curtail.append(g.get("custo_curtailment", 1000.0))
                self.gwd_idx_to_pos[i] = pos  # mapeia índice global para posição na lista de eólicos
        
        self.NGER_GWD = len(barpg_eol)
        self.BARPG_EOL = barpg_eol
        self.PGWD_MAX_ORIGINAL = np.array(pgmax_eol_orig)
        self.PGWD_MAX_EFETIVO = self.PGWD_MAX_ORIGINAL.copy()  # inicialmente igual à original
        self.PGWIND_disponivel = self.PGWD_MAX_EFETIVO.copy()   # disponibilidade atual (será atualizada)
        self.custo_curtailment = np.array(custo_curtail)

        print(f"  ✓ Geradores processados: {self.NGER_GWD} GWD")

    def processa_BESS(self):
        """Processa dados de baterias"""
        self.BARRAS_COM_BATERIA = []

        BATmax_in_base = np.zeros(self.NBAR)
        BATmax_out_base = np.zeros(self.NBAR)
        BATcapacidade_base = np.zeros(self.NBAR)
        BATarm_inicial_base = np.zeros(self.NBAR)
        BATminSoc_base = np.zeros(self.NBAR)

        self.BATTERY_POWER_LIMIT = np.zeros(self.NBAR)
        self.BATTERY_POWER_OUT = np.zeros(self.NBAR)
        self.BATTERY_CAPACITY = np.zeros(self.NBAR)
        self.BATTERY_MIN_SOC = np.zeros(self.NBAR)
        self.BATTERY_INITIAL_SOC = np.zeros(self.NBAR)
        self.BATTERY_COST_CHARGE = np.zeros(self.NBAR)
        self.BATTERY_COST_DISCHARGE = np.zeros(self.NBAR)

        for bat in self.baterias_data:
            id_barra = str(bat["ID_Barra"])
            if id_barra not in self.idx_map:
                print(f"  ⚠️  Bateria em barra {id_barra} não encontrada - ignorando")
                continue

            idx = self.idx_map[id_barra]
            self.BARRAS_COM_BATERIA.append(idx)

            if "Pmax_carga_base_pu" in bat:
                p_max_carga = bat["Pmax_carga_base_pu"]
            else:
                p_max_carga = bat.get("Pmax_carga", 0.0) / self.SB

            if "capacidade_base_pu" in bat:
                capacidade = bat["capacidade_base_pu"]
            else:
                capacidade = bat.get("capacidade_armazenamento", 0.0) / self.SB

            BATmax_in_base[idx] = p_max_carga
            BATmax_out_base[idx] = bat.get("Pmax_descarga_pu", p_max_carga)
            BATcapacidade_base[idx] = capacidade
            BATarm_inicial_base[idx] = bat.get("SOC_inicial_pu", 0.5) * capacidade
            BATminSoc_base[idx] = bat.get("min_soc_pu", 0.1) * capacidade
            self.BATTERY_COST_CHARGE[idx] = bat.get("custo_carga", 10.0)
            self.BATTERY_COST_DISCHARGE[idx] = bat.get("custo_descarga", 10.0)

        self.BATTERY_POWER_LIMIT = BATmax_in_base.copy()
        self.BATTERY_POWER_OUT = BATmax_out_base.copy()
        self.BATTERY_CAPACITY = BATcapacidade_base.copy()
        self.BATTERY_MIN_SOC = BATminSoc_base.copy()
        self.BATTERY_INITIAL_SOC = BATarm_inicial_base.copy()
        self.BATTERIES = self.BARRAS_COM_BATERIA.copy()

        print(f"  ✓ Baterias processadas: {len(self.BARRAS_COM_BATERIA)} bateria(s) em {len(set(self.BARRAS_COM_BATERIA))} barra(s)")

    def atualizar_perfil_eolico(self, fator_vento: float):
        """
        Atualiza a capacidade eólica efetiva com base no fator de vento.
        PGWD_MAX_EFETIVO = PGWD_MAX_ORIGINAL * fator_vento
        PGWIND_disponivel = PGWD_MAX_EFETIVO (disponibilidade atual)
        """
        self.PGWD_MAX_EFETIVO = self.PGWD_MAX_ORIGINAL * fator_vento
        self.PGWIND_disponivel = self.PGWD_MAX_EFETIVO.copy()
        print(f"  ✓ Perfil eólico atualizado: fator={fator_vento:.3f}")

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

            # Geradores convencionais
            'NGER_UTE': self.NGER_UTE,
            'BAR_PGER_UTE': self.BAR_PGER_UTE,
            'GER_TIPO': self.GER_TIPO,
            'PGER_MIN_UTE': self.PGER_MIN_UTE,
            'PGER_MAX_UTE': self.PGER_MAX_UTE,
            'custo_GER': self.custo_GER,
            'RAMP_UP': self.RAMP_UP,
            'RAMP_DOWN': self.RAMP_DOWN,
            'PGER_INICIAL_UTE': self.PGER_INICIAL_UTE,

            # Geradores eólicos
            'NGER_GWD': self.NGER_GWD,
            'BARPG_EOL': self.BARPG_EOL,
            'PGWD_MAX_ORIGINAL': self.PGWD_MAX_ORIGINAL,
            'PGWD_MAX_EFETIVO': self.PGWD_MAX_EFETIVO,
            'PGWIND_disponivel': self.PGWIND_disponivel,
            'custo_curtailment': self.custo_curtailment,
            'gwd_idx_to_pos': self.gwd_idx_to_pos,

            # Déficit
            'barras_PQ_sem_gerador': self.barras_PQ_sem_gerador,
            'custo_DEFICIT ': self.custo_DEFICIT ,

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


def load_system(json_file_path: str):
    """
    Função utilitária para carregar e retornar o objeto de sistema já processado.
    Compatível com o uso em scripts principais.
    """
    return SistemaLoader(json_file_path)