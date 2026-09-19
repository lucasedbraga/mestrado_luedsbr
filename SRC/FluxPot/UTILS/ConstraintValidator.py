#!/usr/bin/env python3
"""
ConstraintValidator.py

Valida as restrições físicas e operativas de um cenário simulado,
utilizando os dados armazenados no banco SQLite e as definições do sistema.

Uso:
    python ConstraintValidator.py <cen_id>
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

# ==================== CONFIGURAÇÕES ====================
DB_PATH = "DATA/output/resultados_PL_acoplado.db"
JSON_PATH = "DATA/input/ieee14_BASE.json"
TOL = 1e-4                     # tolerância para verificações

# Se nenhum argumento for passado, use este cen_id (exemplo)
DEFAULT_CEN_ID = "20260310234208"
# ========================================================

# Ajusta o path para encontrar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from UTILS.SystemLoader import SistemaLoader

# ------------------------------------------------------------
def carregar_sistema(json_path):
    """Carrega os parâmetros do sistema."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Arquivo do sistema não encontrado: {json_path}")
    sistema = SistemaLoader(json_path)
    return sistema

# ------------------------------------------------------------
def obter_dados_cenario(db_path, cen_id):
    """Retorna um DataFrame com todos os dados do cenário."""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT cen_id, data_simulacao, hora_simulacao, BAR_id,
               PLOAD_cenario,
               BESS_init_cenario,
               PGWIND_disponivel_cenario,
               PGER_UTE_result,
               CURTAILMENT_total_result,
               BESS_operation_result
        FROM DBAR_results
        WHERE cen_id = ?
        ORDER BY hora_simulacao, BAR_id
    '''
    df = pd.read_sql_query(query, conn, params=(cen_id,))
    conn.close()
    return df

# ------------------------------------------------------------
def mapear_recursos(sistema):
    """
    Extrai do sistema as barras que possuem cada tipo de recurso
    e calcula parâmetros agregados por barra.
    Retorna dicionário com as informações.
    """
    thermal_bars = {}   # barra -> dict de parâmetros agregados
    # Geradores térmicos
    if hasattr(sistema, 'NGER_UTE') and sistema.NGER_UTE > 0:
        for i in range(sistema.NGER_UTE):
            bar = sistema.BAR_PGER_UTE[i]
            if bar not in thermal_bars:
                thermal_bars[bar] = {
                    'pmax': 0.0,
                    'pmin': 0.0,
                    'ramp_up': 0.0,
                    'ramp_down': 0.0,
                    'first_up': 0.0,
                    'first_down': 0.0
                }
            thermal_bars[bar]['pmax'] += sistema.PGER_MAX_UTE[i]
            thermal_bars[bar]['pmin'] += sistema.PGER_MIN_UTE[i]
            thermal_bars[bar]['ramp_up'] += sistema.RAMP_UP[i]
            thermal_bars[bar]['ramp_down'] += sistema.RAMP_DOWN[i]
            # Para a primeira hora, usamos os mesmos limites de rampa (o sistema não fornece valores especiais)
            thermal_bars[bar]['first_up'] += sistema.RAMP_UP[i]
            thermal_bars[bar]['first_down'] += sistema.RAMP_DOWN[i]

    # Geradores eólicos (lista de barras, sem repetição)
    wind_bars = []
    if hasattr(sistema, 'NGER_GWD') and sistema.NGER_GWD > 0:
        wind_bars = sorted(set(sistema.BARPG_EOL))

    # Baterias
    battery_bars = []
    battery_capacity = {}
    battery_charge_limit = {}
    battery_discharge_limit = {}
    if hasattr(sistema, 'BARRAS_COM_BATERIA'):
        battery_bars = sistema.BARRAS_COM_BATERIA
        if hasattr(sistema, 'BATTERY_CAPACITY'):
            for bar in battery_bars:
                battery_capacity[bar] = sistema.BATTERY_CAPACITY[bar]
        if hasattr(sistema, 'BATTERY_POWER_LIMIT'):
            for bar in battery_bars:
                battery_charge_limit[bar] = sistema.BATTERY_POWER_LIMIT[bar]
        if hasattr(sistema, 'BATTERY_POWER_OUT'):
            for bar in battery_bars:
                battery_discharge_limit[bar] = sistema.BATTERY_POWER_OUT[bar]

    # Barras com carga (PLOAD > 0)
    load_bars = []
    if hasattr(sistema, 'PLOAD'):
        load_bars = [i for i, p in enumerate(sistema.PLOAD) if p > 1e-6]

    return {
        'thermal': thermal_bars,
        'wind': wind_bars,
        'battery': battery_bars,
        'battery_capacity': battery_capacity,
        'battery_charge_limit': battery_charge_limit,
        'battery_discharge_limit': battery_discharge_limit,
        'load': load_bars
    }

# ------------------------------------------------------------
def validar_balanco_global(df, sistema, recursos, tol=TOL):
    """
    Verifica o balanço de potência global para cada hora:
        Σ PGER + Σ (PGWIND_disponivel - CURTAILMENT) + Σ BESS_operation == Σ PLOAD
    O resíduo indica a presença de déficit ou superávit não contabilizado.
    """
    violacoes = []
    for hora in sorted(df['hora_simulacao'].unique()):
        df_h = df[df['hora_simulacao'] == hora]

        # Geração térmica (soma sobre todas as barras)
        pger = df_h['PGER_UTE_result'].sum()

        # Geração eólica efetiva
        df_eol = df_h[df_h['BAR_id'].isin(recursos['wind'])]
        pwind_efet = (df_eol['PGWIND_disponivel_cenario'] - df_eol['CURTAILMENT_total_result']).sum()

        # Injeção líquida das baterias (positiva = descarga)
        df_bat = df_h[df_h['BAR_id'].isin(recursos['battery'])]
        bess_inj = df_bat['BESS_operation_result'].sum()

        # Carga total
        pload = df_h['PLOAD_cenario'].sum()

        lhs = pger + pwind_efet + bess_inj
        rhs = pload
        residuo = lhs - rhs

        if abs(residuo) > tol:
            violacoes.append(
                f"Hora {hora:02d}: Balanço global não fecha. "
                f"LHS = {lhs:.6f}, RHS = {rhs:.6f}, resíduo = {residuo:.6f} "
                f"(possível déficit/superávit não contabilizado)"
            )
    return violacoes

# ------------------------------------------------------------
def validar_geracao_termica(df, sistema, recursos, tol=TOL):
    """
    Valida limites e rampas da geração térmica para cada barra com gerador.
    """
    violacoes = []
    thermal_bars = recursos['thermal']

    for bar, params in thermal_bars.items():
        df_bar = df[df['BAR_id'] == bar].copy().sort_values('hora_simulacao')
        if df_bar.empty:
            continue
        horas = df_bar['hora_simulacao'].values
        pger = df_bar['PGER_UTE_result'].values

        pmax = params['pmax']
        pmin = params['pmin']
        ramp_up = params['ramp_up']
        ramp_down = params['ramp_down']
        # Primeira hora: usamos os mesmos limites de rampa (não há dados especiais)
        first_up = params['first_up']   # = ramp_up (ou poderia ser 0.5 se quisesse)
        first_down = params['first_down']

        # Limites absolutos
        for i, h in enumerate(horas):
            if pger[i] < pmin - tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: PGER = {pger[i]:.4f} < mínimo {pmin}")
            if pger[i] > pmax + tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: PGER = {pger[i]:.4f} > máximo {pmax}")

        # Primeira hora (limites especiais – opcional)
        if pger[0] > first_up + tol:
            violacoes.append(f"Hora {horas[0]:02d} Barra {bar}: PGER = {pger[0]:.4f} > limite primeira hora {first_up}")
        if pger[0] < first_down - tol:
            violacoes.append(f"Hora {horas[0]:02d} Barra {bar}: PGER = {pger[0]:.4f} < limite primeira hora {first_down}")

        # Rampas entre horas consecutivas
        for i in range(1, len(horas)):
            delta = pger[i] - pger[i-1]
            if delta > ramp_up + tol:
                violacoes.append(f"Rampa entre hora {horas[i-1]:02d} e {horas[i]:02d} (barra {bar}): Δ = {delta:.4f} > {ramp_up}")
            if delta < -ramp_down - tol:
                violacoes.append(f"Rampa entre hora {horas[i-1]:02d} e {horas[i]:02d} (barra {bar}): Δ = {delta:.4f} < -{ramp_down}")

    return violacoes

# ------------------------------------------------------------
def validar_geracao_eolica(df, recursos, tol=TOL):
    """
    Valida limites do corte e geração eólica para cada barra com eólica.
    """
    violacoes = []
    wind_bars = recursos['wind']

    for bar in wind_bars:
        df_bar = df[df['BAR_id'] == bar].copy().sort_values('hora_simulacao')
        if df_bar.empty:
            continue
        for _, row in df_bar.iterrows():
            h = row['hora_simulacao']
            disp = row['PGWIND_disponivel_cenario']
            corte = row['CURTAILMENT_total_result']
            if corte < -tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: CURTAILMENT = {corte:.4f} < 0")
            if corte > disp + tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: CURTAILMENT = {corte:.4f} > disponível {disp:.4f}")
            if disp - corte < -tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: Geração eólica efetiva = {disp-corte:.4f} < 0")
    return violacoes

# ------------------------------------------------------------
def validar_bateria(df, sistema, recursos, tol=TOL):
    """
    Valida limites da operação da bateria e evolução do SOC para cada barra com bateria.
    """
    violacoes = []
    battery_bars = recursos['battery']
    if not battery_bars:
        return violacoes

    soc_min = 0.1
    soc_max = 1.0
    soc_inicial_esperado = 0.5
    soc_final_esperado = 0.5
    eff_charge = 0.9
    eff_discharge = 0.9

    for bar in battery_bars:
        df_bar = df[df['BAR_id'] == bar].copy().sort_values('hora_simulacao')
        if df_bar.empty:
            continue

        horas = df_bar['hora_simulacao'].values
        op = df_bar['BESS_operation_result'].values
        soc = df_bar['BESS_init_cenario'].values
        capacity = recursos['battery_capacity'].get(bar, 1.0)
        charge_limit = recursos['battery_charge_limit'].get(bar, 0.1)
        discharge_limit = recursos['battery_discharge_limit'].get(bar, 0.1)

        # Limites de potência
        for i, h in enumerate(horas):
            if op[i] > tol and op[i] > discharge_limit + tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: BESS_operation = {op[i]:.4f} > limite descarga {discharge_limit}")
            if op[i] < -tol and -op[i] > charge_limit + tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: |BESS_operation| = {-op[i]:.4f} > limite carga {charge_limit}")

        # Limites de SOC
        for i, h in enumerate(horas):
            if soc[i] < soc_min - tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: SOC = {soc[i]:.4f} < mínimo {soc_min}")
            if soc[i] > soc_max + tol:
                violacoes.append(f"Hora {h:02d} Barra {bar}: SOC = {soc[i]:.4f} > máximo {soc_max}")

        # Condições inicial e final (se as horas corresponderem)
        if len(horas) > 0 and horas[0] == 0 and abs(soc[0] - soc_inicial_esperado) > tol:
            violacoes.append(f"Hora 0 Barra {bar}: SOC inicial = {soc[0]:.4f} != esperado {soc_inicial_esperado}")
        if len(horas) > 0 and horas[-1] == 23 and abs(soc[-1] - soc_final_esperado) > tol:
            violacoes.append(f"Hora 23 Barra {bar}: SOC final = {soc[-1]:.4f} != esperado {soc_final_esperado}")

        # Evolução do SOC entre horas consecutivas
        for i in range(len(horas)-1):
            h_atual = horas[i]
            h_prox = horas[i+1]
            op_atual = op[i]
            soc_atual = soc[i]
            soc_prox_esperado = soc_atual

            if op_atual > tol:
                # Descarga
                energia_desc = op_atual * 1.0  # MWh
                soc_prox_esperado -= (energia_desc / eff_discharge) / capacity
            elif op_atual < -tol:
                # Carga
                energia_carga = -op_atual * 1.0
                soc_prox_esperado += (energia_carga * eff_charge) / capacity
            # else: op=0, não altera

            if abs(soc_prox_esperado - soc[i+1]) > tol:
                violacoes.append(
                    f"Evolução SOC entre hora {h_atual:02d} e {h_prox:02d} (barra {bar}): "
                    f"calculado {soc_prox_esperado:.4f}, registrado {soc[i+1]:.4f} "
                    f"(diferença {soc_prox_esperado - soc[i+1]:.4f})"
                )
    return violacoes

# ------------------------------------------------------------
def main():
    if len(sys.argv) >= 2:
        cen_id = sys.argv[1]
    else:
        cen_id = DEFAULT_CEN_ID
        print(f"Nenhum cen_id fornecido. Usando default: {cen_id}")

    print("=" * 70)
    print("VALIDADOR DE RESTRIÇÕES DE CENÁRIO (ÊNFASE NO BALANÇO DE MASSA)")
    print("=" * 70)

    # Carregar sistema
    try:
        sistema = carregar_sistema(JSON_PATH)
        print(f"Sistema carregado: {JSON_PATH}")
    except Exception as e:
        print(f"Erro ao carregar sistema: {e}")
        return 1

    # Mapear recursos por barra
    recursos = mapear_recursos(sistema)
    print(f"Barras com térmica: {list(recursos['thermal'].keys())}+1")
    print(f"Barras com eólica: {recursos['wind']+1}")
    print(f"Barras com bateria: {recursos['battery']+1}")
    print(f"Barras com carga: {recursos['load']+1}")

    # Obter dados do cenário
    try:
        df = obter_dados_cenario(DB_PATH, cen_id)
        if df.empty:
            print(f"Nenhum dado encontrado para cen_id {cen_id}")
            return 1
        print(f"Dados carregados: {len(df)} registros")
    except Exception as e:
        print(f"Erro ao acessar banco: {e}")
        return 1

    # Coletar violações
    todas_violacoes = []

    print("\n--- Validando balanço global (massa) ---")
    v = validar_balanco_global(df, sistema, recursos, tol=TOL)
    todas_violacoes.extend(v)
    print(f"  {len(v)} violações encontradas.")

    print("\n--- Validando geração térmica ---")
    v = validar_geracao_termica(df, sistema, recursos, tol=TOL)
    todas_violacoes.extend(v)
    print(f"  {len(v)} violações encontradas.")

    print("\n--- Validando geração eólica ---")
    v = validar_geracao_eolica(df, recursos, tol=TOL)
    todas_violacoes.extend(v)
    print(f"  {len(v)} violações encontradas.")

    print("\n--- Validando bateria (operação e SOC) ---")
    v = validar_bateria(df, sistema, recursos, tol=TOL)
    todas_violacoes.extend(v)
    print(f"  {len(v)} violações encontradas.")

    # Resultado final
    print("\n" + "=" * 70)
    if not todas_violacoes:
        print("Nenhuma violação encontrada. O cenário é factível dentro das tolerâncias.")
    else:
        print(f"Total de violações: {len(todas_violacoes)}")
        print("\nLista de violações:")
        for i, viol in enumerate(todas_violacoes, 1):
            print(f"{i:3d}. {viol}")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())