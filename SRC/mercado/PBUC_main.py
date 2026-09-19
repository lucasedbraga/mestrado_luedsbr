#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBUC_ResultHandler.py
=====================
Manipulação, análise, persistência e visualização de resultados do modelo PBUC
(Profit-Based Unit Commitment com OPF multi-período).

CORREÇÕES APLICADAS:
  [FIX-1] analise_deficit_por_barra  -> DataFrame vazio COM colunas
  [FIX-2] analise_fluxo_linhas       -> DataFrame vazio COM colunas
  [FIX-3] analise_atendimento_horario-> DataFrame vazio COM colunas
  [FIX-4] PBUC_Plotter              -> removida duplicata de plot_commitment_heatmap
  [FIX-5] save_pbuc_result          -> salva déficit por barra de forma segura
  [FIX-6] __main__                  -> impressão segura (checa .empty)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import do OPF_DBHandler existente (com fallbacks)
try:
    from DB.DBhandler_OPF import OPF_DBHandler
except ImportError:
    try:
        from DBhandler_OPF import OPF_DBHandler
    except ImportError:
        class OPF_DBHandler:  # type: ignore
            def __init__(self, db_path: str = 'DATA/output/resultados_PL.db'):
                self.db_path = db_path
                self.conn = None
            def connect(self):
                self.conn = sqlite3.connect(self.db_path)
                return self.conn
            def disconnect(self):
                if self.conn:
                    self.conn.close()
            def create_tables(self):
                pass


# =============================================================================
# 1. DATACLASSES
# =============================================================================
@dataclass
class PBUC_GeneratorResult:
    ID_Gerador: str
    ID_Barra: int
    Zona: str
    U_schedule: List[int] = field(default_factory=list)
    V_startup: List[int] = field(default_factory=list)
    W_shutdown: List[int] = field(default_factory=list)
    PGER: List[float] = field(default_factory=list)
    QGER: List[float] = field(default_factory=list)
    receita_mercado: float = 0.0
    receita_contrato: float = 0.0
    custo_combustivel: float = 0.0
    custo_startup: float = 0.0
    lucro_total: float = 0.0
    horas_ligado: int = 0
    num_partidas: int = 0


@dataclass
class PBUC_ContractResult:
    ID_Contrato: str
    Zona: str
    Preco_contrato: float
    Multa: float
    Amount_contratado: float
    entrega: List[float] = field(default_factory=list)
    deficit: List[float] = field(default_factory=list)
    receita_total: float = 0.0
    multa_total: float = 0.0
    custo_efetivo_cliente: float = 0.0
    taxa_atendimento: float = 0.0
    deficit_total: float = 0.0


@dataclass
class PBUC_SnapshotResult:
    dia: int
    hora: int
    sucesso: bool = True
    dia_semana: int = 0
    dia_semana_nome: str = ""

    PLOAD: List[float] = field(default_factory=list)
    PGER: List[float] = field(default_factory=list)
    QGER: List[float] = field(default_factory=list)
    PGWIND: List[float] = field(default_factory=list)
    PGWIND_disponivel: List[float] = field(default_factory=list)
    CURTAILMENT: List[float] = field(default_factory=list)
    U_schedule: List[int] = field(default_factory=list)
    V_startup: List[int] = field(default_factory=list)
    W_shutdown: List[int] = field(default_factory=list)
    DEFICIT: List[float] = field(default_factory=list)
    LAMBDA_ZONA: List[float] = field(default_factory=list)
    CONT_ENTREGA: List[float] = field(default_factory=list)
    CONT_DEFICIT: List[float] = field(default_factory=list)

    FLUXO_LIN: List[float] = field(default_factory=list)
    LINE_LOADING: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    V: List[float] = field(default_factory=list)

    total_geracao: float = 0.0
    total_demanda: float = 0.0
    total_deficit: float = 0.0
    total_curtailment: float = 0.0
    total_wind: float = 0.0
    balanco: float = 0.0

    receita_mercado: float = 0.0
    receita_contratos: float = 0.0
    custo_combustivel: float = 0.0
    custo_startup: float = 0.0
    multa_contratos: float = 0.0
    custo_deficit: float = 0.0
    custo_curtailment: float = 0.0
    lucro_horario: float = 0.0

    mensagem: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class PBUC_Result:
    snapshots: List[PBUC_SnapshotResult] = field(default_factory=list)
    geradores: List[PBUC_GeneratorResult] = field(default_factory=list)
    contratos: List[PBUC_ContractResult] = field(default_factory=list)
    sucesso_global: bool = True
    mensagem_global: str = ""

    lucro_total: float = 0.0
    receita_mercado_total: float = 0.0
    receita_contratos_total: float = 0.0
    custo_combustivel_total: float = 0.0
    custo_startup_total: float = 0.0
    multa_total: float = 0.0
    custo_deficit_total: float = 0.0
    custo_curtailment_total: float = 0.0


# =============================================================================
# 2. EXTRATOR
# =============================================================================
class PBUC_ResultExtractor:
    def __init__(self, modelo_pbuc, sistema, market_json: dict):
        self.modelo = modelo_pbuc
        self.sistema = sistema
        self.market = market_json
        self.T = modelo_pbuc.horizon_time
        self.SB = sistema.SB

        self.zonas = {z['ID_Zona']: z for z in market_json.get('ZONAS', [])}
        self.zona_list = list(self.zonas.keys())
        self.barra_zona = {}
        for zid, zdata in self.zonas.items():
            for b in zdata['Barras_Associadas']:
                self.barra_zona[int(b) - 1] = self.zona_list.index(zid)

    def extract(self) -> PBUC_Result:
        import pyomo.environ as pyo
        if self.modelo.model is None:
            raise RuntimeError("Modelo não construído.")

        use_dcopf = bool(getattr(self.modelo, 'use_dcopf', True))
        snapshots: List[PBUC_SnapshotResult] = []

        for t in range(self.T):
            dia = t // self.modelo.n_horas
            hora = t % self.modelo.n_horas
            dia_sem = ((self.modelo.dia_inicial + dia) % 7) + 1

            PLOAD = self.modelo.PLOAD[t, :].tolist()
            PGER = [pyo.value(self.modelo.PGER_dict[(t, g)])
                    for g in range(self.modelo.NUTE)]

            if hasattr(self.modelo, 'QGER_dict') and self.modelo.QGER_dict:
                QGER = [pyo.value(self.modelo.QGER_dict[(t, g)])
                        for g in range(self.modelo.NUTE)]
            else:
                QGER = [0.0] * self.modelo.NUTE

            U_sched = [int(round(pyo.value(self.modelo.U_dict[(t, g)])))
                       for g in range(self.modelo.NUTE)]
            V_start = [int(round(pyo.value(self.modelo.V_start_dict[(t, g)])))
                       for g in range(self.modelo.NUTE)]
            W_stop = [int(round(pyo.value(self.modelo.W_stop_dict[(t, g)])))
                      for g in range(self.modelo.NUTE)]

            if self.modelo.NGWD > 0:
                PGWIND = [pyo.value(self.modelo.PGWIND_dict[(t, w)])
                          for w in range(self.modelo.NGWD)]
                PGWIND_disp = self.modelo.PGWIND_AVAIL[t, :].tolist()
                CURT = [pyo.value(self.modelo.CURTAILMENT_dict[(t, w)])
                        for w in range(self.modelo.NGWD)]
            else:
                PGWIND, PGWIND_disp, CURT = [], [], []

            DEF = [pyo.value(self.modelo.DEFICIT_dict[(t, b)])
                   for b in range(self.modelo.NBAR)]
            
            LAMBDA_Z = self.modelo.LAMBDA_ZONA[t, :].tolist()

            CONT_ENT = [pyo.value(self.modelo.CONTRATO_ENTREGA_dict[(t, c)])
                        for c in range(self.modelo.NCONTRATOS)]
            CONT_DEF = [pyo.value(self.modelo.CONTRATO_DEFICIT_dict[(t, c)])
                        for c in range(self.modelo.NCONTRATOS)]

            ANG = [pyo.value(self.modelo.ANG_dict[(t, b)])
                   for b in range(self.modelo.NBAR)]

            if hasattr(self.modelo, 'V_dict') and self.modelo.V_dict:
                V_val = [pyo.value(self.modelo.V_dict[(t, b)])
                         for b in range(self.modelo.NBAR)]
            else:
                V_val = [1.0] * self.modelo.NBAR

            FLUXO = []
            LOADING = []
            for l in range(self.modelo.NLIN):
                if use_dcopf:
                    if (t, l) in self.modelo.FLOW_dict:
                        f_val = pyo.value(self.modelo.FLOW_dict[(t, l)])
                    else:
                        f_val = 0.0
                else:
                    i = int(self.modelo.line_from[l])
                    j = int(self.modelo.line_to[l])
                    r_l = float(self.modelo.line_r[l])
                    x_l = float(self.modelo.line_x[l])
                    z2 = r_l * r_l + x_l * x_l
                    if z2 < 1e-12:
                        f_val = 0.0
                    else:
                        g_l = r_l / z2
                        b_l = -x_l / z2
                        Vi = V_val[i]
                        Vj = V_val[j]
                        th_ij = ANG[i] - ANG[j]
                        f_val = (Vi * Vi * g_l
                                 - Vi * Vj * (g_l * np.cos(th_ij)
                                              + b_l * np.sin(th_ij)))
                FLUXO.append(f_val)
                lim = float(self.modelo.line_flow_max[l])
                LOADING.append(abs(f_val) / lim * 100 if lim > 0 else 0.0)

            rec_merc = 0.0; c_comb = 0.0; c_start = 0.0; c_curt = 0.0
            for g in range(self.modelo.NUTE):
                b = self.modelo.thermal_bus[g]
                zid_idx = self.barra_zona.get(b, 0)
                lam = self.modelo.LAMBDA_ZONA[t, zid_idx]
                rec_merc += lam * PGER[g]
                c_comb += self.modelo.thermal_cost[g] * PGER[g]
                c_start += self.modelo.startup_cost[g] * V_start[g]
            for w in range(self.modelo.NGWD):
                if w < len(self.modelo.wind_bus):
                    b = self.modelo.wind_bus[w]
                    zid_idx = self.barra_zona.get(b, 0)
                    lam = self.modelo.LAMBDA_ZONA[t, zid_idx]
                    rec_merc += lam * PGWIND[w]
                    c_curt += lam * CURT[w]

            rec_cont = 0.0; multa = 0.0
            for c_idx, c in enumerate(self.modelo.contratos):
                rec_cont += c['Preco'] * CONT_ENT[c_idx]
                multa += c['Multa'] * CONT_DEF[c_idx]

            c_def = sum(self.modelo.custo_deficit_global * d for d in DEF)
            lucro_h = rec_merc + rec_cont - c_comb - c_start - multa - c_def - c_curt

            total_ger = sum(PGER) + sum(PGWIND)
            total_dem = sum(PLOAD)
            total_def = sum(DEF)
            total_curt = sum(CURT)
            balanco = (total_ger + total_def) - (total_dem + total_curt)

            snapshots.append(PBUC_SnapshotResult(
                dia=dia, hora=hora, sucesso=True,
                dia_semana=dia_sem, dia_semana_nome="",
                PLOAD=PLOAD, PGER=PGER, QGER=QGER,
                PGWIND=PGWIND, PGWIND_disponivel=PGWIND_disp,
                CURTAILMENT=CURT,
                U_schedule=U_sched, V_startup=V_start, W_shutdown=W_stop,
                DEFICIT=DEF, 
                LAMBDA_ZONA=LAMBDA_Z,
                CONT_ENTREGA=CONT_ENT, CONT_DEFICIT=CONT_DEF,
                FLUXO_LIN=FLUXO, LINE_LOADING=LOADING,
                ANG=ANG, V=V_val,
                total_geracao=total_ger, total_demanda=total_dem,
                total_deficit=total_def, total_curtailment=total_curt,
                total_wind=sum(PGWIND), balanco=balanco,
                receita_mercado=rec_merc, receita_contratos=rec_cont,
                custo_combustivel=c_comb, custo_startup=c_start,
                multa_contratos=multa, custo_deficit=c_def,
                custo_curtailment=c_curt, lucro_horario=lucro_h,
                timestamp=datetime.now()
            ))

        geradores: List[PBUC_GeneratorResult] = []
        for g in range(self.modelo.NUTE):
            b = self.modelo.thermal_bus[g]
            zid_idx = self.barra_zona.get(b, 0)
            U_sched = [s.U_schedule[g] for s in snapshots]
            V_start = [s.V_startup[g] for s in snapshots]
            W_stop = [s.W_shutdown[g] for s in snapshots]
            Pg = [s.PGER[g] for s in snapshots]
            Qg = [s.QGER[g] for s in snapshots]
            rec_m = sum(self.modelo.LAMBDA_ZONA[t, zid_idx] * Pg[t]
                        for t in range(self.T))
            c_comb = sum(self.modelo.thermal_cost[g] * Pg[t]
                         for t in range(self.T))
            c_start = sum(self.modelo.startup_cost[g] * V_start[t]
                          for t in range(self.T))
            geradores.append(PBUC_GeneratorResult(
                ID_Gerador=f"G{b+1}", ID_Barra=b + 1,
                Zona=self.zona_list[zid_idx],
                U_schedule=U_sched, V_startup=V_start, W_shutdown=W_stop,
                PGER=Pg, QGER=Qg,
                receita_mercado=rec_m, receita_contrato=0.0,
                custo_combustivel=c_comb, custo_startup=c_start,
                lucro_total=rec_m - c_comb - c_start,
                horas_ligado=sum(U_sched), num_partidas=sum(V_start)))

        contratos: List[PBUC_ContractResult] = []
        for c_idx, c in enumerate(self.modelo.contratos):
            entrega = [s.CONT_ENTREGA[c_idx] for s in snapshots]
            deficit = [s.CONT_DEFICIT[c_idx] for s in snapshots]
            rec = sum(c['Preco'] * e for e in entrega)
            mult = sum(c['Multa'] * d for d in deficit)
            amount_total = c['Amount'] * self.T
            taxa = (sum(entrega) / amount_total * 100) if amount_total > 0 else 0.0
            contratos.append(PBUC_ContractResult(
                ID_Contrato=c['ID'], Zona=c['Zona'],
                Preco_contrato=c['Preco'], Multa=c['Multa'],
                Amount_contratado=c['Amount'],
                entrega=entrega, deficit=deficit,
                receita_total=rec, multa_total=mult,
                custo_efetivo_cliente=rec + mult,
                taxa_atendimento=taxa, deficit_total=sum(deficit)))

        return PBUC_Result(
            snapshots=snapshots, geradores=geradores, contratos=contratos,
            sucesso_global=True, mensagem_global="OK",
            lucro_total=sum(s.lucro_horario for s in snapshots),
            receita_mercado_total=sum(s.receita_mercado for s in snapshots),
            receita_contratos_total=sum(s.receita_contratos for s in snapshots),
            custo_combustivel_total=sum(s.custo_combustivel for s in snapshots),
            custo_startup_total=sum(s.custo_startup for s in snapshots),
            multa_total=sum(s.multa_contratos for s in snapshots),
            custo_deficit_total=sum(s.custo_deficit for s in snapshots),
            custo_curtailment_total=sum(s.custo_curtailment for s in snapshots))


# =============================================================================
# 3. ANALISADOR
# =============================================================================
class PBUC_Analyzer:
    def __init__(self, resultado: PBUC_Result):
        self.r = resultado
        self.T = len(resultado.snapshots)

    def relatorio_lucro(self) -> pd.DataFrame:
        r = self.r
        return pd.DataFrame({
            'Item': ['Receita Mercado Zonal', 'Receita Contratos',
                     '(-) Custo Combustível', '(-) Custo Startup',
                     '(-) Multa Contratos', '(-) Custo Déficit Global',
                     '(-) Custo Curtailment', '= LUCRO LÍQUIDO'],
            'Valor (€)': [r.receita_mercado_total, r.receita_contratos_total,
                          -r.custo_combustivel_total, -r.custo_startup_total,
                          -r.multa_total, -r.custo_deficit_total,
                          -r.custo_curtailment_total, r.lucro_total]})

    def schedule_uc_dataframe(self) -> pd.DataFrame:
        data = {g.ID_Gerador: g.U_schedule for g in self.r.geradores}
        df = pd.DataFrame(data)
        df.index = [f"H{s.hora:02d}_D{s.dia+1}" for s in self.r.snapshots]
        return df

    def lucro_por_usina(self) -> pd.DataFrame:
        rows = []
        for g in self.r.geradores:
            rows.append({
                'Gerador': g.ID_Gerador, 'Barra': g.ID_Barra, 'Zona': g.Zona,
                'Horas Ligado': g.horas_ligado, 'Partidas': g.num_partidas,
                'Energia (pu·h)': sum(g.PGER),
                'Receita Mercado (€)': g.receita_mercado,
                'Custo Combustível (€)': g.custo_combustivel,
                'Custo Startup (€)': g.custo_startup,
                'Lucro (€)': g.lucro_total})
        return pd.DataFrame(rows).sort_values('Lucro (€)', ascending=False)

    def analise_contratos(self) -> pd.DataFrame:
        rows = []
        for c in self.r.contratos:
            rows.append({
                'Contrato': c.ID_Contrato, 'Zona': c.Zona,
                'Preço (€/MWh)': c.Preco_contrato, 'Multa (€/MWh)': c.Multa,
                'Amount (pu)': c.Amount_contratado,
                'Taxa Atendimento (%)': round(c.taxa_atendimento, 2),
                'Déficit Total (pu)': round(c.deficit_total, 4),
                'Receita (€)': round(c.receita_total, 2),
                'Multa Paga (€)': round(c.multa_total, 2),
                'Custo Cliente (€)': round(c.custo_efetivo_cliente, 2)})
        return pd.DataFrame(rows)

    def analise_curtailment(self) -> pd.DataFrame:
        rows = []
        for s in self.r.snapshots:
            total_disp = sum(s.PGWIND_disponivel)
            total_ger = sum(s.PGWIND)
            total_cort = sum(s.CURTAILMENT)
            taxa = (total_cort / total_disp * 100) if total_disp > 0 else 0.0
            rows.append({'Hora': s.hora, 'Dia': s.dia + 1,
                         'Disponível (pu)': round(total_disp, 4),
                         'Gerado (pu)': round(total_ger, 4),
                         'Cortado (pu)': round(total_cort, 4),
                         'Taxa Corte (%)': round(taxa, 2),
                         'Custo Oportunidade (€)': round(s.custo_curtailment, 2)})
        return pd.DataFrame(rows)

    def analise_precos_zonais(self) -> pd.DataFrame:
        rows = []
        for s in self.r.snapshots:
            row = {'Hora': s.hora, 'Dia': s.dia + 1}
            for i, lam in enumerate(s.LAMBDA_ZONA):
                row[f"Zona_{i+1}"] = round(lam, 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def analise_deficit(self) -> pd.DataFrame:
        rows = []
        for s in self.r.snapshots:
            total_def = sum(s.DEFICIT)
            rows.append({'Hora': s.hora, 'Dia': s.dia + 1,
                         'Déficit Total (pu)': round(total_def, 6),
                         'Custo Déficit (€)': round(s.custo_deficit, 2)})
        return pd.DataFrame(rows)

    def analise_balanco_potencia(self) -> pd.DataFrame:
        rows = []
        for s in self.r.snapshots:
            rows.append({
                'Dia': s.dia + 1, 'Hora': s.hora,
                'Geração UTE (pu)': round(sum(s.PGER), 4),
                'Geração Wind (pu)': round(sum(s.PGWIND), 4),
                'Geração Total (pu)': round(s.total_geracao, 4),
                'Demanda (pu)': round(s.total_demanda, 4),
                'Déficit Global (pu)': round(s.total_deficit, 4),
                'Curtailment (pu)': round(s.total_curtailment, 4),
                'Balanço (pu)': round(s.balanco, 6),
                'Carga Atendida (%)': round(
                    (1 - s.total_deficit / s.total_demanda) * 100
                    if s.total_demanda > 0 else 100, 2)})
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # [FIX-2] analise_fluxo_linhas: DataFrame vazio COM colunas
    # -------------------------------------------------------------------------
    def analise_fluxo_linhas(self) -> pd.DataFrame:
        colunas = ['Linha', 'Carreg. Médio (%)', 'Carreg. Máximo (%)',
                   'Horas > 80%', 'Horas > 95%',
                   'Fluxo Médio (pu)', 'Fluxo Máx (pu)', 'Horas Ativo']

        if not self.r.snapshots or not self.r.snapshots[0].FLUXO_LIN:
            return pd.DataFrame(columns=colunas)

        NLIN = len(self.r.snapshots[0].FLUXO_LIN)
        fluxo_mat = np.array([s.FLUXO_LIN for s in self.r.snapshots])
        loading_mat = np.array([s.LINE_LOADING for s in self.r.snapshots])

        rows = []
        for l in range(NLIN):
            loading_l = loading_mat[:, l]
            fluxo_l = fluxo_mat[:, l]
            rows.append({
                'Linha': l + 1,
                'Carreg. Médio (%)': round(float(np.mean(loading_l)), 2),
                'Carreg. Máximo (%)': round(float(np.max(loading_l)), 2),
                'Horas > 80%': int(np.sum(loading_l > 80)),
                'Horas > 95%': int(np.sum(loading_l > 95)),
                'Fluxo Médio (pu)': round(float(np.mean(np.abs(fluxo_l))), 4),
                'Fluxo Máx (pu)': round(float(np.max(np.abs(fluxo_l))), 4),
                'Horas Ativo': int(np.sum(np.abs(fluxo_l) > 1e-6))})

        if not rows:
            return pd.DataFrame(columns=colunas)

        return pd.DataFrame(rows).sort_values(
            'Carreg. Máximo (%)', ascending=False)

    # -------------------------------------------------------------------------
    # [FIX-1] analise_deficit_por_barra: DataFrame vazio COM colunas
    # -------------------------------------------------------------------------
    def analise_deficit_por_barra(self) -> pd.DataFrame:
        colunas = ['Barra', 'Déficit Total (pu·h)',
                   'Déficit Médio (pu)', 'Horas com Déficit']

        if not self.r.snapshots:
            return pd.DataFrame(columns=colunas)

        T = self.T
        NBAR = len(self.r.snapshots[0].DEFICIT)
        deficit_barra = np.zeros(NBAR)
        for s in self.r.snapshots:
            deficit_barra += np.array(s.DEFICIT)

        rows = []
        for b in range(NBAR):
            if deficit_barra[b] > 1e-6:
                rows.append({
                    'Barra': b + 1,
                    'Déficit Total (pu·h)': round(float(deficit_barra[b]), 4),
                    'Déficit Médio (pu)': round(float(deficit_barra[b] / T), 4),
                    'Horas com Déficit': int(sum(
                        1 for s in self.r.snapshots if s.DEFICIT[b] > 1e-6))})

        if not rows:
            return pd.DataFrame(columns=colunas)

        return pd.DataFrame(rows).sort_values(
            'Déficit Total (pu·h)', ascending=False)

    def diagnostico_deficit(self) -> pd.DataFrame:
        rows = []
        for s in self.r.snapshots:
            preco_medio = np.mean(s.LAMBDA_ZONA) if s.LAMBDA_ZONA else 0.0
            lin_cong = sum(1 for x in s.LINE_LOADING if x > 80)
            lin_crit = sum(1 for x in s.LINE_LOADING if x > 95)
            rows.append({
                'Dia': s.dia + 1, 'Hora': s.hora,
                'Déficit Global (pu)': round(s.total_deficit, 4),
                'Curtailment (pu)': round(s.total_curtailment, 4),
                'Preço Médio (€/MWh)': round(preco_medio, 2),
                'Linhas >80%': lin_cong,
                'Linhas >95%': lin_crit,
                'Geração Total (pu)': round(s.total_geracao, 4),
                'Demanda (pu)': round(s.total_demanda, 4)})
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # [FIX-3] analise_atendimento_horario: DataFrame vazio COM colunas
    # -------------------------------------------------------------------------
    def analise_atendimento_horario(self) -> pd.DataFrame:
        colunas = ['Dia', 'Hora', 'Contratado (pu)', 'Entregue (pu)',
                   'Déficit Contratual (pu)', 'Taxa Atendimento (%)', 'Multa (€)']

        if not self.r.snapshots:
            return pd.DataFrame(columns=colunas)

        rows = []
        for s in self.r.snapshots:
            total_cont = sum(c.Amount_contratado for c in self.r.contratos)
            entregue = sum(s.CONT_ENTREGA)
            deficit = sum(s.CONT_DEFICIT)
            rows.append({
                'Dia': s.dia + 1, 'Hora': s.hora,
                'Contratado (pu)': round(total_cont, 4),
                'Entregue (pu)': round(entregue, 4),
                'Déficit Contratual (pu)': round(deficit, 4),
                'Taxa Atendimento (%)': round(
                    entregue / total_cont * 100 if total_cont > 0 else 100, 2),
                'Multa (€)': round(s.multa_contratos, 2)})
        return pd.DataFrame(rows)

    def resumo_executivo(self) -> str:
        r = self.r
        linhas = [
            "=" * 70,
            "RESUMO EXECUTIVO - PBUC (Profit-Based Unit Commitment)",
            "=" * 70,
            f"Horizonte                  : {self.T} horas",
            f"Geradores                  : {len(r.geradores)}",
            f"Contratos                  : {len(r.contratos)}",
            "",
            "--- DEMONSTRATIVO DE RESULTADO ---",
            f"Receita Mercado Zonal      : € {r.receita_mercado_total:>15,.2f}",
            f"Receita Contratos          : € {r.receita_contratos_total:>15,.2f}",
            f"(-) Custo Combustível      : € {-r.custo_combustivel_total:>15,.2f}",
            f"(-) Custo Startup          : € {-r.custo_startup_total:>15,.2f}",
            f"(-) Multa Contratos        : € {-r.multa_total:>15,.2f}",
            f"(-) Custo Déficit          : € {-r.custo_deficit_total:>15,.2f}",
            f"(-) Custo Curtailment      : € {-r.custo_curtailment_total:>15,.2f}",
            "-" * 70,
            f"= LUCRO LÍQUIDO            : € {r.lucro_total:>15,.2f}",
            ""]
        total_def = sum(s.total_deficit for s in r.snapshots)
        total_dem = sum(s.total_demanda for s in r.snapshots)
        total_curt = sum(s.total_curtailment for s in r.snapshots)
        total_ger = sum(s.total_geracao for s in r.snapshots)
        pct_def = (total_def / total_dem * 100) if total_dem > 0 else 0.0
        linhas += [
            "--- BALANÇO DE POTÊNCIA (TOTAL) ---",
            f"Geração Total              : {total_ger:>15,.4f} pu·h",
            f"Demanda Total              : {total_dem:>15,.4f} pu·h",
            f"Déficit Global             : {total_def:>15,.4f} pu·h "
            f"({pct_def:.2f}% da demanda)",
            f"Curtailment Total          : {total_curt:>15,.4f} pu·h",
            ""]
        df_lin = self.analise_fluxo_linhas()
        if not df_lin.empty:
            linhas.append("--- TOP 5 LINHAS MAIS CARREGADAS ---")
            for _, row in df_lin.head(5).iterrows():
                linhas.append(
                    f"  Linha {int(row['Linha']):>3d} : "
                    f"máx {row['Carreg. Máximo (%)']:>6.1f}% | "
                    f"méd {row['Carreg. Médio (%)']:>6.1f}% | "
                    f"{int(row['Horas > 80%']):>3d}h >80%")
            linhas.append("")
        linhas.append("--- TOP 3 USINAS MAIS LUCRATIVAS ---")
        for _, row in self.lucro_por_usina().head(3).iterrows():
            linhas.append(
                f"  {row['Gerador']} (Zona {row['Zona']}) : "
                f"€ {row['Lucro (€)']:>12,.2f} | "
                f"{row['Horas Ligado']}h | {row['Partidas']} partidas")
        linhas.append("")
        linhas.append("--- CONTRATOS: TAXA DE ATENDIMENTO ---")
        for c in r.contratos:
            linhas.append(
                f"  {c.ID_Contrato} (Zona {c.Zona}) : "
                f"{c.taxa_atendimento:.1f}% atendido | "
                f"Multa € {c.multa_total:,.2f}")
        linhas.append("=" * 70)
        return "\n".join(linhas)


# =============================================================================
# 4. EXPORTADOR
# =============================================================================
class PBUC_Exporter:
    def __init__(self, resultado: PBUC_Result,
                 output_dir: str = "DATA/output/PBUC"):
        self.r = resultado
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = PBUC_Analyzer(resultado)

    def export_csv(self):
        def safe_to_csv(df, name):
            if df is None or df.empty:
                print(f"  (sem dados para {name} — CSV não gerado)")
                return
            df.to_csv(self.output_dir / f"{name}.csv", index=False)

        safe_to_csv(self.analyzer.relatorio_lucro(), "dre_lucro")
        safe_to_csv(self.analyzer.schedule_uc_dataframe(), "schedule_uc")
        safe_to_csv(self.analyzer.lucro_por_usina(), "lucro_por_usina")
        safe_to_csv(self.analyzer.analise_contratos(), "analise_contratos")
        safe_to_csv(self.analyzer.analise_curtailment(), "analise_curtailment")
        safe_to_csv(self.analyzer.analise_precos_zonais(), "precos_zonais")
        safe_to_csv(self.analyzer.analise_deficit(), "analise_deficit")
        safe_to_csv(self.analyzer.analise_balanco_potencia(), "balanco_potencia")
        safe_to_csv(self.analyzer.analise_fluxo_linhas(), "fluxo_linhas")
        safe_to_csv(self.analyzer.analise_deficit_por_barra(), "deficit_por_barra")
        safe_to_csv(self.analyzer.diagnostico_deficit(), "diagnostico_deficit")
        safe_to_csv(self.analyzer.analise_atendimento_horario(), "atendimento_horario")
        print(f"✓ CSV exportado em: {self.output_dir}")

    def export_json(self):
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Tipo não serializável: {type(obj)}")
        data = asdict(self.r)
        path = self.output_dir / "pbuc_resultado.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=default_serializer)
        print(f"✓ JSON exportado em: {path}")

    def export_sqlite(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(self.output_dir / "pbuc_resultados.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dia INTEGER, hora INTEGER, dia_semana INTEGER,
            lucro_horario REAL, receita_mercado REAL, receita_contratos REAL,
            custo_combustivel REAL, custo_startup REAL,
            multa_contratos REAL, custo_deficit REAL, custo_curtailment REAL,
            total_geracao REAL, total_demanda REAL,
            total_deficit REAL, total_curtailment REAL, total_wind REAL,
            balanco REAL,
            PLOAD TEXT, PGER TEXT, QGER TEXT, PGWIND TEXT, CURTAILMENT TEXT,
            U_schedule TEXT, LAMBDA_ZONA TEXT,
            CONT_ENTREGA TEXT, CONT_DEFICIT TEXT,
            FLUXO_LIN TEXT, LINE_LOADING TEXT, ANG TEXT, V TEXT
        )''')

        for s in self.r.snapshots:
            cursor.execute('''
            INSERT INTO pbuc_snapshots (
                dia, hora, dia_semana, lucro_horario,
                receita_mercado, receita_contratos,
                custo_combustivel, custo_startup,
                multa_contratos, custo_deficit, custo_curtailment,
                total_geracao, total_demanda, total_deficit,
                total_curtailment, total_wind, balanco,
                PLOAD, PGER, QGER, PGWIND, CURTAILMENT,
                U_schedule, LAMBDA_ZONA,
                CONT_ENTREGA, CONT_DEFICIT,
                FLUXO_LIN, LINE_LOADING, ANG, V
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                s.dia, s.hora, s.dia_semana, s.lucro_horario,
                s.receita_mercado, s.receita_contratos,
                s.custo_combustivel, s.custo_startup,
                s.multa_contratos, s.custo_deficit, s.custo_curtailment,
                s.total_geracao, s.total_demanda, s.total_deficit,
                s.total_curtailment, s.total_wind, s.balanco,
                json.dumps(s.PLOAD), json.dumps(s.PGER), json.dumps(s.QGER),
                json.dumps(s.PGWIND), json.dumps(s.CURTAILMENT),
                json.dumps(s.U_schedule), json.dumps(s.LAMBDA_ZONA),
                json.dumps(s.CONT_ENTREGA), json.dumps(s.CONT_DEFICIT),
                json.dumps(s.FLUXO_LIN), json.dumps(s.LINE_LOADING),
                json.dumps(s.ANG), json.dumps(s.V)
            ))

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_geradores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Gerador TEXT, ID_Barra INTEGER, Zona TEXT,
            horas_ligado INTEGER, num_partidas INTEGER,
            receita_mercado REAL, custo_combustivel REAL,
            custo_startup REAL, lucro_total REAL)''')
        for g in self.r.geradores:
            cursor.execute('''
            INSERT INTO pbuc_geradores (
                ID_Gerador, ID_Barra, Zona, horas_ligado, num_partidas,
                receita_mercado, custo_combustivel, custo_startup, lucro_total
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ''', (g.ID_Gerador, g.ID_Barra, g.Zona, g.horas_ligado,
                  g.num_partidas, g.receita_mercado, g.custo_combustivel,
                  g.custo_startup, g.lucro_total))

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_contratos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Contrato TEXT, Zona TEXT, Preco_contrato REAL, Multa REAL,
            Amount_contratado REAL, taxa_atendimento REAL, deficit_total REAL,
            receita_total REAL, multa_total REAL, custo_efetivo_cliente REAL)''')
        for c in self.r.contratos:
            cursor.execute('''
            INSERT INTO pbuc_contratos (
                ID_Contrato, Zona, Preco_contrato, Multa,
                Amount_contratado, taxa_atendimento, deficit_total,
                receita_total, multa_total, custo_efetivo_cliente
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', (c.ID_Contrato, c.Zona, c.Preco_contrato, c.Multa,
                  c.Amount_contratado, c.taxa_atendimento, c.deficit_total,
                  c.receita_total, c.multa_total, c.custo_efetivo_cliente))

        conn.commit()
        conn.close()
        print(f"✓ SQLite exportado em: {db_path}")

    def export_txt(self):
        path = self.output_dir / "resumo_executivo.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.analyzer.resumo_executivo())
        print(f"✓ TXT exportado em: {path}")


# =============================================================================
# 5. COMPARADOR
# =============================================================================
class PBUC_Comparator:
    def __init__(self, res_pto: PBUC_Result, res_baseline: PBUC_Result,
                 nome_pto: str = "PtO (HGB)", nome_base: str = "Baseline"):
        self.pto = res_pto
        self.base = res_baseline
        self.nome_pto = nome_pto
        self.nome_base = nome_base

    def comparar(self) -> pd.DataFrame:
        metricas = {
            'Lucro Total (€)': (self.pto.lucro_total, self.base.lucro_total),
            'Receita Mercado (€)': (self.pto.receita_mercado_total,
                                    self.base.receita_mercado_total),
            'Receita Contratos (€)': (self.pto.receita_contratos_total,
                                      self.base.receita_contratos_total),
            'Custo Combustível (€)': (self.pto.custo_combustivel_total,
                                      self.base.custo_combustivel_total),
            'Custo Startup (€)': (self.pto.custo_startup_total,
                                  self.base.custo_startup_total),
            'Multa Total (€)': (self.pto.multa_total, self.base.multa_total),
            'Custo Curtailment (€)': (self.pto.custo_curtailment_total,
                                      self.base.custo_curtailment_total)}
        rows = []
        for nome, (v_pto, v_base) in metricas.items():
            delta = v_pto - v_base
            pct = (delta / abs(v_base) * 100) if v_base != 0 else float('inf')
            rows.append({'Métrica': nome, self.nome_pto: round(v_pto, 2),
                         self.nome_base: round(v_base, 2),
                         'Δ': round(delta, 2),
                         'Δ%': round(pct, 2) if pct != float('inf') else 'N/A'})
        return pd.DataFrame(rows)

    def report(self) -> str:
        df = self.comparar()
        return "\n".join(["=" * 80,
                          f"COMPARAÇÃO: {self.nome_pto} vs. {self.nome_base}",
                          "=" * 80, df.to_string(index=False), "=" * 80])


# =============================================================================
# 6. DBHandler ESTENDIDO
# =============================================================================
class PBUC_DBHandler(OPF_DBHandler):
    def __init__(self, db_path: str = 'DATA/output/pbuc.db'):
        super().__init__(db_path)

    def create_tables(self):
        super().create_tables()
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, timestamp TEXT,
            dia INTEGER, hora INTEGER, dia_semana INTEGER,
            lucro_horario REAL, receita_mercado REAL, receita_contratos REAL,
            custo_combustivel REAL, custo_startup REAL,
            multa_contratos REAL, custo_deficit REAL, custo_curtailment REAL,
            total_geracao REAL, total_demanda REAL,
            total_deficit REAL, total_curtailment REAL, total_wind REAL,
            balanco REAL,
            PLOAD_json TEXT, PGER_json TEXT, QGER_json TEXT, PGWIND_json TEXT,
            PGWIND_disponivel_json TEXT, CURTAILMENT_json TEXT,
            U_schedule_json TEXT, V_startup_json TEXT, DEFICIT_json TEXT,
            LAMBDA_ZONA_json TEXT, CONT_ENTREGA_json TEXT, CONT_DEFICIT_json TEXT,
            FLUXO_LIN_json TEXT, LINE_LOADING_json TEXT, ANG_json TEXT,
            V_json TEXT,
            UNIQUE(cen_id, dia, hora)
        )''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pbuc_snap_cen '
                       'ON pbuc_snapshots(cen_id)')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_fluxo_linhas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, dia INTEGER, hora INTEGER,
            linha_id INTEGER, de_barra INTEGER, para_barra INTEGER,
            fluxo_pu REAL, limite_pu REAL, loading_pct REAL,
            UNIQUE(cen_id, dia, hora, linha_id)
        )''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pbuc_fluxo_cen '
                       'ON pbuc_fluxo_linhas(cen_id)')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_balanco_horario (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, dia INTEGER, hora INTEGER,
            geracao_ute REAL, geracao_wind REAL, geracao_total REAL,
            demanda REAL, deficit_global REAL, curtailment REAL,
            balanco REAL,
            UNIQUE(cen_id, dia, hora)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_deficit_barra (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, barra INTEGER,
            deficit_total_puh REAL, deficit_medio_pu REAL,
            horas_com_deficit INTEGER,
            UNIQUE(cen_id, barra)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_geradores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, ID_Gerador TEXT, ID_Barra INTEGER, Zona TEXT,
            horas_ligado INTEGER, num_partidas INTEGER, energia_total REAL,
            receita_mercado REAL, custo_combustivel REAL,
            custo_startup REAL, lucro_total REAL,
            U_schedule_json TEXT, V_startup_json TEXT, PGER_json TEXT,
            UNIQUE(cen_id, ID_Gerador)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_contratos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, ID_Contrato TEXT, Zona TEXT,
            Preco_contrato REAL, Multa REAL, Amount_contratado REAL,
            taxa_atendimento REAL, deficit_total REAL,
            receita_total REAL, multa_total REAL, custo_efetivo_cliente REAL,
            entrega_json TEXT, deficit_json TEXT,
            UNIQUE(cen_id, ID_Contrato)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_precos_zonais (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, dia INTEGER, hora INTEGER, zona TEXT,
            lambda_previsto REAL, lambda_p10 REAL,
            lambda_p50 REAL, lambda_p90 REAL,
            UNIQUE(cen_id, dia, hora, zona)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_uc_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT NOT NULL, dia INTEGER, hora INTEGER,
            ID_Gerador TEXT, U_status INTEGER, V_startup INTEGER,
            W_shutdown INTEGER, PGER REAL, QGER REAL,
            UNIQUE(cen_id, dia, hora, ID_Gerador)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbuc_resumo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT UNIQUE NOT NULL, timestamp TEXT,
            horizonte_horas INTEGER, num_geradores INTEGER, num_contratos INTEGER,
            lucro_total REAL, receita_mercado_total REAL,
            receita_contratos_total REAL, custo_combustivel_total REAL,
            custo_startup_total REAL, multa_total REAL,
            custo_deficit_total REAL, custo_curtailment_total REAL,
            curtailment_total_pu REAL, deficit_total_pu REAL,
            geracao_total_puh REAL, demanda_total_puh REAL
        )''')

        conn.commit()
        conn.close()
        print(f"✓ Tabelas PBUC criadas em {self.db_path}")

    def save_pbuc_result(self, modelo_pbuc, sistema, mercado: dict,
                         cen_id: str) -> None:
        extrator = PBUC_ResultExtractor(modelo_pbuc, sistema, mercado)
        resultado = extrator.extract()
        T = modelo_pbuc.horizon_time
        timestamp = datetime.now().isoformat()

        zonas = {z['ID_Zona']: z for z in mercado.get('ZONAS', [])}
        zona_list = list(zonas.keys())

        conn = self.connect()
        cursor = conn.cursor()

        for s in resultado.snapshots:
            cursor.execute('''
            INSERT OR REPLACE INTO pbuc_snapshots (
                cen_id, timestamp, dia, hora, dia_semana,
                lucro_horario, receita_mercado, receita_contratos,
                custo_combustivel, custo_startup,
                multa_contratos, custo_deficit, custo_curtailment,
                total_geracao, total_demanda, total_deficit,
                total_curtailment, total_wind, balanco,
                PLOAD_json, PGER_json, QGER_json, PGWIND_json,
                PGWIND_disponivel_json, CURTAILMENT_json,
                U_schedule_json, V_startup_json, DEFICIT_json,
                LAMBDA_ZONA_json, CONT_ENTREGA_json, CONT_DEFICIT_json,
                FLUXO_LIN_json, LINE_LOADING_json, ANG_json, V_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (cen_id, timestamp, s.dia, s.hora, s.dia_semana,
                  s.lucro_horario, s.receita_mercado, s.receita_contratos,
                  s.custo_combustivel, s.custo_startup,
                  s.multa_contratos, s.custo_deficit, s.custo_curtailment,
                  s.total_geracao, s.total_demanda, s.total_deficit,
                  s.total_curtailment, s.total_wind, s.balanco,
                  json.dumps(s.PLOAD), json.dumps(s.PGER), json.dumps(s.QGER),
                  json.dumps(s.PGWIND), json.dumps(s.PGWIND_disponivel),
                  json.dumps(s.CURTAILMENT), json.dumps(s.U_schedule),
                  json.dumps(s.V_startup), json.dumps(s.DEFICIT),
                  json.dumps(s.LAMBDA_ZONA), json.dumps(s.CONT_ENTREGA),
                  json.dumps(s.CONT_DEFICIT), json.dumps(s.FLUXO_LIN),
                  json.dumps(s.LINE_LOADING), json.dumps(s.ANG),
                  json.dumps(s.V)))

            for g_idx, g in enumerate(resultado.geradores):
                qg = s.QGER[g_idx] if g_idx < len(s.QGER) else 0.0
                cursor.execute('''
                INSERT OR REPLACE INTO pbuc_uc_schedule (
                    cen_id, dia, hora, ID_Gerador,
                    U_status, V_startup, W_shutdown, PGER, QGER
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ''', (cen_id, s.dia, s.hora, g.ID_Gerador,
                      s.U_schedule[g_idx], s.V_startup[g_idx],
                      s.W_shutdown[g_idx], s.PGER[g_idx], qg))

            for z_idx, zid in enumerate(zona_list):
                lam = s.LAMBDA_ZONA[z_idx]
                cursor.execute('''
                INSERT OR REPLACE INTO pbuc_precos_zonais (
                    cen_id, dia, hora, zona,
                    lambda_previsto, lambda_p10, lambda_p50, lambda_p90
                ) VALUES (?,?,?,?,?,?,?,?)
                ''', (cen_id, s.dia, s.hora, zid,
                      lam, lam * 0.85, lam, lam * 1.15))

            if s.FLUXO_LIN:
                lf = np.array(getattr(sistema, 'line_fr', []), dtype=int)
                lt = np.array(getattr(sistema, 'line_to', []), dtype=int)
                if len(lf) > 0 and lf.max() >= sistema.NBAR:
                    lf = lf - 1
                    lt = lt - 1
                for l_idx, f_val in enumerate(s.FLUXO_LIN):
                    loading = (s.LINE_LOADING[l_idx]
                               if l_idx < len(s.LINE_LOADING) else 0.0)
                    from_bus = int(lf[l_idx]) + 1 if l_idx < len(lf) else l_idx
                    to_bus = int(lt[l_idx]) + 1 if l_idx < len(lt) else l_idx
                    lim = modelo_pbuc.line_flow_max[l_idx]
                    cursor.execute('''
                    INSERT OR REPLACE INTO pbuc_fluxo_linhas (
                        cen_id, dia, hora, linha_id, de_barra, para_barra,
                        fluxo_pu, limite_pu, loading_pct
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    ''', (cen_id, s.dia, s.hora, l_idx, from_bus, to_bus,
                          f_val, lim, loading))

            cursor.execute('''
            INSERT OR REPLACE INTO pbuc_balanco_horario (
                cen_id, dia, hora, geracao_ute, geracao_wind,
                geracao_total, demanda, deficit_global, curtailment, balanco
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', (cen_id, s.dia, s.hora,
                  sum(s.PGER), sum(s.PGWIND),
                  s.total_geracao, s.total_demanda, s.total_deficit,
                  s.total_curtailment, s.balanco))

        for g in resultado.geradores:
            cursor.execute('''
            INSERT OR REPLACE INTO pbuc_geradores (
                cen_id, ID_Gerador, ID_Barra, Zona,
                horas_ligado, num_partidas, energia_total,
                receita_mercado, custo_combustivel, custo_startup, lucro_total,
                U_schedule_json, V_startup_json, PGER_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (cen_id, g.ID_Gerador, g.ID_Barra, g.Zona,
                  g.horas_ligado, g.num_partidas, sum(g.PGER),
                  g.receita_mercado, g.custo_combustivel, g.custo_startup,
                  g.lucro_total,
                  json.dumps(g.U_schedule), json.dumps(g.V_startup),
                  json.dumps(g.PGER)))

        for c in resultado.contratos:
            cursor.execute('''
            INSERT OR REPLACE INTO pbuc_contratos (
                cen_id, ID_Contrato, Zona, Preco_contrato, Multa,
                Amount_contratado, taxa_atendimento, deficit_total,
                receita_total, multa_total, custo_efetivo_cliente,
                entrega_json, deficit_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (cen_id, c.ID_Contrato, c.Zona, c.Preco_contrato, c.Multa,
                  c.Amount_contratado, c.taxa_atendimento, c.deficit_total,
                  c.receita_total, c.multa_total, c.custo_efetivo_cliente,
                  json.dumps(c.entrega), json.dumps(c.deficit)))

        # ---------------------------------------------------------------------
        # [FIX-5] Salvar déficit por barra — só se houver déficit
        # ---------------------------------------------------------------------
        analyzer = PBUC_Analyzer(resultado)
        df_def_barra = analyzer.analise_deficit_por_barra()
        if not df_def_barra.empty:
            for _, row in df_def_barra.iterrows():
                cursor.execute('''
                INSERT OR REPLACE INTO pbuc_deficit_barra (
                    cen_id, barra, deficit_total_puh, deficit_medio_pu,
                    horas_com_deficit
                ) VALUES (?,?,?,?,?)
                ''', (cen_id, int(row['Barra']),
                      float(row['Déficit Total (pu·h)']),
                      float(row['Déficit Médio (pu)']),
                      int(row['Horas com Déficit'])))
            print(f"  ✓ Déficit salvo em {len(df_def_barra)} barras")
        else:
            print("  (sem déficit — tabela pbuc_deficit_barra vazia)")

        curt_total = sum(s.total_curtailment for s in resultado.snapshots)
        def_total = sum(s.total_deficit for s in resultado.snapshots)
        ger_total = sum(s.total_geracao for s in resultado.snapshots)
        dem_total = sum(s.total_demanda for s in resultado.snapshots)
        cursor.execute('''
        INSERT OR REPLACE INTO pbuc_resumo (
            cen_id, timestamp, horizonte_horas, num_geradores, num_contratos,
            lucro_total, receita_mercado_total, receita_contratos_total,
            custo_combustivel_total, custo_startup_total,
            multa_total, custo_deficit_total, custo_curtailment_total,
            curtailment_total_pu, deficit_total_pu,
            geracao_total_puh, demanda_total_puh
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (cen_id, timestamp, T, len(resultado.geradores),
              len(resultado.contratos),
              resultado.lucro_total, resultado.receita_mercado_total,
              resultado.receita_contratos_total,
              resultado.custo_combustivel_total,
              resultado.custo_startup_total,
              resultado.multa_total, resultado.custo_deficit_total,
              resultado.custo_curtailment_total,
              curt_total, def_total, ger_total, dem_total))

        conn.commit()
        conn.close()
        print(f"✓ Resultado PBUC salvo (cen_id={cen_id}) em {self.db_path}")

    # --- CONSULTAS ---
    def query_snapshots(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_snapshots WHERE cen_id=? ORDER BY dia, hora",
            conn, params=(cen_id,)); conn.close(); return df

    def query_geradores(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_geradores WHERE cen_id=? "
            "ORDER BY lucro_total DESC", conn, params=(cen_id,))
        conn.close(); return df

    def query_contratos(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_contratos WHERE cen_id=?",
            conn, params=(cen_id,)); conn.close(); return df

    def query_precos_zonais(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_precos_zonais WHERE cen_id=? "
            "ORDER BY dia, hora, zona", conn, params=(cen_id,))
        conn.close(); return df

    def query_uc_schedule(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_uc_schedule WHERE cen_id=? "
            "ORDER BY dia, hora, ID_Gerador", conn, params=(cen_id,))
        conn.close(); return df

    def query_resumo(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_resumo WHERE cen_id=?",
            conn, params=(cen_id,)); conn.close(); return df

    def query_fluxo_linhas(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_fluxo_linhas WHERE cen_id=? "
            "ORDER BY dia, hora, linha_id", conn, params=(cen_id,))
        conn.close(); return df

    def query_balanco_horario(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_balanco_horario WHERE cen_id=? "
            "ORDER BY dia, hora", conn, params=(cen_id,))
        conn.close(); return df

    def query_deficit_barra(self, cen_id: str) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM pbuc_deficit_barra WHERE cen_id=? "
            "ORDER BY deficit_total_puh DESC", conn, params=(cen_id,))
        conn.close(); return df

    def list_cenarios(self) -> pd.DataFrame:
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT cen_id, timestamp, lucro_total, horizonte_horas "
            "FROM pbuc_resumo ORDER BY timestamp DESC", conn)
        conn.close(); return df


# =============================================================================
# 7. PLOTTER (COM CORREÇÕES)
# =============================================================================
class PBUC_Plotter:
    def __init__(self, db_handler: PBUC_DBHandler, cen_id: str,
                 output_dir: str = "DATA/output/figuras",
                 network_bus_pos: Optional[Dict[int, Tuple[float, float]]] = None):
        self.db = db_handler
        self.cen_id = cen_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.network_bus_pos = network_bus_pos

        self.df_snap = db_handler.query_snapshots(cen_id)
        self.df_ger = db_handler.query_geradores(cen_id)
        self.df_cont = db_handler.query_contratos(cen_id)
        self.df_precos = db_handler.query_precos_zonais(cen_id)
        self.df_uc = db_handler.query_uc_schedule(cen_id)
        self.df_resumo = db_handler.query_resumo(cen_id)
        self.df_fluxo = db_handler.query_fluxo_linhas(cen_id)
        self.df_balanco = db_handler.query_balanco_horario(cen_id)
        self.df_def_barra = db_handler.query_deficit_barra(cen_id)

        self.horas = self.df_snap['hora'].values
        self.dias = self.df_snap['dia'].values
        self.tempo = self.dias * 24 + self.horas

    def _salvar(self, fig, nome: str):
        path = self.output_dir / f"{nome}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ {path}")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    def plot_commitment_heatmap(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        if self.df_uc.empty:
            print("  (sem UC — pulando commitment heatmap)"); return
        if len(self.df_uc['dia'].unique()) > 1:
            pivot = self.df_uc.groupby(['hora', 'ID_Gerador'])['U_status'].mean().unstack()
        else:
            pivot = self.df_uc.pivot_table(
                index='hora', columns='ID_Gerador',
                values='U_status', aggfunc='first')
        fig, ax = plt.subplots(figsize=(14, max(4, 0.3 * len(pivot.columns))))
        cmap = ListedColormap(['#d62728', '#2ca02c'])
        im = ax.imshow(pivot.values.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns, fontsize=8)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{h:02d}h" for h in pivot.index], fontsize=7)
        ax.set_xlabel('Hora do dia'); ax.set_ylabel('Gerador')
        ax.set_title(f'Schedule UC (1=ligado, 0=desligado) — {self.cen_id}')
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Desligado', 'Ligado'])
        self._salvar(fig, f"{self.cen_id}_commitment_heatmap")

    def plot_contratos_atendimento(self):
        import matplotlib.pyplot as plt
        TOL = 1e-1   # ajuste conforme precisão desejada
        if self.df_snap.empty:
            return
        entrega_matrix = np.array([json.loads(x) for x in self.df_snap['CONT_ENTREGA_json']])
        deficit_matrix = np.array([json.loads(x) for x in self.df_snap['CONT_DEFICIT_json']])
        deficit_matrix = np.where(np.abs(deficit_matrix) < TOL, 0.0, deficit_matrix)
        labels = list(self.df_cont['ID_Contrato']) if not self.df_cont.empty else []
        cmap = plt.get_cmap('tab20')
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].stackplot(self.tempo, entrega_matrix.T, labels=labels,
                          colors=[cmap(i % 20) for i in range(len(labels))], alpha=0.85)
        axes[0].set_ylabel('Entrega (pu)')
        axes[0].set_title(f'Atendimento de Contratos — {self.cen_id}')
        axes[0].legend(loc='upper left', ncol=3, fontsize=7, frameon=False)
        axes[0].grid(True, alpha=0.3)
        axes[1].stackplot(self.tempo, deficit_matrix.T,
                          labels=[f"Déficit {l}" for l in labels],
                          colors=[cmap(i % 20) for i in range(len(labels))], alpha=0.85)
        axes[1].set_xlabel('Tempo (h)'); axes[1].set_ylabel('Déficit (pu)')
        axes[1].set_title('Déficit por Contrato (multa)')
        axes[1].legend(loc='upper left', ncol=3, fontsize=7, frameon=False)
        axes[1].grid(True, alpha=0.3)
        self._salvar(fig, f"{self.cen_id}_contratos_atendimento")

    def plot_precos_zonais(self):
        import matplotlib.pyplot as plt
        import numpy as np

        if self.df_precos.empty:
            return

        zonas = list(self.df_precos['zona'].unique())
        cmap = plt.get_cmap('tab10')

        fig, ax = plt.subplots(figsize=(14, 5))

        for i, z in enumerate(zonas):
            sub = (self.df_precos[self.df_precos['zona'] == z]
                   .sort_values(['dia', 'hora'])
                   .reset_index(drop=True))

            # Preços P50 ordenados de forma crescente (escadinha)
            p50_sorted = np.sort(sub['lambda_p50'].values)
            p10_sorted = np.sort(sub['lambda_p10'].values)
            p90_sorted = np.sort(sub['lambda_p90'].values)

            x = np.arange(len(p50_sorted))
            cor = cmap(i % 10)

            # Barras do P50 (escadinha crescente) lado a lado entre zonas
            width = 0.8 / len(zonas)
            offset = (i - (len(zonas) - 1) / 2) * width
            ax.bar(x + offset, p50_sorted, width=width,
                   color=cor, alpha=0.85, edgecolor='none',
                   label=f"{z} (P50)")

            # Faixa de incerteza P10–P90 como barras de erro
            yerr_low = p50_sorted - p10_sorted
            yerr_high = p90_sorted - p50_sorted
            ax.errorbar(x + offset, p50_sorted,
                        yerr=[yerr_low, yerr_high],
                        fmt='none', ecolor='black',
                        elinewidth=0.5, capsize=2, alpha=0.4)

        ax.set_xlabel('Ranking da hora (preço crescente)')
        ax.set_ylabel('Preço (€/MWh)')
        ax.set_title(f'Preços Zonais Previstos (HGB) — {self.cen_id}')
        ax.legend(loc='upper left', ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        self._salvar(fig, f"{self.cen_id}_precos_zonais")

    def plot_lucro_horario(self):
        import matplotlib.pyplot as plt
        if self.df_snap.empty:
            return
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax = axes[0]
        ax.bar(self.tempo, self.df_snap['receita_mercado'], label='Receita Mercado',
               color='#2ca02c', alpha=0.7, width=0.8)
        ax.bar(self.tempo, self.df_snap['receita_contratos'],
               bottom=self.df_snap['receita_mercado'], label='Receita Contratos',
               color='#1f77b4', alpha=0.7, width=0.8)
        ax.set_ylabel('Receita (€)'); ax.set_title(f'Receitas e Custos — {self.cen_id}')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        ax2 = axes[1]
        ax2.plot(self.tempo, self.df_snap['lucro_horario'], color='#d62728',
                 linewidth=2, marker='o', markersize=3, label='Lucro Horário')
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.fill_between(self.tempo, 0, self.df_snap['lucro_horario'],
                         where=self.df_snap['lucro_horario'] >= 0, color='#2ca02c', alpha=0.3)
        ax2.fill_between(self.tempo, 0, self.df_snap['lucro_horario'],
                         where=self.df_snap['lucro_horario'] < 0, color='#d62728', alpha=0.3)
        ax2.set_xlabel('Tempo (h)'); ax2.set_ylabel('Lucro (€)')
        ax2.legend(loc='upper left', fontsize=8); ax2.grid(True, alpha=0.3)
        self._salvar(fig, f"{self.cen_id}_lucro_horario")

    def plot_curtailment(self):
        import matplotlib.pyplot as plt
        if self.df_snap.empty:
            return
        if self.df_snap['PGWIND_disponivel_json'].iloc[0] == '[]':
            print("  (sem geração eólica — pulando curtailment)"); return
        disp = np.array([sum(json.loads(x)) for x in self.df_snap['PGWIND_disponivel_json']])
        ger = np.array([sum(json.loads(x)) for x in self.df_snap['PGWIND_json']])
        cort = np.array([sum(json.loads(x)) for x in self.df_snap['CURTAILMENT_json']])
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(self.tempo, 0, disp, color='#aec7e8', alpha=0.5, label='Disponível')
        ax.fill_between(self.tempo, 0, ger, color='#1f77b4', alpha=0.7, label='Gerado')
        ax.bar(self.tempo, cort, bottom=ger, color='#d62728', alpha=0.7,
               label='Cortado', width=0.8)
        ax.set_xlabel('Tempo (h)'); ax.set_ylabel('Potência eólica (pu)')
        ax.set_title(f'Curtailment Eólico — {self.cen_id}')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        self._salvar(fig, f"{self.cen_id}_curtailment")

    def plot_lucro_por_gerador(self):
        import matplotlib.pyplot as plt
        if self.df_ger.empty:
            return
        df = self.df_ger.sort_values('lucro_total', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
        cores = ['#2ca02c' if x >= 0 else '#d62728' for x in df['lucro_total']]
        ax.barh(df['ID_Gerador'], df['lucro_total'], color=cores, alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Lucro (€)'); ax.set_ylabel('Gerador')
        ax.set_title(f'Lucro Individual por Usina — {self.cen_id}')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(df['lucro_total']):
            ax.text(v + (abs(v) * 0.02), i, f"{v:,.0f}", va='center', fontsize=7)
        self._salvar(fig, f"{self.cen_id}_lucro_por_gerador")

    def plot_dre_resumo(self):
        import matplotlib.pyplot as plt
        if self.df_resumo.empty:
            return
        r = self.df_resumo.iloc[0]
        itens = [('Receita Mercado', r['receita_mercado_total']),
                 ('Receita Contratos', r['receita_contratos_total']),
                 ('Custo Combustível', -r['custo_combustivel_total']),
                 ('Custo Startup', -r['custo_startup_total']),
                 ('Multa Contratos', -r['multa_total']),
                 ('Custo Déficit', -r['custo_deficit_total']),
                 ('Custo Curtailment', -r['custo_curtailment_total']),
                 ('LUCRO LÍQUIDO', r['lucro_total'])]
        labels = [x[0] for x in itens]; vals = [x[1] for x in itens]
        fig, ax = plt.subplots(figsize=(12, 5))
        cores = ['#2ca02c' if v >= 0 else '#d62728' for v in vals]
        cores[-1] = '#1f77b4'
        bars = ax.barh(labels, vals, color=cores, alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('€'); ax.set_title(f'DRE — {self.cen_id}')
        ax.grid(True, alpha=0.3, axis='x')
        for b, v in zip(bars, vals):
            x_pos = v + (abs(v) * 0.02 if v >= 0 else -abs(v) * 0.05)
            ha = 'left' if v >= 0 else 'right'
            ax.text(x_pos, b.get_y() + b.get_height() / 2, f"€ {v:,.0f}",
                    va='center', ha=ha, fontsize=8)
        self._salvar(fig, f"{self.cen_id}_dre_resumo")

    def plot_precos_vs_commitment(self):
        import matplotlib.pyplot as plt
        if self.df_precos.empty or self.df_uc.empty:
            return
        fig, ax1 = plt.subplots(figsize=(14, 5))
        precos_med = self.df_precos.groupby(['dia', 'hora'])['lambda_p50'].mean().reset_index()
        precos_med['tempo'] = precos_med['dia'] * 24 + precos_med['hora']
        precos_med = precos_med.sort_values('tempo')
        ax1.plot(precos_med['tempo'], precos_med['lambda_p50'], color='#1f77b4',
                 linewidth=2, label='Preço zonal (médio P50)')
        ax1.set_xlabel('Tempo (h)'); ax1.set_ylabel('Preço (€/MWh)', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4'); ax1.grid(True, alpha=0.3)
        ligadas = self.df_uc.groupby(['dia', 'hora'])['U_status'].sum().reset_index()
        ligadas['tempo'] = ligadas['dia'] * 24 + ligadas['hora']
        ligadas = ligadas.sort_values('tempo')
        ax2 = ax1.twinx()
        ax2.bar(ligadas['tempo'], ligadas['U_status'], color='#ff7f0e',
                alpha=0.3, width=0.8, label='Usinas ligadas')
        ax2.set_ylabel('Nº usinas ligadas', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        fig.suptitle(f'Preço Zonal vs. Commitment — {self.cen_id}')
        fig.tight_layout()
        self._salvar(fig, f"{self.cen_id}_precos_vs_commitment")

    def plot_balanco_potencia(self):
        import matplotlib.pyplot as plt
        if self.df_balanco.empty:
            return
        df = self.df_balanco.sort_values(['dia', 'hora'])
        t = df['dia'].values * 24 + df['hora'].values
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
        ax = axes[0]
        ax.fill_between(t, 0, df['geracao_ute'], color='steelblue',
                        alpha=0.7, label='Geração UTE')
        ax.fill_between(t, df['geracao_ute'],
                        df['geracao_ute'] + df['geracao_wind'],
                        color='lightgreen', alpha=0.7, label='Geração Eólica')
        ax.bar(t, df['deficit_global'], bottom=df['geracao_total'],
               color='red', alpha=0.7, label='Déficit Global', width=0.8)
        ax.plot(t, df['demanda'], color='black', linewidth=2.5,
                marker='o', markersize=4, label='Demanda Total')
        ax.plot(t, df['geracao_total'] + df['deficit_global'], color='orange',
                linewidth=1.5, linestyle='--', label='Oferta Total')
        ax.set_ylabel('Potência (pu)')
        ax.set_title(f'Balanço de Potência Horário — Cenário {self.cen_id}')
        ax.legend(loc='upper left', ncol=3, fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
        ax2 = axes[1]
        w = 0.4
        ax2.bar(t - w / 2, df['deficit_global'], width=w,
                color='#d62728', alpha=0.85, label='Déficit')
        ax2.bar(t + w / 2, df['curtailment'], width=w,
                color='#ff7f0e', alpha=0.85, label='Curtailment')
        ax2.set_xlabel('Tempo (h)'); ax2.set_ylabel('Potência (pu)')
        ax2.set_title('Déficit vs Curtailment por hora')
        ax2.legend(loc='upper left', fontsize=8); ax2.grid(True, alpha=0.3)
        self._salvar(fig, f"{self.cen_id}_balanco_potencia")

    def plot_fluxo_linhas_heatmap(self):
        import matplotlib.pyplot as plt
        if self.df_fluxo.empty:
            print("  (sem dados de fluxo — pulando)"); return
        pivot = self.df_fluxo.pivot_table(
            index='hora', columns='linha_id', values='loading_pct',
            aggfunc='mean')
        fig, ax = plt.subplots(figsize=(14, 7))
        im = ax.imshow(pivot.values.T, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=100, interpolation='nearest')
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels([f"L{l}" for l in pivot.columns], fontsize=6)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{h:02d}h" for h in pivot.index], fontsize=7)
        ax.set_xlabel('Hora do dia'); ax.set_ylabel('Linha')
        ax.set_title(f'Carregamento das Linhas (%) — Cenário {self.cen_id}')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Carregamento (%)')
        cbar.ax.axhline(y=80, color='black', linestyle='--', linewidth=0.8)
        cbar.ax.axhline(y=100, color='black', linestyle='-', linewidth=0.8)
        self._salvar(fig, f"{self.cen_id}_fluxo_linhas_heatmap")

    def plot_deficit_por_barra(self):
        import matplotlib.pyplot as plt
        if self.df_def_barra.empty:
            print("  (sem déficit registrado — pulando)"); return
        df = self.df_def_barra.sort_values('deficit_total_puh', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
        ax.barh([f"B{int(b)}" for b in df['barra']],
                df['deficit_total_puh'], color='#d62728', alpha=0.85)
        ax.set_xlabel('Déficit Total (pu·h)'); ax.set_ylabel('Barra')
        ax.set_title(f'Déficit Acumulado por Barra — {self.cen_id}')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(df['deficit_total_puh']):
            ax.text(v + v * 0.02, i, f"{v:.3f}", va='center', fontsize=7)
        self._salvar(fig, f"{self.cen_id}_deficit_por_barra")

    def plot_diagnostico_deficit(self):
        import matplotlib.pyplot as plt
        if self.df_balanco.empty:
            return
        df = self.df_balanco.sort_values(['dia', 'hora']).copy()
        df['tempo'] = df['dia'] * 24 + df['hora']
        precos_med = self.df_precos.groupby(['dia', 'hora'])['lambda_p50'].mean().reset_index()
        precos_med['tempo'] = precos_med['dia'] * 24 + precos_med['hora']
        df = df.merge(precos_med[['tempo', 'lambda_p50']], on='tempo', how='left')
        if not self.df_fluxo.empty:
            lin_cong = self.df_fluxo[self.df_fluxo['loading_pct'] > 80] \
                .groupby(['dia', 'hora']).size().reset_index(name='lin_cong')
            lin_cong['tempo'] = lin_cong['dia'] * 24 + lin_cong['hora']
            df = df.merge(lin_cong[['tempo', 'lin_cong']], on='tempo', how='left')
            df['lin_cong'] = df['lin_cong'].fillna(0)
        else:
            df['lin_cong'] = 0
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ax1 = axes[0]
        ax1.bar(df['tempo'] - 0.2, df['deficit_global'], width=0.4,
                color='#d62728', alpha=0.8, label='Déficit (pu)')
        ax1.bar(df['tempo'] + 0.2, df['curtailment'], width=0.4,
                color='#ff7f0e', alpha=0.8, label='Curtailment (pu)')
        ax1.set_ylabel('Potência (pu)')
        ax1.set_title(f'Diagnóstico de Déficit — {self.cen_id}')
        ax1.legend(loc='upper left', fontsize=8); ax1.grid(True, alpha=0.3)
        ax2 = axes[1]
        ax2.plot(df['tempo'], df['lambda_p50'], color='#1f77b4',
                 linewidth=2, marker='o', markersize=3, label='Preço médio (€/MWh)')
        ax2.set_ylabel('Preço (€/MWh)')
        ax2.legend(loc='upper left', fontsize=8); ax2.grid(True, alpha=0.3)
        ax3 = axes[2]
        ax3.bar(df['tempo'], df['lin_cong'], color='#9467bd',
                alpha=0.8, width=0.8, label='Linhas >80%')
        ax3.set_xlabel('Tempo (h)'); ax3.set_ylabel('Nº linhas')
        ax3.legend(loc='upper left', fontsize=8); ax3.grid(True, alpha=0.3)
        fig.tight_layout()
        self._salvar(fig, f"{self.cen_id}_diagnostico_deficit")

    def plot_rede_carregamento(self, dia: int = 0, hora: int = 12):
        import matplotlib.pyplot as plt
        import networkx as nx
        if self.df_fluxo.empty:
            return
        df_h = self.df_fluxo[(self.df_fluxo['dia'] == dia) &
                             (self.df_fluxo['hora'] == hora)]
        if df_h.empty:
            print(f"  (sem dados de fluxo para dia={dia}, hora={hora})"); return
        G = nx.Graph()
        for _, row in df_h.iterrows():
            G.add_edge(int(row['de_barra']), int(row['para_barra']),
                       loading=row['loading_pct'], fluxo=row['fluxo_pu'])
        if self.network_bus_pos:
            pos = {n: self.network_bus_pos.get(n, (0, 0)) for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(12, 9))
        edges = list(G.edges())
        loadings = [G[u][v]['loading'] for u, v in edges]
        colors = ['green' if l < 50 else 'orange' if l < 80 else 'red'
                  for l in loadings]
        widths = [max(1.0, l / 30) for l in loadings]
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue',
                               edgecolors='black', linewidths=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors,
                               width=widths, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
        for (u, v) in edges:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(xm, ym, f"{G[u][v]['loading']:.0f}%",
                    fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        ax.set_title(f'Carregamento da Rede — {self.cen_id} | Dia {dia+1} H{hora:02d}')
        ax.axis('off')
        self._salvar(fig, f"{self.cen_id}_rede_d{dia+1}_h{hora:02d}")

    def plot_all(self):
        print(f"\n📊 Gerando gráficos para cenário {self.cen_id}...")
        self.plot_commitment_heatmap()
        self.plot_contratos_atendimento()
        self.plot_precos_zonais()
        self.plot_lucro_horario()
        self.plot_curtailment()
        self.plot_lucro_por_gerador()
        self.plot_dre_resumo()
        self.plot_precos_vs_commitment()
        self.plot_balanco_potencia()
        self.plot_fluxo_linhas_heatmap()
        self.plot_deficit_por_barra()
        self.plot_diagnostico_deficit()
        self.plot_rede_carregamento(dia=0, hora=12)
        print(f"✓ Todos os gráficos salvos em {self.output_dir}")


# =============================================================================
# 8. EXEMPLO DE USO — [FIX-6] impressão segura
# =============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from UTILS.SystemLoader import SistemaLoader
    from mercado.PBUC_solver import PBUC_TimeCoupled

    sistema = SistemaLoader("DATA/input/3barras_QUAD.json")
    with open("DATA/input/nordpool_60barras.json", "r") as f:
        mercado = json.load(f)

    modelo = PBUC_TimeCoupled(
        sistema=sistema, json_market=mercado,
        n_horas=24, n_dias=1)

    T = 24
    NZ = len(mercado['ZONAS'])
    fator_carga = np.ones((T, sistema.NBAR))
    fator_vento = (np.ones((T, sistema.NGER_GWD)) * 0.8
                   if sistema.NGER_GWD > 0 else None)
    lambda_zona = np.tile(np.array([3.5, 3.2, 2.8, 3.0][:NZ]), (T, 1))

    modelo.solve_timecoupled(
        solver_name='ipopt', fator_carga=fator_carga,
        fator_vento=fator_vento, lambda_zona_previsto=lambda_zona, tee=False)

    extrator = PBUC_ResultExtractor(modelo, sistema, mercado)
    resultado = extrator.extract()
    analisador = PBUC_Analyzer(resultado)

    print(analisador.resumo_executivo())

    def print_safe(title, df, n=10):
        print(f"\n--- {title} ---")
        if df is None or df.empty:
            print("  (vazio)")
        else:
            print(df.head(n).to_string(index=False))

    print_safe("Balanço de Potência", analisador.analise_balanco_potencia())
    print_safe("Fluxo nas Linhas", analisador.analise_fluxo_linhas())
    print_safe("Déficit por Barra", analisador.analise_deficit_por_barra())
    print_safe("Diagnóstico do Déficit", analisador.diagnostico_deficit())

    cen_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    exporter = PBUC_Exporter(resultado, f"DATA/output/PBUC/{cen_id}")
    exporter.export_csv()
    exporter.export_json()
    exporter.export_sqlite()
    exporter.export_txt()

    db = PBUC_DBHandler('DATA/output/pbuc_resultados.db')
    db.create_tables()
    db.save_pbuc_result(modelo, sistema, mercado, cen_id=cen_id)

    plotter = PBUC_Plotter(db, cen_id,
                           output_dir=f"DATA/output/figuras/{cen_id}")
    plotter.plot_all()