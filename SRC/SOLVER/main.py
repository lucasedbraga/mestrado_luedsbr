#!/usr/bin/env python3
"""
Script principal para resolver um único snapshot de OPF
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Adiciona o diretório SRC ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.models.systemLoader import SistemaLoader
from SOLVER.FOB.economic_dispatch import OPF_Solver
from SOLVER.DB.OPF_DBhandler import OPF_DBHandler
from SOLVER.models.DC_OPF_MultiDay_Model import MultiDayOPFModel

def main():
    print("=" * 70)
    print("SISTEMA DE OTIMIZAÇÃO DE FLUXO DE POTÊNCIA (OPF) - MULTI-DIA")
    print("=" * 70)
    try:
        # 1. CARREGAR SISTEMA
        print("\n1. Carregando dados do sistema...")
        json_path = "DATA/input/B6L8_BASE.json"
        
        if not os.path.exists(json_path):
            print(f"ERRO: Arquivo não encontrado: {json_path}")
            print("Por favor, verifique se o arquivo existe em DATA/input/")
            return
        
        sistema_loader = SistemaLoader(json_path)
        sistema = sistema_loader
        
        print(f"   ✓ Sistema carregado: {json_path}")
        print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
        print(f"   ✓ Barras: {sistema.NBAR}")
        print(f"   ✓ Linhas: {sistema.NLIN}")
        print(f"   ✓ Geradores: {sistema.NGER_ORIGINAL} originais + {sistema.NGER_CURTAILMENT} curtailment + {sistema.NGER_DEFICIT} déficit")
        print(f"   ✓ Geradores eólicos (GWD): {len(sistema.BAR_GWD)}")
        print(f"   ✓ Carga total: {np.sum(sistema.PLOAD):.3f} pu ({np.sum(sistema.PLOAD) * sistema.SB:.1f} MW)")
        
        # 2. CONFIGURAR PARÂMETROS DO SNAPSHOT
        print("\n2. Configurando snapshot...")
        
        # Definir fator de vento (0-1)
        filepath = r"/home/lucasedbraga/repositorios/ufjf/mestrado_luedsbr/SRC/SOLVER/DB/getters/intermittent-renewables-production-france.csv"

        # Carregar dados brutos
        print(f"Carregando: {filepath}")
        df = pd.read_csv(filepath)
        # Processar data/hora
        df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6))
        df = df.sort_values('DateTime')

        # Separar vento e solar
        df_wind = df[df['Source'] == 'Wind'].copy()
        # Normalizar produção (fator 0-1)
        for df_source in [df_wind]:
            if len(df_source) > 0:
                max_prod = df_source['Production'].max()
                if max_prod > 0:
                    df_source['Factor'] = df_source['Production'] / max_prod
                else:
                    df_source['Factor'] = 0

        # Selecionar colunas finais
        cols = ['DateTime', 'Production', 'Factor']
        df_wind = df_wind[cols].copy()
        # Renomear
        df_wind.columns = ['timestamp', 'wind_production_mw', 'wind_factor']

        
        # Simulação para 2 dias
        n_dias = 2
        n_horas = 24

        multi_model = MultiDayOPFModel(sistema, n_horas=n_horas, n_dias=n_dias)
        print(f"\nIniciando simulação multi-dia ({n_dias} dias, {n_horas} horas por dia)...")
        # SOC inicial da(s) bateria(s) (exemplo: 50%)
        SOC_inicial_bateria = 0.5*sistema.BATTERY_CAPACITY  # ou ajuste conforme necessário
        fator_vento = np.array([df_wind["wind_factor"].sample(n_horas),
                                df_wind["wind_factor"].sample(n_horas)])
        print(f"   ✓ Fator de vento: {fator_vento}")
        # Parâmetros para sorteio de demanda
        fator_carga = np.random.normal(1, 0.3, size=(n_dias,n_horas))
        print(f"   ✓ Fator de carga: {fator_carga}")
        opf_result = multi_model.solve_multiday_sequencial(
            solver_name='glpk',
            fator_carga=fator_carga,
            soc_inicial=SOC_inicial_bateria,
            fator_vento=fator_vento
        )
        print("\n✓ Simulação multi-dia concluída!")
        # Exportar resultados para o banco de dados
        db_handler = OPF_DBHandler('DATA/SMA/resultados_PL.db')
        db_handler.create_tables()
        for snapshot in opf_result.snapshots:
            db_handler.save_hourly_result(
                resultado=snapshot,
                sistema=sistema,
                hora=snapshot.hora,
                perfil_carga=fator_carga[snapshot.dia, snapshot.hora],
                perfil_eolica=fator_vento[snapshot.dia, snapshot.hora],
                solver_name='glpk',
                dia=str(snapshot.dia+1)
            )
            print(f"Dia {snapshot.dia+1}, Hora {snapshot.hora:02d}:00, Sucesso: {snapshot.sucesso}")
        print("\nEXECUÇÃO FINALIZADA")
    except Exception as e:
        print(f"\n✗ ERRO durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())