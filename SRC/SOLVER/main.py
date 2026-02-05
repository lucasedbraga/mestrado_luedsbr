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

def main():
    """Executa um único snapshot de OPF"""
    print("=" * 70)
    print("SISTEMA DE OTIMIZAÇÃO DE FLUXO DE POTÊNCIA (OPF) - SNAPSHOT")
    print("=" * 70)
    
    try:
        # 1. CARREGAR SISTEMA
        print("\n1. Carregando dados do sistema...")
        json_path = "DATA/input/3barras_TESTE.json"
        
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
        filepath = r"C:\Users\lucas\repositorios\mestrado_luedsbr\SRC\SOLVER\DB\getters\intermittent-renewables-production-france.csv"

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
        fator_vento = float(df_wind["wind_factor"].sample(1))
        print(f"   ✓ Fator de vento: {fator_vento:.1%}")
        
        # Atualizar capacidade eólica com fator de vento
        sistema.atualizar_perfil_eolico(fator_vento)
        
        capacidade_eolica = sum(sistema.PGMAX_EFETIVO[g] for g in sistema.BAR_GWD)
        print(f"   ✓ Capacidade eólica disponível: {capacidade_eolica:.3f} pu ({capacidade_eolica * sistema.SB:.1f} MW)")
        
        # 3. INSTANCIAR SOLVER
        print("\n3. Instanciando solver...")
        solver = OPF_Solver(sistema)
        print("   ✓ Solver instanciado")
        
        # 4. RESOLVER OPF
        print("\n4. Resolvendo OPF...")
        resultado = solver.solve_with_losses(
            max_iter=20,
            tol=1e-5,
            solver_name='glpk',  # ou 'cbc', 'ipopt'
            verbose=True
        )
        
        # 5. ANALISAR RESULTADOS
        print("\n" + "=" * 70)
        print("RESULTADOS DO OPF")
        print("=" * 70)
        
        if resultado.sucesso:
            print(f"✓ ÓTIMO ENCONTRADO!")
            print(f"\nCusto total: ${resultado.custo_total:.2f}")
            print(f"Iterações: {resultado.iteracoes}")
            print(f"Tempo de execução: {resultado.tempo_execucao:.2f} segundos")
            
            # Cálculos em MW
            deficit_total_mw = np.sum(resultado.DEFICIT)
            curtailment_total_mw = np.sum(resultado.CURTAILMENT)
            perdas_total_mw = resultado.perdas
            
            print(f"\nMétricas do sistema:")
            print(f"  Déficit total: {deficit_total_mw:.2f} MW")
            print(f"  Curtailment eólico: {curtailment_total_mw:.2f} MW")
            print(f"  Perdas nas linhas: {perdas_total_mw:.2f} MW")
            
            # Análise de geração
            print(f"\nGeração por tipo:")
            
            # Geração convencional
            geracao_conv = 0
            for g_idx in range(sistema.NGER_ORIGINAL):
                if sistema.GER_TIPOS[g_idx] != "GWD":
                    geracao_conv += resultado.PG[g_idx] * sistema.SB
            
            # Geração eólica utilizada
            geracao_eolica = 0
            for g_idx in sistema.BAR_GWD:
                if g_idx < len(resultado.PG):
                    geracao_eolica += resultado.PG[g_idx] * sistema.SB
            
            print(f"  Geração convencional: {geracao_conv:.2f} MW")
            print(f"  Geração eólica utilizada: {geracao_eolica:.2f} MW")
            print(f"  Geração total: {(geracao_conv + geracao_eolica):.2f} MW")

             # Verificar se há baterias
            if hasattr(sistema, 'BARRAS_COM_BATERIA') and len(sistema.BARRAS_COM_BATERIA) > 0:
                print(f"   ✓ Baterias: {len(sistema.BARRAS_COM_BATERIA)} bateria(s) encontrada(s)")
                capacidade_total = sum(sistema.BATTERY_CAPACITY) * sistema.SB
                potencia_total = sum(sistema.BATTERY_POWER_LIMIT) * sistema.SB
                print(f"   ✓ Capacidade total de armazenamento: {capacidade_total:.1f} MWh")
                print(f"   ✓ Potência máxima de carga: {potencia_total:.1f} MW")
            else:
                print(f"   ⚠️  Nenhuma bateria configurada no sistema")
            
            # Análise de fluxos
            print(f"\nAnálise de fluxos nas linhas:")
            
            linhas_congestionadas = []
            for e_idx in range(sistema.NLIN):
                fluxo_mw = resultado.FLUXO[e_idx] * sistema.SB
                limite_mw = sistema.FLIM[e_idx] * sistema.SB
                carregamento = abs(fluxo_mw / limite_mw * 100) if limite_mw > 0 else 0
                
                if carregamento > 90:  # Linha com carregamento > 90%
                    de_barra = sistema.indice_para_barra[sistema.line_fr[e_idx]]
                    para_barra = sistema.indice_para_barra[sistema.line_to[e_idx]]
                    linhas_congestionadas.append({
                        'linha': e_idx,
                        'de': de_barra,
                        'para': para_barra,
                        'fluxo': fluxo_mw,
                        'limite': limite_mw,
                        'carregamento': carregamento
                    })
            
            if linhas_congestionadas:
                print(f"  ⚠️  {len(linhas_congestionadas)} linha(s) com carregamento > 90%:")
                for linha in linhas_congestionadas:
                    print(f"    Linha {linha['linha']} ({linha['de']}→{linha['para']}): "
                          f"{linha['fluxo']:.1f}/{linha['limite']:.1f} MW "
                          f"({linha['carregamento']:.1f}%)")
            else:
                print(f"  ✓ Nenhuma linha com carregamento crítico")
            
            # 6. SALVAR NO BANCO DE DADOS
            print(f"\n5. Salvando resultados no banco de dados...")
            
            db_handler = OPF_DBHandler('DATA/SMA/resultados_opf.db')
            db_handler.create_tables()
            
            # Salvar resultado (hora 0 para snapshot único)
            resultado_id = db_handler.save_hourly_result(
                resultado, sistema, 
                hora=0,  # Snapshot único
                perfil_carga=1.0,  # Fator de carga = 100%
                perfil_eolica=fator_vento,
                solver_name='glpk'
            )
            
            print(f"   ✓ Resultados salvos com ID: {resultado_id}")
            print(f"   ✓ Banco: DATA/SMA/resultados_opf.db")
            
            # Exportar para CSV
            db_handler.export_to_csv('DATA/SMA/export')
            print(f"   ✓ Dados exportados para CSV")
            
        else:
            print(f"✗ OPF NÃO CONVERGIU")
            print(f"  Mensagem: {resultado.mensagem if hasattr(resultado, 'mensagem') else 'Desconhecido'}")
        
        print("\n" + "=" * 70)
        print("EXECUÇÃO FINALIZADA")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERRO durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())