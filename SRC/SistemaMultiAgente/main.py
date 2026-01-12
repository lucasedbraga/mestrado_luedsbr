"""
Sistema MultiAgente Integrado para Análise de Sistemas de Potência
Executa as 3 etapas sequencialmente e gera gráficos comparativos
Caminhos corrigidos para estrutura mestrado_luedsbr/SRC/SistemaMultiAgente/
"""

import os
import sys
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== CONFIGURAÇÃO DE CAMINHOS ====================

# Obter o diretório raiz do projeto (mestrado_luedsbr)
DIRETORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Diretório raiz do projeto: {DIRETORIO_RAIZ}")

# Caminhos absolutos
SISTEMA_JSON = os.path.join(DIRETORIO_RAIZ, "DATA", "input", "3barras_BASE.json")

#SISTEMA_JSON = os.path.join(DIRETORIO_RAIZ, "DATA", "input", "B6L8_BASE.json")
DIRETORIO_DADOS =  DIRETORIO_RAIZ+'/DATA/SMA' # Onde os arquivos .db serão salvos

print(f"Arquivo do sistema: {SISTEMA_JSON}")
print(f"Diretório de dados: {DIRETORIO_DADOS}")

# Verificar se o diretório de dados existe
if not os.path.exists(DIRETORIO_DADOS):
    os.makedirs(DIRETORIO_DADOS)
    print(f"Criado diretório: {DIRETORIO_DADOS}")

# ==================== FUNÇÕES DE CAMINHOS ====================

def caminho_arquivo(nome_arquivo):
    """Retorna o caminho completo para um arquivo no diretório de dados"""
    return os.path.join(DIRETORIO_DADOS, nome_arquivo)

def encontrar_arquivo_mais_recente(prefixo, sufixo=".db"):
    """Encontra o arquivo mais recente com determinado prefixo"""
    arquivos = []
    
    for arquivo in os.listdir(DIRETORIO_DADOS):
        if arquivo.startswith(prefixo) and arquivo.endswith(sufixo):
            caminho_completo = caminho_arquivo(arquivo)
            data_modificacao = os.path.getmtime(caminho_completo)
            arquivos.append((caminho_completo, data_modificacao, arquivo))
    
    if not arquivos:
        return None
    
    # Ordenar por data de modificação (mais recente primeiro)
    arquivos.sort(key=lambda x: x[1], reverse=True)
    return arquivos[0][0]  # Retorna o caminho do arquivo mais recente

# ==================== FUNÇÕES DE EXECUÇÃO ====================

def verificar_arquivo_json():
    """Verifica se o arquivo JSON existe"""
    if not os.path.exists(SISTEMA_JSON):
        print(f"❌ ERRO: Arquivo do sistema não encontrado: {SISTEMA_JSON}")
        print(f"\nVerificando estrutura de diretórios...")
        print(f"Diretório atual: {os.getcwd()}")
        print(f"Diretório do script: {os.path.dirname(__file__)}")
        print(f"Diretório raiz: {DIRETORIO_RAIZ}")
        
        # Verificar se o diretório DATA/input existe
        data_input_dir = os.path.join(DIRETORIO_RAIZ, "DATA", "input")
        if os.path.exists(data_input_dir):
            print(f"\nConteúdo de DATA/input:")
            for item in os.listdir(data_input_dir):
                print(f"  - {item}")
        else:
            print(f"\nDiretório DATA/input não existe!")
            
        return False
    return True

def executar_etapa1():
    """Executa Etapa 1: Programaçao Linear"""
    print("\n" + "="*60)
    print("ETAPA 1: PROGRAMAÇÃO LINEAR (PL)")
    print("="*60)
    
    try:
        # Adicionar diretório atual ao path para importações
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Importar e executar etapa 1
        from etapa1_PL import PlanejamentoTransmissao
        
        print("Inicializando sistema...")
        planejador = PlanejamentoTransmissao(SISTEMA_JSON)
        
        print(f"\nSistema carregado:")
        print(f"  Número de barras: {planejador.sistema_base.NBAR}")
        print(f"  Número de linhas: {planejador.sistema_base.NLIN}")
        print(f"  Número de geradores: {planejador.sistema_base.NGER}")
        print(f"  Carga total inicial: {np.sum(planejador.sistema_base.PLOAD):.3f} pu")
        
        # Executar planejamento apenas para hora 1
        print("\nExecutando planejamento...")
        custo_total, resultados = planejador.executar_planejamento()
        
        if custo_total != float('inf'):
            print(f"\n✅ Etapa 1 concluída com sucesso!")
            print(f"Custo total: ${custo_total:.2f}")
            
            # Encontrar o arquivo gerado
            arquivo_pl = encontrar_arquivo_mais_recente("resultados_PL")
            if arquivo_pl:
                print(f"Arquivo gerado: {os.path.basename(arquivo_pl)}")
                return True, arquivo_pl
            else:
                print("⚠️  Não foi possível encontrar o arquivo de resultados PL")
                return False, None
        else:
            print("\n❌ Falha na Etapa 1")
            return False, None
            
    except Exception as e:
        print(f"\n❌ Erro na Etapa 1: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def executar_etapa2(arquivo_pl):
    """Executa Etapa 2: Fluxo de Potência"""
    print("\n" + "="*60)
    print("ETAPA 2: FLUXO DE POTÊNCIA SIMPLIFICADO (PF)")
    print("="*60)
    
    try:
        if not arquivo_pl or not os.path.exists(arquivo_pl):
            print("❌ Arquivo de resultados PL não encontrado!")
            print(f"Arquivo esperado: {arquivo_pl}")
            return False, None
        
        print(f"Usando resultados do PL: {os.path.basename(arquivo_pl)}")
        
        # Importar e executar etapa 2
        from etapa2_PF import FluxoPotenciaComPLSimplificado
        
        # Criar instância e executar análise
        analisador = FluxoPotenciaComPLSimplificado(SISTEMA_JSON, arquivo_pl)
        resultados = analisador.executar_analise_24h()
        if resultados:
            analisador.plotar_resultados(resultados)
        
        print(f"\n✅ Etapa 2 concluída com sucesso!")
        
        # Encontrar o arquivo gerado
        arquivo_pf = encontrar_arquivo_mais_recente("resultados_PF")
        if arquivo_pf:
            print(f"Arquivo gerado: {os.path.basename(arquivo_pf)}")
            return True, arquivo_pf
        else:
            print("⚠️  Não foi possível encontrar o arquivo de resultados PF")
            return False, None
        
    except Exception as e:
        print(f"\n❌ Erro na Etapa 2: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def executar_etapa3(arquivo_pf):
    """Executa Etapa 3: Fluxo de Potência Ótimo"""
    print("\n" + "="*60)
    print("ETAPA 3: FLUXO DE POTÊNCIA ÓTIMO (OPF)")
    print("="*60)
    
    try:
        if not arquivo_pf or not os.path.exists(arquivo_pf):
            print("❌ Arquivo de resultados PF não encontrado!")
            print(f"Arquivo esperado: {arquivo_pf}")
            return False, None
        
        print(f"Usando resultados do PF: {os.path.basename(arquivo_pf)}")
        
        # Carregar sistema
        with open(SISTEMA_JSON, 'r') as f:
            dados_rede = json.load(f)
        
        # Importar e executar etapa 3
        from etapa3_PNL import OPFNaoLinear
        
        opf_system = OPFNaoLinear(dados_rede)
        
        # Executar apenas para hora 1
        resultados_opf = opf_system.resolver_opf_multiplas_horas(arquivo_pf, [0])
        opf_system.salvar_resultados_opf(resultados_opf, 'DATA/SMA/resultados_OPF.db')
        
        if resultados_opf:
            print(f"\n✅ Etapa 3 concluída com sucesso!")
            
            # Encontrar o arquivo gerado
            arquivo_opf = encontrar_arquivo_mais_recente("resultados_OPF")
            if arquivo_opf:
                print(f"Arquivo gerado: {os.path.basename(arquivo_opf)}")
                return True, arquivo_opf
            else:
                print("⚠️  Não foi possível encontrar o arquivo de resultados OPF")
                return False, None
        else:
            print("\n❌ Falha na Etapa 3")
            return False, None
            
    except Exception as e:
        print(f"\n❌ Erro na Etapa 3: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# ==================== GRÁFICOS COMPARATIVOS ====================

def carregar_resultados_para_graficos(arquivo_pl, arquivo_pf, arquivo_opf):
    """Carrega resultados das 3 etapas para geração de gráficos"""
    dados = {
        'PL': {'V_mag': None, 'V_ang': None, 'fluxos': None, 'geradores':None},
        'PF': {'V_mag': None, 'V_ang': None, 'fluxos': None, 'geradores':None},
        'OPF': {'V_mag': None, 'V_ang': None, 'fluxos': None, 'geradores':None},
    }
    
    # Carregar dados do PL
    if arquivo_pl and os.path.exists(arquivo_pl):
        try:
            conn = sqlite3.connect(arquivo_pl)
            cursor = conn.cursor()
            cursor.execute("SELECT tensoes_mag_json, tensoes_ang_json, fluxos_json, pg_json FROM resultados_PL WHERE hora = 0")
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                dados['PL']['V_mag'] = json.loads(resultado[0])
                dados['PL']['V_ang'] = json.loads(resultado[1])
                dados['PL']['fluxos'] = json.loads(resultado[2])
                dados['PL']['geradores'] = json.loads(resultado[3])
        except Exception as e:
            print(f"Aviso: Não foi possível carregar dados do PL para gráficos: {e}")
    
    # Carregar dados do PF
    if arquivo_pf and os.path.exists(arquivo_pf):
        try:
            conn = sqlite3.connect(arquivo_pf)
            cursor = conn.cursor()
            cursor.execute("SELECT tensoes_mag_json, tensoes_ang_json, fluxos_json, P_gerado_json FROM resultados_fluxo WHERE hora = 0")
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                dados['PF']['V_mag'] = json.loads(resultado[0])
                dados['PF']['V_ang'] = json.loads(resultado[1])
                dados['PF']['fluxos'] = json.loads(resultado[2])
                dados['PF']['geradores'] = json.loads(resultado[3])
        except Exception as e:
            print(f"Aviso: Não foi possível carregar dados do PF para gráficos: {e}")
    
    # Carregar dados do OPF
    if arquivo_opf and os.path.exists(arquivo_opf):
        try:
            conn = sqlite3.connect(arquivo_opf)
            cursor = conn.cursor()
            cursor.execute("SELECT tensoes_mag_json, tensoes_ang_json, fluxos_json, P_gerado_json FROM resultados_OPF WHERE hora = 0")
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                dados['OPF']['V_mag'] = json.loads(resultado[0])
                dados['OPF']['V_ang'] = json.loads(resultado[1])
                dados['OPF']['fluxos'] = json.loads(resultado[2])
                dados['OPF']['geradores'] = json.loads(resultado[3])
        except Exception as e:
            print(f"Aviso: Não foi possível carregar dados do OPF para gráficos: {e}")
    
    return dados

def gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf):
    #TODO: Refazer esse lixo:
    #######################################
    import pandas as pd

    def extrair_fluxos_por_etapa(dados_etapa, tipo_etapa):
        """
        Extrai fluxos das linhas independentemente da estrutura de dados
        
        Args:
            dados_etapa: Dados de fluxos da etapa (PL, PF ou OPF)
            tipo_etapa: 'PL', 'PF' ou 'OPF'
        
        Returns:
            DataFrame com fluxos por linha
        """
        # Mapeamento de linhas do sistema
        linhas_map = {
            0: '1-2',
            1: '1-3', 
            2: '2-3'
        }
        
        fluxos_dict = {}
        
        if tipo_etapa == 'PL':
            # PL: [0.2, 0.5, 0.6] - lista de floats
            if isinstance(dados_etapa, list):
                for idx, valor in enumerate(dados_etapa):
                    linha_id = linhas_map.get(idx, f'linha_{idx}')
                    fluxos_dict[linha_id] = {
                        'P_ij': float(valor),
                        'Q_ij': 0.0,  # PL não tem Q
                        'S_ij': abs(float(valor))
                    }
        
        elif tipo_etapa == 'PF':
            # PF: lista de dicionários
            if isinstance(dados_etapa, list):
                for item in dados_etapa:
                    if isinstance(item, dict):
                        # Criar ID da linha
                        de = item.get('de')
                        para = item.get('para')
                        if de is not None and para is not None:
                            linha_id = f"{de}-{para}"
                            fluxos_dict[linha_id] = {
                                'P_ij': item.get('P_para_de', 0.0),
                                'Q_ij': item.get('Q_de_para', 0.0),
                                'S_ij': abs(item.get('P_para_de', 0.0))  # Simplificado
                            }
        
        elif tipo_etapa == 'OPF':
            # OPF: dicionário {linha_id: {dados}}
            if isinstance(dados_etapa, dict):
                for linha_id, dados in dados_etapa.items():
                    if isinstance(dados, dict):
                        fluxos_dict[linha_id] = {
                            'P_ij': dados.get('P_ij', 0.0),
                            'Q_ij': dados.get('Q_ij', 0.0),
                            'S_ij': dados.get('S_ij', 0.0)
                        }
        
        # Criar DataFrame
        df = pd.DataFrame.from_dict(fluxos_dict, orient='index')
        df.index.name = 'linha'
        
        return df


    def criar_df_comparacao(dados_etapas):
        """
        Cria DataFrame único para comparação entre etapas
        
        Args:
            dados_etapas: Dicionário com dados das etapas
        
        Returns:
            DataFrame consolidado
        """
        dfs = {}
        
        for etapa, dados in dados_etapas.items():
            if dados['fluxos']:
                df_etapa = extrair_fluxos_por_etapa(dados['fluxos'], etapa)
                # Renomear colunas para incluir etapa
                df_etapa.columns = [f"{col}_{etapa}" for col in df_etapa.columns]
                dfs[etapa] = df_etapa
        
        # Juntar todos os DataFrames
        if dfs:
            # Começar com o primeiro DataFrame
            df_final = list(dfs.values())[0]
            
            # Juntar os restantes
            for etapa, df_etapa in list(dfs.items())[1:]:
                df_final = df_final.join(df_etapa, how='outer')
            
            return df_final
        else:
            return pd.DataFrame()
    
    """Gera gráficos comparativos simplificados"""
    print("\n" + "="*60)
    print("GERANDO GRÁFICOS COMPARATIVOS")
    print("="*60)
    
    # Carregar dados
    dados = carregar_resultados_para_graficos(arquivo_pl, arquivo_pf, arquivo_opf)
    
    # Verificar se temos dados suficientes
    etapas_com_dados = []
    for etapa in ['PL', 'PF', 'OPF']:
        if dados[etapa]['V_mag'] is not None:
            etapas_com_dados.append(etapa)
    
    if not etapas_com_dados:
        print("❌ Não há dados suficientes para gerar gráficos!")
        return
    
    print(f"Gerando gráficos com dados de: {', '.join(etapas_com_dados)}")
    
    # Criar gráfico único de comparação
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('COMPARAÇÃO ENTRE ETAPAS - Sistema 3 Barras', fontsize=16, fontweight='bold')
    
    # 1. Gráfico de Tensões (Magnitude)
    ax1 = axes[0, 0]
    cores = {'PL': 'blue',
            'PF': 'green',
            'OPF': 'orange'}
    
    barras = [0, 1, 2]  # Índices das barras
    width = 0.25
    
    for idx, etapa in enumerate(etapas_com_dados):
        if dados[etapa]['V_mag']:
            ax1.bar([b + idx*width for b in barras], 
                   dados[etapa]['V_mag'], 
                   width=width, 
                   label=etapa, 
                   alpha=0.7,
                   color=cores.get(etapa, 'gray'))
    
    ax1.set_xlabel('Barra')
    ax1.set_ylabel('Tensão (pu)')
    ax1.set_title('Magnitude de Tensão')
    ax1.set_xticks([b + width for b in barras])
    ax1.set_xticklabels(['Barra 1', 'Barra 2', 'Barra 3'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Limites (0.95-1.05 pu)')
    ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.5)
    
    # 2. Gráfico de Ângulos
    ax2 = axes[0, 1]
    
    for idx, etapa in enumerate(etapas_com_dados):
        if dados[etapa]['V_ang']:
            # Converter para graus
            angulos = [abs(np.degrees(ang)) for ang in dados[etapa]['V_ang']]
            ax2.plot(barras, angulos, 'o-', label=etapa, 
                    linewidth=2, markersize=8, color=cores.get(etapa, 'gray'))
    
    ax2.set_xlabel('Barra')
    ax2.set_ylabel('Ângulo (graus)')
    ax2.set_title('Ângulo de Tensão')
    ax2.set_xticks(barras)
    ax2.set_xticklabels(['Barra 1', 'Barra 2', 'Barra 3'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Gráfico de Fluxos nas Linhas
    ax3 = axes[1, 0]

    # Linhas do sistema e seus limites (hardcoded)
    linhas = ['1-2', '1-3', '2-3']
    limites = {
        '1-2': 0.2,   # Limite da linha 1-2
        '1-3': 1.0,   # Limite da linha 1-3
        '2-3': 0.6    # Limite da linha 2-3
    }

    # Criar DataFrame comparativo
    df_comparacao = criar_df_comparacao(dados)

    for idx, etapa in enumerate(etapas_com_dados):
        if etapa in df_comparacao.columns.str.split('_').str[-1].unique():
            # Extrair fluxos P para esta etapa
            coluna_p = f'P_ij_{etapa}'
            if coluna_p in df_comparacao.columns:
                fluxos_valores = []
                for linha in linhas:
                    if linha in df_comparacao.index:
                        fluxo = df_comparacao.loc[linha, coluna_p]
                        fluxos_valores.append(float(fluxo) if pd.notna(fluxo) else 0.0)
                    else:
                        fluxos_valores.append(0.0)
            else:
                # Se não encontrou a coluna, usa zeros
                fluxos_valores = [0.0] * len(linhas)
            
            # Plotar barras
            ax3.bar([i + idx*width for i in range(len(linhas))], 
                fluxos_valores, 
                width=width, 
                label=etapa,
                alpha=0.7,
                color=cores.get(etapa, 'gray'))

    # Adicionar linhas de limite como retas pontilhadas
    for i, linha in enumerate(linhas):
        limite_positivo = limites[linha]
        limite_negativo = -limites[linha]
        
        # Calcular a posição horizontal média das barras para esta linha
        posicao_media = i + width * (len(etapas_com_dados) - 1) / 2
        
        # Linha de limite superior (positivo)
        ax3.axhline(y=limite_positivo, xmin=0, xmax=1, 
                    color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # # Linha de limite inferior (negativo)
        # ax3.axhline(y=limite_negativo, xmin=0, xmax=1, 
        #             color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Adicionar texto com o valor do limite
        ax3.text(posicao_media + 0.1, limite_positivo + 0.02, 
                f'{limite_positivo} pu', fontsize=9, color='red', ha='center')
        # ax3.text(posicao_media + 0.1, limite_negativo - 0.04, 
        #         f'{limite_negativo} pu', fontsize=9, color='red', ha='center')

    ax3.set_xlabel('Linha')
    ax3.set_ylabel('Fluxo de Potência (pu)')
    ax3.set_title('Fluxo Ativo nas Linhas com Limites')
    ax3.set_xticks([i + width for i in range(len(linhas))])
    ax3.set_xticklabels(linhas)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Ajustar limites do eixo Y para melhor visualização dos limites
    limite_max = max(limites.values())
    ax3.set_ylim(0, limite_max * 1.3)

    # # Adicionar anotação explicativa
    # ax3.text(0.02, 0.98, 'Linhas vermelhas tracejadas: Limites de fluxo', 
    #         transform=ax3.transAxes, fontsize=10, color='red',
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Gráfico comparativo de geração por barra
    ax4 = axes[1, 1]
    df = pd.DataFrame(dados)
        # Função para calcular perdas
   

    # Lista de geradores (barras)
    geradores = [f"Gerador {i+1}" for i in range(3)]  # Assumindo 3 geradores

    width = 0.35  # Largura das barras
    cores = {'PL': 'blue',
        'PF': 'green',
        'OPF': 'orange'}

    # Preparar dados para cada etapa
    dados_geradores = []

    for idx, etapa in enumerate(etapas_com_dados):
        if etapa in df.columns:
            # Extrair valores dos geradores para esta etapa
            geradores_valores = df.loc['geradores', etapa]
            
            # Se for numpy array ou lista, converter para lista
            if hasattr(geradores_valores, 'tolist'):
                geradores_valores = geradores_valores.tolist()
            
            # Plotar barras para cada gerador
            posicoes = [i + idx * width for i in range(len(geradores))]
            
            # Verificar se temos valores suficientes
            if len(geradores_valores) >= len(geradores):
                ax4.bar(posicoes, 
                    geradores_valores[:len(geradores)], 
                    width=width, 
                    label=etapa,
                    alpha=0.7,
                    color=cores.get(etapa, 'gray'))
            else:
                # Se não tiver valores suficientes, completar com zeros
                valores_completos = geradores_valores + [0] * (len(geradores) - len(geradores_valores))
                ax4.bar(posicoes, 
                    valores_completos, 
                    width=width, 
                    label=etapa,
                    alpha=0.7,
                    color=cores.get(etapa, 'gray'))

    ax4.set_xlabel('Gerador')
    ax4.set_ylabel('Geração (pu)')
    ax4.set_title('Geração Ativa por Barra Geradora')
    ax4.set_xticks([i + width/2 for i in range(len(geradores))])
    ax4.set_xticklabels(geradores)
    ax4.legend(title='Etapa')
    ax4.grid(True, alpha=0.3, axis='y')

    # Adicionar valores nas barras
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.3f', padding=3, fontsize=8)

    # Adicionar linha horizontal para mostrar limites (opcional)
    ax4.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    # # Preparar dados para tabela
    # tabela_dados = []
    # cabecalhos = ['Etapa', 'V_medio', 'θ_medio', 'Perdas (pu)']

    # for etapa in etapas_com_dados:
    #     if dados[etapa]['V_mag'] and dados[etapa]['V_ang']:
    #         v_medio = np.mean(dados[etapa]['V_mag'])
    #         theta_medio = np.mean([np.degrees(ang) for ang in dados[etapa]['V_ang']])
            
    #         # Calcular perdas usando a nova função
    #         perdas = calcular_perdas_aproximadas(dados[etapa]['fluxos'], etapa)
            
    #         tabela_dados.append([etapa, f"{v_medio:.4f}", f"{theta_medio:.2f}°", f"{perdas:.4f}"])

    # if tabela_dados:
    #     # Criar tabela
    #     tabela = ax4.table(cellText=tabela_dados, 
    #                     colLabels=cabecalhos, 
    #                     cellLoc='center', 
    #                     loc='center',
    #                     colWidths=[0.2, 0.25, 0.25, 0.3])
        
    #     tabela.auto_set_font_size(False)
    #     tabela.set_fontsize(10)
    #     tabela.scale(1.2, 1.5)
        
    #     # Colorir cabeçalho
    #     for i in range(len(cabecalhos)):
    #         tabela[(0, i)].set_facecolor('#40466e')
    #         tabela[(0, i)].set_text_props(weight='bold', color='white')
        
    plt.tight_layout()
        
    # Salvar figura no diretório de dados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figura_path = os.path.join(DIRETORIO_DADOS, f'comparacao_etapas_{timestamp}.png')
    plt.savefig(figura_path, dpi=300, bbox_inches='tight')

    print(f"✅ Gráficos salvos em: {figura_path}")
    plt.show()
    
    # Imprimir tabela no console também
#     if tabela_dados:
#         print("\n" + "="*60)
#         print("RESUMO COMPARATIVO - Hora 1")
#         print("="*60)
#         print(f"{'Etapa':<10} {'V_medio':<12} {'θ_medio':<12} {'Perdas (pu)':<12}")
#         print("-"*48)
        
#         for linha in tabela_dados:
#             print(f"{linha[0]:<10} {linha[1]:<12} {linha[2]:<12} {linha[3]:<12}")

# # ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal que orquestra todas as etapas"""
    print("\n" + "="*60)
    print("SISTEMA MULTIAGENTE - ANÁLISE DE SISTEMAS DE POTÊNCIA")
    print("="*60)
    print(f"Diretório raiz: {DIRETORIO_RAIZ}")
    print(f"Diretório de trabalho atual: {os.getcwd()}")
    print("="*60)
    
    # Verificar arquivo de entrada
    if not verificar_arquivo_json():
        return
    
    print(f"\n📁 Sistema elétrico: {SISTEMA_JSON}")
    print("🔍 Analisando apenas a hora 1 para simplificação")
    print("="*60)
    
    # Arquivos gerados
    arquivo_pl = None
    arquivo_pf = None
    arquivo_opf = None
    
    # Executar as 3 etapas sequencialmente
    try:
        print("\n🔧 Iniciando execução das etapas...")
        
        # Etapa 1: Programaçao Linear
        sucesso1, arquivo_pl = executar_etapa1()
        
        # Etapa 2: Fluxo de Potência
        if sucesso1 and arquivo_pl:
            sucesso2, arquivo_pf = executar_etapa2(arquivo_pl)
        else:
            sucesso2 = False
            print("\n⚠️  Pulando Etapa 2 devido a falha na Etapa 1")
        
        # Etapa 3: Fluxo de Potência Ótimo
        if sucesso2 and arquivo_pf:
            sucesso3, arquivo_opf = executar_etapa3(arquivo_pf)
        else:
            sucesso3 = False
            print("\n⚠️  Pulando Etapa 3 devido a falha na Etapa 2")
        
        # Gerar gráficos se pelo menos uma etapa foi bem sucedida
        arquivos_existentes = [a for a in [arquivo_pl, arquivo_pf, arquivo_opf] if a is not None]
        
        if arquivos_existentes:
            print("\n📊 Gerando gráficos comparativos...")
            gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf)
        else:
            print("\n❌ Nenhum arquivo de resultados foi gerado.")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO DA EXECUÇÃO")
    print("="*60)
    
    # Listar arquivos gerados
    print("\n📁 Arquivos gerados no diretório de dados:")
    arquivos_gerados = []
    
    for pattern in ['resultados_PL_*.db', 'resultados_PF_*.db', 'resultados_OPF_*.db', 'comparacao_etapas_*.png']:
        for arquivo in os.listdir(DIRETORIO_DADOS):
            if arquivo.startswith(pattern.split('*')[0]) and arquivo not in arquivos_gerados:
                arquivos_gerados.append(arquivo)
    
    if arquivos_gerados:
        for arquivo in sorted(arquivos_gerados):
            caminho_completo = os.path.join(DIRETORIO_DADOS, arquivo)
            size_kb = os.path.getsize(caminho_completo) / 1024 if os.path.exists(caminho_completo) else 0
            print(f"  📄 {arquivo} - {size_kb:.1f} KB")
    else:
        print("  Nenhum arquivo gerado.")
    
    print("\n" + "="*60)
    print("EXECUÇÃO CONCLUÍDA")
    print("="*60)
    
    # Sugestão para próxima execução
    print("\n💡 DICA: Para executar novamente, você pode:")
    print("  1. Mover/renomear os arquivos .db existentes")
    print("  2. Executar: python SRC/SistemaMultiAgente/main.py")
    print("  3. Os resultados serão salvos em:", DIRETORIO_DADOS)

# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    # Mudar para o diretório raiz para garantir que os arquivos sejam salvos no lugar certo
    os.chdir(DIRETORIO_RAIZ)
    print(f"Mudando para diretório: {DIRETORIO_RAIZ}")
    
    main()