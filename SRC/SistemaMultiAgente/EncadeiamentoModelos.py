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
SISTEMA_JSON = os.path.join(DIRETORIO_RAIZ, "DATA", "input", "B6L8_BASE.json")

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
        # if resultados:
        #     analisador.plotar_resultados(resultados)
        
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
        horas = list(np.arange(24))
        
        resultados_opf = opf_system.resolver_opf_multiplas_horas(arquivo_pf)
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

def gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf, hora_escolha='aleatoria'):
    """Gera gráficos comparativos simplificados para qualquer tamanho de sistema"""
    print("\n" + "="*60)
    print("GERANDO GRÁFICOS COMPARATIVOS")
    print("="*60)
    
    # Determinar qual hora usar
    horas_disponiveis = []
    
    # Verificar quais horas estão disponíveis nos arquivos
    for arquivo, etapa in [(arquivo_pl, 'PL'), (arquivo_pf, 'PF'), (arquivo_opf, 'OPF')]:
        if arquivo and os.path.exists(arquivo):
            try:
                conn = sqlite3.connect(arquivo)
                cursor = conn.cursor()
                if etapa == 'PL':
                    cursor.execute("SELECT DISTINCT hora FROM resultados_PL ORDER BY hora")
                elif etapa == 'PF':
                    cursor.execute("SELECT DISTINCT hora FROM resultados_fluxo ORDER BY hora")
                elif etapa == 'OPF':
                    cursor.execute("SELECT DISTINCT hora FROM resultados_OPF ORDER BY hora")
                
                horas_etapa = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                if horas_etapa:
                    horas_disponiveis.extend(horas_etapa)
                    print(f"Horas disponíveis em {etapa}: {horas_etapa}")
            except Exception as e:
                print(f"Aviso: Não foi possível verificar horas em {etapa}: {e}")
    
    # Remover duplicatas e ordenar
    horas_disponiveis = sorted(set(horas_disponiveis))
    
    if not horas_disponiveis:
        print("❌ Não há horas disponíveis para gerar gráficos!")
        return
    
    # Escolher a hora baseada no parâmetro
    if hora_escolha == 'primeira':
        hora_selecionada = horas_disponiveis[0]
    elif hora_escolha == 'ultima':
        hora_selecionada = horas_disponiveis[-1]
    elif hora_escolha == 'aleatoria':
        hora_selecionada = np.random.choice(horas_disponiveis)
    elif isinstance(hora_escolha, int):
        if hora_escolha in horas_disponiveis:
            hora_selecionada = hora_escolha
        else:
            print(f"⚠️  Hora {hora_escolha} não disponível. Usando hora mais próxima.")
            # Encontrar a hora mais próxima
            hora_selecionada = min(horas_disponiveis, key=lambda x: abs(x - hora_escolha))
    else:
        print("⚠️  Opção de hora inválida. Usando hora aleatória.")
        hora_selecionada = np.random.choice(horas_disponiveis)
    
    print(f"\n📊 Gerando gráficos para hora: {hora_selecionada}")
    
    # Carregar dados para a hora selecionada
    dados = carregar_resultados_para_graficos(arquivo_pl, arquivo_pf, arquivo_opf, hora_selecionada)
    
    # Verificar se temos dados suficientes
    etapas_com_dados = []
    for etapa in ['PL', 'PF', 'OPF']:
        if dados[etapa]['V_mag'] is not None:
            etapas_com_dados.append(etapa)
    
    if not etapas_com_dados:
        print("❌ Não há dados suficientes para gerar gráficos!")
        return
    
    print(f"Gerando gráficos com dados de: {', '.join(etapas_com_dados)}")
    
    # Carregar dados do sistema para informações estruturais
    try:
        with open(SISTEMA_JSON, 'r') as f:
            sistema = json.load(f)
        
        n_barras = len(sistema['BARRAS'])
        n_linhas = len(sistema['LINHAS'])
        n_geradores = len(sistema['GERADORES'])
        
        print(f"\n📋 Informações do sistema:")
        print(f"  • Barras: {n_barras}")
        print(f"  • Linhas: {n_linhas}")
        print(f"  • Geradores: {n_geradores}")
        
        # Obter lista de linhas (origem-destino)
        linhas = []
        limites_linhas = {}
        for linha in sistema['LINHAS']:
            linha_id = f"{linha['ID_Barra_Origem']}-{linha['ID_Barra_Destino']}"
            linhas.append(linha_id)
            # Converter limite para pu se necessário
            limite = linha.get('LIM_Fluxo', 0.0)
            if linha.get('LIM_Fluxo_Unidade', 'MW') == 'MW':
                S_base = sistema.get('S_base', 100.0)
                limite_pu = limite / S_base
            else:
                limite_pu = limite
            limites_linhas[linha_id] = limite_pu
        
        # Obter lista de geradores
        geradores = []
        for gerador in sistema['GERADORES']:
            gerador_id = f"Gerador {gerador['ID_Gerador']}"
            geradores.append(gerador_id)
        
    except Exception as e:
        print(f"⚠️  Não foi possível carregar informações do sistema: {e}")
        # Usar valores padrão baseados nos dados disponíveis
        if dados[etapas_com_dados[0]]['V_mag']:
            n_barras = len(dados[etapas_com_dados[0]]['V_mag'])
        else:
            n_barras = 0
        
        # Criar listas genéricas
        linhas = []
        limites_linhas = {}
        geradores = []
    
    # Criar gráfico único de comparação
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'COMPARAÇÃO ENTRE ETAPAS - Hora {hora_selecionada}', fontsize=16, fontweight='bold')
    
    # Cores para cada etapa
    cores = {'PL': '#1f77b4',  # Azul
            'PF': '#2ca02c',  # Verde
            'OPF': '#ff7f0e'} # Laranja
    
    # 1. Gráfico de Tensões (Magnitude)
    ax1 = axes[0, 0]
    
    if n_barras > 0:
        barras_indices = list(range(n_barras))
        width = 0.25 if len(etapas_com_dados) > 1 else 0.6
        
        for idx, etapa in enumerate(etapas_com_dados):
            if dados[etapa]['V_mag'] and len(dados[etapa]['V_mag']) >= n_barras:
                # Ajustar posição das barras
                posicoes = [b + idx*width - (len(etapas_com_dados)-1)*width/2 for b in barras_indices]
                
                ax1.bar(posicoes, 
                       dados[etapa]['V_mag'][:n_barras], 
                       width=width, 
                       label=etapa, 
                       alpha=0.7,
                       color=cores.get(etapa, 'gray'))
        
        ax1.set_xlabel('Barra')
        ax1.set_ylabel('Tensão (pu)')
        ax1.set_title('Magnitude de Tensão por Barra')
        ax1.set_xticks(barras_indices)
        ax1.set_xticklabels([f'Barra {i+1}' for i in barras_indices], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Adicionar limites de tensão
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Limites (0.95-1.05 pu)')
        ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.5)
        ax1.set_ylim(0.9, 1.1)
    else:
        ax1.text(0.5, 0.5, 'Sem dados de tensão disponíveis', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Magnitude de Tensão (Sem dados)')
    
    # 2. Gráfico de Ângulos de Tensão
    ax2 = axes[0, 1]
    
    if n_barras > 0:
        barras_indices = list(range(n_barras))
        
        for idx, etapa in enumerate(etapas_com_dados):
            if dados[etapa]['V_ang'] and len(dados[etapa]['V_ang']) >= n_barras:
                # Converter para graus
                angulos = [np.degrees(ang) for ang in dados[etapa]['V_ang'][:n_barras]]
                
                # Plotar com marcadores diferentes para cada etapa
                marcadores = ['o', 's', '^', 'D', 'v']
                marcador = marcadores[idx % len(marcadores)]
                
                ax2.plot(barras_indices, angulos, marker=marcador, label=etapa, 
                        linewidth=2, markersize=8, color=cores.get(etapa, 'gray'))
        
        ax2.set_xlabel('Barra')
        ax2.set_ylabel('Ângulo (graus)')
        ax2.set_title('Ângulo de Tensão por Barra')
        ax2.set_xticks(barras_indices)
        ax2.set_xticklabels([f'Barra {i+1}' for i in barras_indices], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Sem dados de ângulo disponíveis', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Ângulo de Tensão (Sem dados)')
    
    # 3. Gráfico de Fluxos nas Linhas
    ax3 = axes[1, 0]
    
    if len(linhas) > 0:
        # Função para extrair fluxos de uma etapa
        def extrair_fluxos_etapa(dados_fluxos, etapa):
            fluxos = {}
            
            if etapa == 'PL':
                # PL: lista simples de valores
                if isinstance(dados_fluxos, list):
                    for i, valor in enumerate(dados_fluxos):
                        if i < len(linhas):
                            linha_id = linhas[i]
                            fluxos[linha_id] = float(valor)
            
            elif etapa == 'PF':
                # PF: lista de dicionários
                if isinstance(dados_fluxos, list):
                    for item in dados_fluxos:
                        if isinstance(item, dict):
                            de = item.get('de')
                            para = item.get('para')
                            if de is not None and para is not None:
                                linha_id = f"{de}-{para}"
                                if linha_id in linhas:
                                    fluxos[linha_id] = abs(item.get('P_para_de', 0.0))
            
            elif etapa == 'OPF':
                # OPF: dicionário aninhado
                if isinstance(dados_fluxos, dict):
                    for linha_id, fluxo_data in dados_fluxos.items():
                        if isinstance(fluxo_data, dict):
                            fluxos[linha_id] = abs(fluxo_data.get('P_ij', 0.0))
            
            return fluxos
        
        # Coletar fluxos de todas as etapas
        fluxos_por_etapa = {}
        for etapa in etapas_com_dados:
            if dados[etapa]['fluxos']:
                fluxos_por_etapa[etapa] = extrair_fluxos_etapa(dados[etapa]['fluxos'], etapa)
        
        # Configurar posições das barras
        n_etapas = len(fluxos_por_etapa)
        n_linhas_plot = len(linhas)
        
        if n_etapas > 0 and n_linhas_plot > 0:
            width = 0.8 / n_etapas  # Largura ajustável
            espacamento = 0.1
            
            for idx, (etapa, fluxos) in enumerate(fluxos_por_etapa.items()):
                valores = []
                for linha_id in linhas:
                    valor = fluxos.get(linha_id, 0.0)
                    valores.append(valor)
                
                # Posições das barras
                posicoes = [i + idx*width - (n_etapas-1)*width/2 for i in range(n_linhas_plot)]
                
                ax3.bar(posicoes, valores, width=width, label=etapa,
                       alpha=0.7, color=cores.get(etapa, 'gray'))
            
            # Adicionar limites de fluxo
            for i, linha_id in enumerate(linhas):
                limite = limites_linhas.get(linha_id, 0.0)
                if limite > 0:
                    # Linha horizontal para limite
                    ax3.axhline(y=limite, xmin=0, xmax=1, color='red', 
                               linestyle='--', linewidth=1, alpha=0.7)
                    # Texto com o limite
                    ax3.text(i, limite + limite*0.05, f'{limite:.2f} pu', 
                            fontsize=8, color='red', ha='center')
            
            ax3.set_xlabel('Linha')
            ax3.set_ylabel('Fluxo de Potência (pu)')
            ax3.set_title('Fluxo Ativo nas Linhas')
            ax3.set_xticks(range(n_linhas_plot))
            ax3.set_xticklabels(linhas, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Ajustar limites do eixo Y
            if limites_linhas:
                limite_max = max(limites_linhas.values())
                ax3.set_ylim(0, limite_max * 1.3)
        else:
            ax3.text(0.5, 0.5, 'Sem dados de fluxo disponíveis', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Fluxos nas Linhas (Sem dados)')
    else:
        ax3.text(0.5, 0.5, 'Sem informações de linhas', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Fluxos nas Linhas (Sem dados)')
    # 4. Gráfico de Geração por Gerador
    ax4 = axes[1, 1]

    print("\n🔍 DEBUG: Estrutura dos dados de geração:")
    for etapa in etapas_com_dados:
        if dados[etapa]['geradores'] is not None:
            print(f"{etapa}: Tipo = {type(dados[etapa]['geradores'])}, Tamanho = {len(dados[etapa]['geradores']) if hasattr(dados[etapa]['geradores'], '__len__') else 'N/A'}")
            print(f"  Primeiros 3 valores: {dados[etapa]['geradores'][:3] if isinstance(dados[etapa]['geradores'], list) else dados[etapa]['geradores']}")

    # Função para extrair geração de forma robusta
    def extrair_geracao_robusta(dados_geradores, etapa, n_barras=None):
        """Extrai dados de geração de forma robusta para diferentes estruturas"""
        geracao = {}
        
        if dados_geradores is None:
            return geracao
        
        # Caso 1: Lista de valores (formato mais comum)
        if isinstance(dados_geradores, list):
            print(f"  {etapa}: É uma lista com {len(dados_geradores)} elementos")
            
            # Se não especificamos n_barras, usamos o comprimento da lista
            if n_barras is None:
                n_barras = len(dados_geradores)
            
            # Para cada barra, atribuir o valor correspondente
            for i in range(min(n_barras, len(dados_geradores))):
                valor = dados_geradores[i]
                # Verificar se o valor é numérico e significativo
                if isinstance(valor, (int, float)):
                    # Arredondar valores muito pequenos para zero
                    if abs(valor) < 1e-6:
                        valor = 0.0
                    geracao[f"Barra {i+1}"] = float(valor)
                else:
                    # Se não for numérico, tentar converter
                    try:
                        valor_float = float(valor)
                        if abs(valor_float) < 1e-6:
                            valor_float = 0.0
                        geracao[f"Barra {i+1}"] = valor_float
                    except:
                        geracao[f"Barra {i+1}"] = 0.0
            
            # Filtrar barras com geração zero (opcional)
            # geracao = {k: v for k, v in geracao.items() if abs(v) > 1e-6}
            
            return geracao
        
        # Caso 2: Dicionário (formato menos comum)
        elif isinstance(dados_geradores, dict):
            print(f"  {etapa}: É um dicionário com {len(dados_geradores)} chaves")
            
            for key, value in dados_geradores.items():
                if isinstance(value, (int, float)):
                    if abs(value) < 1e-6:
                        value = 0.0
                    geracao[key] = float(value)
                elif isinstance(value, dict):
                    # Se for um dicionário aninhado, procurar por 'P_ger' ou similar
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            geracao[f"{key}_{subkey}"] = float(subvalue)
            
            return geracao
        
        # Caso 3: Outros tipos
        else:
            print(f"  {etapa}: Tipo não reconhecido: {type(dados_geradores)}")
            return geracao

    # Coletar geração de todas as etapas
    geracao_por_etapa = {}
    for etapa in etapas_com_dados:
        if dados[etapa]['geradores'] is not None:
            # Passar o número de barras para a função
            geracao = extrair_geracao_robusta(dados[etapa]['geradores'], etapa, n_barras)
            if geracao:  # Só adicionar se extraiu algum dado
                geracao_por_etapa[etapa] = geracao
                print(f"✓ {etapa}: Extraídos {len(geracao)} valores de geração")
                
                # Mostrar resumo
                total_geracao = sum(geracao.values())
                num_barras_ativas = sum(1 for v in geracao.values() if abs(v) > 1e-6)
                print(f"    Total: {total_geracao:.3f} pu em {num_barras_ativas} barras")
            else:
                print(f"✗ {etapa}: Não foi possível extrair dados de geração")

    # Se não conseguiu extrair dados de nenhuma etapa, mostrar mensagem
    if not geracao_por_etapa:
        print("⚠️  Nenhum dado de geração pôde ser extraído!")
        ax4.text(0.5, 0.5, 'Dados de geração não puderam ser extraídos', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax4.set_title('Geração por Barra (Sem dados)')
    else:
        # Determinar quais barras mostrar (todas as barras que aparecem em qualquer etapa)
        todas_barras = set()
        for etapa, geracao in geracao_por_etapa.items():
            todas_barras.update(geracao.keys())
        
        # Ordenar barras numericamente (Barra 1, Barra 2, ...)
        def sort_barra_key(barra):
            try:
                # Extrair número da barra
                num = int(barra.split()[1])
                return num
            except:
                return 999  # Colocar no final se não conseguir extrair número
        
        barras_plot = sorted(list(todas_barras), key=sort_barra_key)
        n_barras_plot = len(barras_plot)
        
        print(f"\n📊 Plotando geração para {n_barras_plot} barras: {barras_plot}")
        
        # Limitar a um número razoável para visualização
        MAX_BARRAS_PLOT = 15
        if n_barras_plot > MAX_BARRAS_PLOT:
            print(f"⚠️  Limitando a {MAX_BARRAS_PLOT} barras para melhor visualização")
            barras_plot = barras_plot[:MAX_BARRAS_PLOT]
            n_barras_plot = MAX_BARRAS_PLOT
        
        # Configurar posições das barras
        n_etapas = len(geracao_por_etapa)
        if n_etapas == 0:
            ax4.text(0.5, 0.5, 'Nenhum dado de geração disponível', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Geração por Barra (Sem dados)')
        else:
            width = 0.8 / n_etapas if n_etapas > 0 else 0.6
            
            for idx, (etapa, geracao) in enumerate(geracao_por_etapa.items()):
                valores = []
                for barra in barras_plot:
                    valor = geracao.get(barra, 0.0)
                    valores.append(valor)
                
                # Posições das barras (centralizadas)
                posicoes = [i + idx*width - (n_etapas-1)*width/2 for i in range(n_barras_plot)]
                
                # Plotar barras
                bars = ax4.bar(posicoes, valores, width=width, label=etapa,
                            alpha=0.7, color=cores.get(etapa, 'gray'))
                
                # Adicionar valores nas barras (apenas se > 0.001)
                for bar, valor in zip(bars, valores):
                    height = bar.get_height()
                    if abs(height) > 0.001:  # Mostrar valores significativos
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Configurar eixos
            ax4.set_xlabel('Barra')
            ax4.set_ylabel('Geração Ativa (pu)')
            ax4.set_title(f'Geração Ativa por Barra - Hora {hora_selecionada}')
            
            # Rotacionar labels se houver muitas barras
            rotation = 45 if n_barras_plot > 5 else 0
            
            ax4.set_xticks(range(n_barras_plot))
            
            # Encurtar labels para melhor visualização
            labels = []
            for barra in barras_plot:
                try:
                    # Extrair apenas o número da barra
                    num = barra.split()[1]
                    labels.append(f"B{num}")
                except:
                    labels.append(barra[:10])  # Limitar tamanho
            
            ax4.set_xticklabels(labels, rotation=rotation, fontsize=9)
            
            # Adicionar legenda
            ax4.legend()
            
            # Adicionar grid
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Adicionar linha horizontal em zero para referência
            ax4.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
            
            # Ajustar limites do eixo Y para incluir um pequeno espaço acima
            if valores:
                max_val = max(abs(v) for v in valores if abs(v) > 0.001)
                if max_val > 0:
                    ax4.set_ylim(0, max_val * 1.2)
            
            # Adicionar nota se há mais barras
            if len(todas_barras) > n_barras_plot:
                ax4.text(0.98, 0.98, f'+{len(todas_barras) - n_barras_plot} mais', 
                        transform=ax4.transAxes, ha='right', va='top',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Adicionar anotação sobre escala
            total_por_etapa = {}
            for etapa, geracao in geracao_por_etapa.items():
                total = sum(v for v in geracao.values() if abs(v) > 0.001)
                total_por_etapa[etapa] = total
            
            if total_por_etapa:
                texto = "Total: "
                for etapa, total in total_por_etapa.items():
                    texto += f"{etapa}={total:.3f} pu "
                ax4.text(0.02, 0.98, texto.strip(), 
                        transform=ax4.transAxes, ha='left', va='top',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Salvar figura no diretório de dados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figura_path = os.path.join(DIRETORIO_DADOS, f'comparacao_etapas_{timestamp}.png')
    #plt.savefig(figura_path, dpi=300, bbox_inches='tight')

    print(f"✅ Gráficos salvos em: {figura_path}")
    plt.show()

# ==================== FUNÇÃO AUXILIAR ATUALIZADA ====================

def carregar_resultados_para_graficos(arquivo_pl, arquivo_pf, arquivo_opf, hora=0):
    """Carrega resultados das 3 etapas para geração de gráficos para uma hora específica"""
    dados = {
        'PL': {'V_mag': None, 'V_ang': None, 'fluxos': None, 'geradores': None},
        'PF': {'V_mag': None, 'V_ang': None, 'fluxos': None, 'geradores': None},
        'OPF': {'V_mag': None, 'V_ang': None, 'fluxos': None, 'geradores': None},
    }
    
    # Carregar dados do PL
    if arquivo_pl and os.path.exists(arquivo_pl):
        try:
            conn = sqlite3.connect(arquivo_pl)
            cursor = conn.cursor()
            cursor.execute("SELECT tensoes_mag_json, tensoes_ang_json, fluxos_json, pg_json FROM resultados_PL WHERE hora = ?", (hora,))
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                dados['PL']['V_mag'] = json.loads(resultado[0])
                dados['PL']['V_ang'] = json.loads(resultado[1])
                dados['PL']['fluxos'] = json.loads(resultado[2])
                dados['PL']['geradores'] = json.loads(resultado[3])
                print(f"✓ Dados do PL carregados para hora {hora}")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar dados do PL para gráficos (hora {hora}): {e}")
    
    # Carregar dados do PF
    if arquivo_pf and os.path.exists(arquivo_pf):
        try:
            conn = sqlite3.connect(arquivo_pf)
            cursor = conn.cursor()
            cursor.execute("SELECT tensoes_mag_json, tensoes_ang_json, fluxos_json, P_gerado_json FROM resultados_fluxo WHERE hora = ?", (hora,))
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                dados['PF']['V_mag'] = json.loads(resultado[0])
                dados['PF']['V_ang'] = json.loads(resultado[1])
                dados['PF']['fluxos'] = json.loads(resultado[2])
                dados['PF']['geradores'] = json.loads(resultado[3])
                print(f"✓ Dados do PF carregados para hora {hora}")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar dados do PF para gráficos (hora {hora}): {e}")
    
    # Carregar dados do OPF
    if arquivo_opf and os.path.exists(arquivo_opf):
        try:
            conn = sqlite3.connect(arquivo_opf)
            cursor = conn.cursor()
            cursor.execute("SELECT tensoes_mag_json, tensoes_ang_json, fluxos_json, P_gerado_json FROM resultados_OPF WHERE hora = ?", (hora,))
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                dados['OPF']['V_mag'] = json.loads(resultado[0])
                dados['OPF']['V_ang'] = json.loads(resultado[1])
                dados['OPF']['fluxos'] = json.loads(resultado[2])
                dados['OPF']['geradores'] = json.loads(resultado[3])
                print(f"✓ Dados do OPF carregados para hora {hora}")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar dados do OPF para gráficos (hora {hora}): {e}")
    
    return dados

# ==================== ATUALIZAR A CHAMADA NA FUNÇÃO PRINCIPAL ====================

# Na função main(), altere a chamada para:
# gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf, hora_escolha='aleatoria')
# ou
# gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf, hora_escolha='primeira')
# ou
# gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf, hora_escolha='ultima')
# ou
# gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf, hora_escolha=12)  # hora específica

def RodadaEncadeada():
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
            gerar_graficos_simples(arquivo_pl, arquivo_pf, arquivo_opf, hora_escolha=12)
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
    
    RodadaEncadeada()