"""
Script para executar múltiplas simulações e treinar rede neural com os resultados
Sistema genérico para qualquer número de barras
"""
import os
import sys
import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório raiz ao path
DIRETORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, DIRETORIO_RAIZ)

# Tentar importar o sistema principal
try:
    from SRC.SistemaMultiAgente.EncadeiamentoModelos import *
    print("✅ Sistema principal importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar sistema principal: {e}")
    print("Certifique-se de que o arquivo main.py está no caminho correto")
    sys.exit(1)

# ==================== CONFIGURAÇÕES ====================
NUM_SIMULACOES = 1000
TAMANHO_TREINAMENTO = int(round(0.8*NUM_SIMULACOES))
TAMANHO_TESTE = int(round(0.2*NUM_SIMULACOES))

# Caminhos dos arquivos
SISTEMA_JSON = os.path.join(DIRETORIO_RAIZ, "DATA", "input", "B6L8_BASE.json")
DIRETORIO_DADOS = os.path.join(DIRETORIO_RAIZ, "DATA", "SMA")
DIRETORIO_RNA = os.path.join(DIRETORIO_RAIZ, "DATA", "SMA", "RNA")

# ==================== FUNÇÕES AUXILIARES ====================

def criar_diretorios():
    """Cria os diretórios necessários"""
    for diretorio in [DIRETORIO_DADOS, DIRETORIO_RNA]:
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
            print(f"Criado diretório: {diretorio}")

def carregar_dados_sistema():
    """Carrega os dados do sistema para obter informações estruturais"""
    try:
        with open(SISTEMA_JSON, 'r') as f:
            dados = json.load(f)

        print(SISTEMA_JSON)
        print(dados)
        
        n_barras = len(dados['BARRAS'])
        n_geradores = len(dados['GERADORES'])
        S_base = dados.get('S_base', 100.0)
        
        print(f"\n📋 Informações do sistema:")
        print(f"  • Barras: {n_barras}")
        print(f"  • Geradores: {n_geradores}")
        print(f"  • S_base: {S_base} MVA")
        
        return dados, n_barras, n_geradores, S_base
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados do sistema: {e}")
        return None, 0, 0, 100.0

def executar_simulacao_unica(simulacao_id):
    """Executa uma única simulação do sistema"""
    print(f"\n🎯 Executando simulação {simulacao_id}...")
    
    try:
        # Executar o sistema principal
        RodadaEncadeada()
        
        # Coletar resultados do banco de dados
        resultados = coletar_resultados_simulacao()
        
        if resultados:
            print(f"✓ Simulação {simulacao_id} concluída com sucesso")
            print(f"  Amostras coletadas: {len(resultados)} horas")
            return resultados
        else:
            print(f"✗ Simulação {simulacao_id} não gerou resultados válidos")
            return None
            
    except Exception as e:
        print(f"❌ Erro na simulação {simulacao_id}: {e}")
        return None

def coletar_resultados_simulacao():
    """Coleta resultados de todas as horas da simulação atual"""
    resultados = []
    
    # Caminhos dos bancos de dados
    db_pf = os.path.join(DIRETORIO_DADOS, "resultados_PF.db")
    db_opf = os.path.join(DIRETORIO_DADOS, "resultados_OPF.db")
    
    if not os.path.exists(db_pf) or not os.path.exists(db_opf):
        print("❌ Bancos de dados não encontrados")
        return resultados
    
    try:
        # Conectar aos bancos de dados
        conn_pf = sqlite3.connect(db_pf)
        conn_opf = sqlite3.connect(db_opf)
        
        cursor_pf = conn_pf.cursor()
        cursor_opf = conn_opf.cursor()
        
        # Obter todas as horas disponíveis
        cursor_pf.execute("SELECT DISTINCT hora FROM resultados_fluxo ORDER BY hora")
        horas = [row[0] for row in cursor_pf.fetchall()]
        
        for hora in horas:
            try:
                # Buscar dados do PF
                cursor_pf.execute("""
                    SELECT tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
                           P_carga_json, Q_carga_json
                    FROM resultados_fluxo 
                    WHERE hora = ?
                """, (hora,))
                resultado_pf = cursor_pf.fetchone()
                
                # Buscar dados do OPF
                cursor_opf.execute("""
                    SELECT tensoes_mag_json, tensoes_ang_json, P_gerado_json, Q_gerado_json,
                           custo_total, curtailment_total, deficit_total, perdas_ativas,
                           custo_curtailment, custo_geracao
                    FROM resultados_OPF 
                    WHERE hora = ? AND sucesso = 1
                """, (hora,))
                resultado_opf = cursor_opf.fetchone()
                
                if resultado_pf and resultado_opf:
                    # Processar dados do PF
                    V_mag_pf = json.loads(resultado_pf[0])
                    V_ang_pf = json.loads(resultado_pf[1])
                    P_ger_pf = json.loads(resultado_pf[2])
                    Q_ger_pf = json.loads(resultado_pf[3])
                    
                    # Processar dados do OPF
                    V_mag_opf = json.loads(resultado_opf[0])
                    V_ang_opf = json.loads(resultado_opf[1])
                    P_ger_opf = json.loads(resultado_opf[2])
                    Q_ger_opf = json.loads(resultado_opf[3])
                    
                    # Coletar métricas
                    custo_total = float(resultado_opf[4])
                    curtailment = float(resultado_opf[5])
                    deficit = float(resultado_opf[6])
                    perdas = float(resultado_opf[7])
                    custo_curtailment = float(resultado_opf[8])
                    custo_geracao = float(resultado_opf[9])
                    
                    # Criar amostra
                    amostra = {
                        'hora': hora,
                        'V_mag_pf': V_mag_pf,
                        'V_ang_pf': V_ang_pf,
                        'P_ger_pf': P_ger_pf,
                        'Q_ger_pf': Q_ger_pf,
                        'V_mag_opf': V_mag_opf,
                        'V_ang_opf': V_ang_opf,
                        'P_ger_opf': P_ger_opf,
                        'Q_ger_opf': Q_ger_opf,
                        'custo_total': custo_total,
                        'curtailment': curtailment,
                        'deficit': deficit,
                        'perdas': perdas,
                        'custo_curtailment': custo_curtailment,
                        'custo_geracao': custo_geracao
                    }
                    
                    resultados.append(amostra)
                    
            except Exception as e:
                print(f"  ⚠️  Erro ao processar hora {hora}: {e}")
                continue
        
        conn_pf.close()
        conn_opf.close()
        
        print(f"  Coletadas {len(resultados)} amostras válidas")
        
    except Exception as e:
        print(f"❌ Erro ao coletar resultados: {e}")
    
    return resultados

def preparar_dados_treinamento(todas_amostras, n_barras):
    """Prepara os dados para treinamento da RNA"""
    print(f"\n📊 Preparando dados para treinamento...")
    print(f"  Total de amostras disponíveis: {len(todas_amostras)}")
    
    # Verificar se temos amostras suficientes
    amostras_necessarias = TAMANHO_TREINAMENTO + TAMANHO_TESTE
    if len(todas_amostras) < amostras_necessarias:
        print(f"⚠️  Amostras insuficientes: {len(todas_amostras)} < {amostras_necessarias}")
        # Usar todas as amostras disponíveis
        amostras_disponiveis = len(todas_amostras)
        TAMANHO_TREINAMENTO_REAL = int(amostras_disponiveis * 0.9)  # 90% para treino
        TAMANHO_TESTE_REAL = amostras_disponiveis - TAMANHO_TREINAMENTO_REAL
    else:
        TAMANHO_TREINAMENTO_REAL = TAMANHO_TREINAMENTO
        TAMANHO_TESTE_REAL = TAMANHO_TESTE
        amostras_disponiveis = TAMANHO_TREINAMENTO_REAL + TAMANHO_TESTE_REAL
        todas_amostras = todas_amostras[:amostras_disponiveis]
    
    print(f"  Amostras para treinamento: {TAMANHO_TREINAMENTO_REAL}")
    print(f"  Amostras para teste: {TAMANHO_TESTE_REAL}")
    
    # Preparar arrays
    X_data = []
    Y_data = []
    X_test = []
    Y_test = []
    
    for i, amostra in enumerate(todas_amostras):
        # Codificar hora como variável cíclica
        hora = amostra['hora']
        hora_cos = np.cos(2 * np.pi * hora / 24)
        hora_sen = np.sin(2 * np.pi * hora / 24)
        
        # Entradas (X): hora, tensões PF, gerações PF
        X = [hora_cos, hora_sen]
        X.extend(amostra['V_mag_pf'])  # Tensões
        
        # Para sistemas maiores, pode ser necessário selecionar apenas algumas barras
        # ou usar PCA para redução de dimensionalidade
        X.extend(amostra['P_ger_pf'][:n_barras])  # Gerações ativas
        
        # Saídas (Y): gerações ótimas, curtailment, déficit
        Y = amostra['P_ger_opf'][:n_barras]  # Gerações ótimas
        Y.append(amostra['curtailment'])
        Y.append(amostra['deficit'])
        
        # Separar em treinamento e teste
        if i < TAMANHO_TREINAMENTO_REAL:
            X_data.append(X)
            Y_data.append(Y)
        else:
            X_test.append(X)
            Y_test.append(Y)
    
    # Converter para numpy arrays
    X_train_array = np.array(X_data, dtype=np.float32)
    Y_train_array = np.array(Y_data, dtype=np.float32)
    X_test_array = np.array(X_test, dtype=np.float32)
    Y_test_array = np.array(Y_test, dtype=np.float32)
    
    print(f"\n📐 Dimensões dos conjuntos de dados:")
    print(f"  X_train: {X_train_array.shape}")
    print(f"  Y_train: {Y_train_array.shape}")
    print(f"  X_test:  {X_test_array.shape}")
    print(f"  Y_test:  {Y_test_array.shape}")
    
    # Verificar se há valores NaN
    if np.any(np.isnan(X_train_array)) or np.any(np.isnan(Y_train_array)):
        print("⚠️  Aviso: Valores NaN encontrados nos dados de treinamento")
        # Preencher NaNs com 0
        X_train_array = np.nan_to_num(X_train_array)
        Y_train_array = np.nan_to_num(Y_train_array)
    
    if np.any(np.isnan(X_test_array)) or np.any(np.isnan(Y_test_array)):
        print("⚠️  Aviso: Valores NaN encontrados nos dados de teste")
        X_test_array = np.nan_to_num(X_test_array)
        Y_test_array = np.nan_to_num(Y_test_array)
    
    return X_train_array, Y_train_array, X_test_array, Y_test_array

def treinar_rede_neural(X_train, Y_train, X_test, Y_test, n_barras):
    """Treina uma rede neural usando scikit-learn"""
    print(f"\n🧠 Treinando Rede Neural...")
    
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Normalizar os dados
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        Y_test_scaled = scaler_Y.transform(Y_test)
        
        print(f"  Dados normalizados com sucesso")
        
        # Configurar rede neural
        # A arquitetura é adaptável ao tamanho do problema
        n_features = X_train.shape[1]
        n_outputs = Y_train.shape[1]
        
        # Definir arquitetura baseada na complexidade
        if n_features < 50:
            hidden_layers = (100, 50, 25)
        elif n_features < 100:
            hidden_layers = (200, 100, 50)
        else:
            hidden_layers = (300, 150, 75)
        
        print(f"  Arquitetura: {n_features} → {hidden_layers} → {n_outputs}")
        
        # Criar e treinar modelo
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='logistic',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=5000,
            shuffle=True,
            random_state=42,
            tol=0.001,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50
        )
        
        print("  Iniciando treinamento...")
        mlp.fit(X_train_scaled, Y_train_scaled)
        
        print(f"  Número de iterações: {mlp.n_iter_}")
        print(f"  Loss final: {mlp.loss_:.6f}")
        
        # Fazer previsões
        Y_train_pred_scaled = mlp.predict(X_train_scaled)
        Y_test_pred_scaled = mlp.predict(X_test_scaled)
        
        # Desnormalizar previsões
        Y_train_pred = scaler_Y.inverse_transform(Y_train_pred_scaled)
        Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_scaled)
        
        # Calcular métricas
        mse_train = mean_squared_error(Y_train, Y_train_pred)
        mse_test = mean_squared_error(Y_test, Y_test_pred)
        
        mae_train = mean_absolute_error(Y_train, Y_train_pred)
        mae_test = mean_absolute_error(Y_test, Y_test_pred)
        
        r2_train = r2_score(Y_train, Y_train_pred)
        r2_test = r2_score(Y_test, Y_test_pred)
        
        print(f"\n📈 Métricas de desempenho:")
        print(f"  TREINAMENTO:")
        print(f"    MSE:  {mse_train:.6f}")
        print(f"    MAE:  {mae_train:.6f}")
        print(f"    R²:   {r2_train:.6f}")
        print(f"  TESTE:")
        print(f"    MSE:  {mse_test:.6f}")
        print(f"    MAE:  {mae_test:.6f}")
        print(f"    R²:   {r2_test:.6f}")
        
        # Salvar modelo
        import joblib
        modelo_path = os.path.join(DIRETORIO_RNA, "modelo_rna.pkl")
        joblib.dump(mlp, modelo_path)
        joblib.dump(scaler_X, os.path.join(DIRETORIO_RNA, "scaler_X.pkl"))
        joblib.dump(scaler_Y, os.path.join(DIRETORIO_RNA, "scaler_Y.pkl"))
        
        print(f"\n💾 Modelo salvo em: {modelo_path}")
        
        return mlp, scaler_X, scaler_Y, Y_test_pred
        
    except ImportError as e:
        print(f"❌ Erro ao importar scikit-learn: {e}")
        print("Instale com: pip install scikit-learn joblib")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Erro ao treinar rede neural: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime
import os

def plotar_resultados_rna(Y_test, Y_pred, n_barras):
    print("\n📊 Gerando gráficos comparativos...")

    # ---- Garantir formato seguro 2D ----
    Y_test = np.array(Y_test, dtype=float)
    Y_pred = np.array(Y_pred, dtype=float)

    if Y_test.ndim == 0:
        Y_test = np.array([[Y_test]])
    if Y_pred.ndim == 0:
        Y_pred = np.array([[Y_pred]])

    if Y_test.ndim == 1:
        Y_test = Y_test.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    # Garantir que têm mesmo shape
    min_len = min(len(Y_test), len(Y_pred))
    Y_test = Y_test[:min_len]
    Y_pred = Y_pred[:min_len]

    n_amostras = min(min_len, 100)
    Y_test_p = Y_test[:n_amostras]
    Y_pred_p = Y_pred[:n_amostras]

    n_cols = Y_test_p.shape[1]

    # ---- Criar figura ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparação: RNA vs Valores Reais', fontsize=16, fontweight='bold')

    # # 1. Dispersão
    # ax = axes[0,0]
    yt = Y_test_p.flatten()
    yp = Y_pred_p.flatten()
    # ax.scatter(yt, yp, alpha=0.5, s=10)
    # ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--')
    # ax.set_title("Dispersão (todas as saídas)")
    # ax.set_xlabel("Real")
    # ax.set_ylabel("Predição")

    # try:
    #     ax.text(0.05, 0.95, f"R²={r2_score(yt, yp):.4f}",
    #             transform=ax.transAxes, ha="left",
    #             bbox=dict(facecolor='white', alpha=0.6))
    # except:
    #     pass

    # 2. Erro médio por amostra
    ax = axes[0,0]
    erro = np.abs(Y_test_p - Y_pred_p).mean(axis=1)
    ax.plot(erro, lw=1.5)
    ax.set_title("Erro absoluto médio por amostra")
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Erro médio")

    # 3. Barras de geração média
    ax = axes[0,1]
    barras = min(n_barras, n_cols)
    if barras > 0:
        ax.bar(np.arange(barras)-0.15, Y_test_p[:,:barras].mean(axis=0), width=0.3, label='Real')
        ax.bar(np.arange(barras)+0.15, Y_pred_p[:,:barras].mean(axis=0), width=0.3, label='RNA')
        ax.set_xticks(range(barras))
        ax.set_xticklabels([f'B{i+1}' for i in range(barras)])
        ax.set_title("Geração média por barra")
        ax.legend()
    else:
        ax.set_title("Sem barras para plotar")

    # 4. Curtailment - penúltima
    ax = axes[1,0]
    if n_cols >= 2:
        ax.plot(np.round(Y_test_p[:,-2]), label="Real")
        ax.plot(np.round(Y_pred_p[:,-2]), label="RNA")
        ax.set_title("Curtailment")
        ax.legend()
    else:
        ax.text(0.5,0.5,"Sem coluna",ha="center")

    # 5. Déficit - última
    ax = axes[1,1]
    ax.set_title("Déficit")
    if n_cols >= 1:
        ax.plot(Y_test_p[:,-1], label="Real")
        ax.plot(Y_pred_p[:,-1], label="RNA")
        ax.legend()
    else:
        ax.text(0.5,0.5,"Sem coluna",ha="center")

    # # 6. Histograma dos erros
    # ax = axes[2,1]
    dif = yt - yp
    # ax.hist(dif, bins=40, edgecolor='black')
    # ax.set_title("Distribuição de erros")

    # plt.tight_layout()

    # Salvar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(DIRETORIO_RNA, f'comparacao_rna_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {path}")

    print("\n📊 Estatísticas:")
    print("  Erro médio:", float(np.mean(np.abs(dif))))
    print("  Máx erro:", float(np.max(np.abs(dif))))
    print("  Mín erro:", float(np.min(np.abs(dif))))

    return fig

def salvar_dados_treinamento(X_train, Y_train, X_test, Y_test):
    """Salva os dados de treinamento em arquivos NPY"""
    print(f"\n💾 Salvando dados de treinamento...")
    
    # Salvar arrays numpy
    np.save(os.path.join(DIRETORIO_RNA, 'X_train.npy'), X_train)
    np.save(os.path.join(DIRETORIO_RNA, 'Y_train.npy'), Y_train)
    np.save(os.path.join(DIRETORIO_RNA, 'X_test.npy'), X_test)
    np.save(os.path.join(DIRETORIO_RNA, 'Y_test.npy'), Y_test)
    
    # Salvar informações sobre os dados
    info = {
        'data_criacao': datetime.now().isoformat(),
        'n_amostras_treino': len(X_train),
        'n_amostras_teste': len(X_test),
        'n_features': X_train.shape[1],
        'n_outputs': Y_train.shape[1],
        'X_train_shape': X_train.shape,
        'Y_train_shape': Y_train.shape,
        'X_test_shape': X_test.shape,
        'Y_test_shape': Y_test.shape
    }
    
    with open(os.path.join(DIRETORIO_RNA, 'info_dados.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"✅ Dados salvos em: {DIRETORIO_RNA}")
    print(f"   • X_train.npy ({X_train.shape})")
    print(f"   • Y_train.npy ({Y_train.shape})")
    print(f"   • X_test.npy ({X_test.shape})")
    print(f"   • Y_test.npy ({Y_test.shape})")
    print(f"   • info_dados.json")

# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal"""
    print("="*80)
    print("SISTEMA DE TREINAMENTO DE REDE NEURAL PARA OPF")
    print("="*80)
    print(f"Número de simulações: {NUM_SIMULACOES}")
    print(f"Tamanho treinamento: {TAMANHO_TREINAMENTO}")
    print(f"Tamanho teste: {TAMANHO_TESTE}")
    print("="*80)
    
    # Criar diretórios
    criar_diretorios()

    # Carregar dados do sistema
    dados_sistema, n_barras, n_geradores, S_base = carregar_dados_sistema()
    if dados_sistema is None:
        print("❌ Não foi possível carregar dados do sistema")
        return
    
    # Executar simulações e coletar dados
    print(f"\n🚀 Iniciando execução de {NUM_SIMULACOES} simulações...")
    
    todas_amostras = []
    amostras_coletadas = 0
    amostras_necessarias = TAMANHO_TREINAMENTO + TAMANHO_TESTE
    
    for i in range(NUM_SIMULACOES):
        print(f"\n{'='*60}")
        print(f"SIMULAÇÃO {i+1}/{NUM_SIMULACOES}")
        print(f"{'='*60}")
        
        # Executar simulação
        resultados = executar_simulacao_unica(i + 1)
        
        if resultados:
            todas_amostras.extend(resultados)
            amostras_coletadas += len(resultados)
            
            print(f"📈 Progresso: {amostras_coletadas} / {amostras_necessarias} amostras")
            
            # Verificar se já coletamos amostras suficientes
            if amostras_coletadas >= amostras_necessarias:
                print(f"✅ Amostras suficientes coletadas ({amostras_coletadas})")
                break
        
        # Salvar checkpoint a cada 10 simulações
        if (i + 1) % 10 == 0:
            print(f"\n💾 Checkpoint após {i+1} simulações")
            print(f"   Amostras coletadas: {amostras_coletadas}")
    
    # print(f"\n✅ Coleta de dados concluída")
    # print(f"   Total de amostras coletadas: {len(todas_amostras)}")
    
    # # Preparar dados para treinamento
    # if len(todas_amostras) < 100:
    #     print(f"❌ Amostras insuficientes para treinamento ({len(todas_amostras)} < 100)")
    #     return
    
    X_train, Y_train, X_test, Y_test = preparar_dados_treinamento(todas_amostras, n_barras)
    
    # Salvar dados
    salvar_dados_treinamento(X_train, Y_train, X_test, Y_test)
    
    # Treinar rede neural
    mlp, scaler_X, scaler_Y, Y_pred = treinar_rede_neural(X_train, Y_train, X_test, Y_test, n_barras)
    
    if mlp is not None:
        # Gerar gráficos comparativos
        fig = plotar_resultados_rna(Y_test, Y_pred, n_barras)
        
        # Mostrar resumo final
        print(f"\n{'='*80}")
        print("RESUMO DA EXECUÇÃO")
        print("="*80)
        print(f"• Simulações executadas: {min(NUM_SIMULACOES, i+1)}")
        print(f"• Amostras totais coletadas: {len(todas_amostras)}")
        print(f"• Amostras de treinamento: {len(X_train)}")
        print(f"• Amostras de teste: {len(X_test)}")
        print(f"• Número de features: {X_train.shape[1]}")
        print(f"• Número de outputs: {Y_train.shape[1]}")
        print(f"• Modelo salvo em: {DIRETORIO_RNA}")
        print("="*80)
        
        # Mostrar gráficos
        plt.show()
    else:
        print("❌ Falha no treinamento da rede neural")

# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    # Mudar para o diretório raiz
    os.chdir(DIRETORIO_RAIZ)
    
    # Executar sistema
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()