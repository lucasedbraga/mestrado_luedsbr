import sys
import os
import json
import random
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from SRC.modelos_matematicos.fluxPotContingencia import *

class StudentSearch:
    def __init__(self, pop_size=10, max_iter=100, prob_mutacao=0.1):
        # Carregar dados
        with open("DATA/input/B6L8_BASE.json", "r") as f:
            data = json.load(f)
        
        self.sistema = criar_sistema(data)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.prob_mutacao = prob_mutacao
        self.melhor_fitness = float('inf')
        self.melhor_candidato = None
        self.historico_fitness = []
        
    def gerar_candidato_aleatorio(self):
        """Gera um candidato aleatório (vetor binário indicando quais linhas serão investidas)"""
        n_linhas = self.sistema.NLIN
        
        while True:
            # Gera vetor onde 1 indica que a linha será investida (ativa para contingências)
            candidato = [random.randint(0, 1) for _ in range(n_linhas)]
            
            # Verifica se pelo menos uma linha foi selecionada para investimento
            if sum(candidato) > 0:
                return candidato
    
    def avaliar_candidato(self, candidato):
        """
        Nova lógica: Quanto MAIOR o impacto da contingência, MAIS a linha deve ser investida
        """
        sistema = self.sistema
        FITNESS = 0
        
        linhas_investidas = [i for i, investida in enumerate(candidato) if investida == 1]
        
        # 1. Caso base (referência)
        resultado_base = resolver_opf(sistema, sistema.Bbus_base, None, "caso_base")
        fob_base = resultado_base.custo + sum(resultado_base.dual_power_balance) + 10 * resultado_base.deficit_total
        
        # 2. Avaliar impacto das contingências NAS LINHAS INVESTIDAS
        impacto_total = 0
        
        for linha_idx in linhas_investidas:
            linha_id = sistema.linhas[linha_idx]["ID_linha"]
            
            # Impacto da REMOÇÃO (quanto PIOR, mais importante é a linha)
            linhas_ativas = [True] * sistema.NLIN
            linhas_ativas[linha_idx] = False
            
            Bbus_remocao = calcular_Bbus_linhas_ativas(sistema, linhas_ativas)
            resultado_remocao = resolver_opf(sistema, Bbus_remocao, linhas_ativas, f"remocao_{linha_id}")
            
            if resultado_remocao is not None:
                fob_remocao = resultado_remocao.custo + sum(resultado_remocao.dual_power_balance) + 10 * resultado_remocao.deficit_total
                impacto_remocao = fob_remocao - fob_base
                impacto_total += impacto_remocao
            
            # Impacto da DUPLICAÇÃO (quanto MELHOR, mais benéfico é investir)
            Bbus_dup = calcular_Bbus_duplicar_linha(sistema, linha_id)
            resultado_dup = resolver_opf(sistema, Bbus_dup, None, f"duplicacao_{linha_id}")
            
            if resultado_dup is not None:
                fob_dup = resultado_dup.custo + sum(resultado_dup.dual_power_balance) + 10 * resultado_dup.deficit_total
                beneficio_dup = fob_base - fob_dup  # Quanto MAIOR, mais benéfico duplicar
                impacto_total += max(0, beneficio_dup)  # Só considera benefícios
        
        # 3. Custo do investimento (penaliza muitas linhas)
        custo_investimento = sum(candidato) * 1000  # Custo fixo por linha investida
        
        # 4. FUNÇÃO FITNESS FINAL
        # Queremos MAXIMIZAR: impacto_total (benefício das linhas críticas)
        # Queremos MINIMIZAR: custo_investimento (número de linhas)
        fitness = impacto_total + custo_investimento
        
        print(f"\nCandidato: {linhas_investidas}")
        print(f"Impacto total: {impacto_total:.2f}")
        print(f"Custo investimento: {custo_investimento:.2f}")
        print(f"Fitness: {fitness:.2f}")
        
        return fitness, impacto_total, custo_investimento
        
    def mutacao(self, candidato):
        """Aplica mutação em um candidato, garantindo que pelo menos uma linha seja selecionada"""
        while True:
            candidato_mutado = candidato.copy()
            n_linhas = len(candidato)
            
            for i in range(n_linhas):
                if random.random() < self.prob_mutacao:
                    candidato_mutado[i] = 1 - candidato_mutado[i]  # Flip bit
            
            # Verifica se pelo menos uma linha foi selecionada
            if sum(candidato_mutado) > 0:
                return candidato_mutado
    
    def crossover(self, pai1, pai2):
        """Realiza crossover entre dois pais, garantindo que os filhos tenham pelo menos uma linha selecionada"""
        while True:
            ponto_corte = random.randint(1, len(pai1) - 1)
            filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
            filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
            
            # Verifica se ambos os filhos têm pelo menos uma linha selecionada
            if sum(filho1) > 0 and sum(filho2) > 0:
                return filho1, filho2
    
    def executar_busca(self):
        """Executa a metaheurística Student Search"""
        print("=== INICIANDO STUDENT SEARCH ===")
        print(f"População: {self.pop_size}, Iterações: {self.max_iter}")
        
        # Inicializar população
        populacao = [self.gerar_candidato_aleatorio() for _ in range(self.pop_size)]
        fitness_populacao = []
        
        # Avaliar população inicial
        for candidato in populacao:
            fitness, fob, n_linhas = self.avaliar_candidato(candidato)
            fitness_populacao.append((fitness, fob, n_linhas, candidato))
        
        # Ordenar por fitness (menor é melhor)
        fitness_populacao.sort(key=lambda x: x[0])
        self.melhor_fitness, melhor_fob, melhor_linhas, self.melhor_candidato = fitness_populacao[0]
        
        print(f"\nMelhor inicial - Fitness: {self.melhor_fitness:.2f}, FOB: {melhor_fob:.2f}, Linhas: {melhor_linhas}")
        
        # Loop principal da metaheurística
        for iteracao in range(self.max_iter):
            print(f"\n--- Iteração {iteracao + 1}/{self.max_iter} ---")
            
            nova_populacao = []
            
            # Manter os melhores (elitismo)
            n_elite = max(2, self.pop_size // 5)
            nova_populacao.extend([candidato for _, _, _, candidato in fitness_populacao[:n_elite]])
            
            # Gerar nova população através de crossover e mutação
            while len(nova_populacao) < self.pop_size:
                # Selecionar pais por torneio
                pai1 = self.torneio(fitness_populacao)
                pai2 = self.torneio(fitness_populacao)
                
                # Crossover
                filho1, filho2 = self.crossover(pai1, pai2)
                filho1 = self.mutacao(filho1)
                filho2 = self.mutacao(filho2)
                
                nova_populacao.extend([filho1, filho2])
            
            # Avaliar nova população
            fitness_populacao = []
            for candidato in nova_populacao[:self.pop_size]:
                fitness, fob, n_linhas = self.avaliar_candidato(candidato)
                fitness_populacao.append((fitness, fob, n_linhas, candidato))
            
            # Ordenar por fitness
            fitness_populacao.sort(key=lambda x: x[0])
            
            # Atualizar melhor candidato global
            if fitness_populacao[0][0] < self.melhor_fitness:
                self.melhor_fitness, melhor_fob, melhor_linhas, self.melhor_candidato = fitness_populacao[0]
                print(f"Novo melhor - Fitness: {self.melhor_fitness:.2f}, FOB: {melhor_fob:.2f}, Linhas: {melhor_linhas}")
            
            self.historico_fitness.append(self.melhor_fitness)
        
        return self.melhor_candidato, self.melhor_fitness
    
    def torneio(self, fitness_populacao, k=3):
        """Seleção por torneio"""
        participantes = random.sample(fitness_populacao, k)
        participantes.sort(key=lambda x: x[0])  # Ordena por fitness (menor é melhor)
        return participantes[0][3]  # Retorna o candidato
    
    def mostrar_resultado(self):
        """Mostra o resultado final da busca"""
        if self.melhor_candidato is None:
            print("Nenhuma solução encontrada.")
            return
        
        # Reavaliar o melhor candidato para obter os valores corretos
        fitness, fob, n_linhas = self.avaliar_candidato(self.melhor_candidato)
        
        linhas_investidas = [i for i, investida in enumerate(self.melhor_candidato) if investida == 1]
        nomes_linhas = [self.sistema.linhas[i]["ID_linha"] for i in linhas_investidas]
        
        print("\n" + "="*50)
        print("🎯 RESULTADO FINAL - STUDENT SEARCH")
        print("="*50)
        print(f"Melhor Fitness: {fitness:.2f}")
        print(f"FOB Total: {fob:.2f}")
        print(f"Número de linhas ativas totais: {n_linhas}")
        print(f"Linhas selecionadas para investimento ({len(linhas_investidas)}):")
        for i, idx_linha in enumerate(linhas_investidas):
            nome_linha = self.sistema.linhas[idx_linha]["ID_linha"]
            print(f"  {i+1}. Linha {idx_linha} ({nome_linha})")
        print("="*50)

if __name__ == '__main__':
    # Para executar a metaheurística:
    student_search = StudentSearch(pop_size=8, max_iter=20, prob_mutacao=0.1)
    melhor_candidato, melhor_fitness = student_search.executar_busca()
    student_search.mostrar_resultado()