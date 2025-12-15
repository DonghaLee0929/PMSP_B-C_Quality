import numpy as np
import random
import copy
import math
from Env import Environment
from atcsq import ATCSQ_Scheduler

# ---------------------------------------------------------
# Genetic Algorithm Class
# ---------------------------------------------------------
class GeneticOptimizer:
    def __init__(self, 
                 pop_size=20, 
                 generations=10, 
                 mutation_rate=0.1, 
                 crossover_rate=0.7,
                 param_bounds={'k1': (0.1, 5.0), 'k2': (0.001, 0.1), 'k3': (0.1, 5.0)}):
        
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bounds = param_bounds
        self.population = []
        
        # 테스트할 40개의 인스턴스 설정 미리 생성
        self.test_instances = []
        for j in [100, 200]:
            for m in [20, 25]:
                for s in range(5, 105, 10):
                    self.test_instances.append((j, m, s))
                    
        print(f"GA Initialized: Optimizing over {len(self.test_instances)} instances.")

    def _init_population(self):
        """초기 인구 생성"""
        self.population = []
        for _ in range(self.pop_size):
            gene = {
                'k1': random.uniform(self.bounds['k1'][0], self.bounds['k1'][1]),
                'k2': random.uniform(self.bounds['k2'][0], self.bounds['k2'][1]),
                'k3': random.uniform(self.bounds['k3'][0], self.bounds['k3'][1])
            }
            self.population.append(gene)

    def _check_feasibility(self, env, schedules, base_params):
        """
        Feasibility 판별 (True/False 반환)
        """
        durations = np.array(env.duration)
        total_dur = durations.sum()
        current_yield_sum = 0
        current_var_sum = 0
        late_count = 0
        
        for m_idx, m_sched in enumerate(schedules):
            for job_info in m_sched:
                j_id, start, end, d, f = job_info
                p = durations[j_id]
                current_yield_sum += p * env.spec_cdf[j_id][m_idx]
                current_var_sum += p * env.scaled_v[j_id][m_idx]
                if end > d:
                    late_count += 1
        
        yield_ok = current_yield_sum >= total_dur * base_params['Gamma']
        var_ok = current_var_sum <= total_dur * base_params['Lambda']
        deadline_ok = (late_count == 0)
        
        return (yield_ok and var_ok and deadline_ok)

    def _evaluate_fitness(self, gene):
        """
        [변경된 평가 로직]
        목표: Setup 횟수 최소화 (Infeasible 시 페널티 부과)
        Fitness: - (Total Setups + Penalty)
        """
        total_cost = 0 # 낮을수록 좋음
        base_params = {'Gamma': 0.95, 'Lambda': 1.15}
        
        for (job_num, machine_num, seed) in self.test_instances:
            family_num = math.floor(job_num/8) + 1
            
            # 환경 생성 및 실행
            env = Environment(job_num, machine_num, family_num, setup_time=10)
            env.reset(seed=seed, tight='base', quality='high')
            
            scheduler = ATCSQ_Scheduler(
                env,
                Gamma=base_params['Gamma'],
                Lambda=base_params['Lambda'],
                k1=gene['k1'],
                k2=gene['k2'],
                k3=gene['k3']
            )
            
            schedules, total_setups = scheduler.solve()
            is_feasible = self._check_feasibility(env, schedules, base_params)
            
            # 비용 계산
            instance_cost = total_setups
            
            if not is_feasible:
                # 요청하신 페널티 공식: (작업 수 - 기계 수) 감점(비용 추가)
                penalty = (job_num - machine_num)
                instance_cost += penalty
            
            total_cost += instance_cost
            
        # GA는 Maximize하므로 음수로 변환
        return -total_cost

    def _select_parents(self, fitness_scores):
        """토너먼트 선택"""
        tournament_size = 3
        parent_idx = []
        for _ in range(2):
            candidates = random.sample(range(self.pop_size), tournament_size)
            winner = max(candidates, key=lambda i: fitness_scores[i])
            parent_idx.append(winner)
        return [self.population[i] for i in parent_idx]

    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            child1, child2 = {}, {}
            for key in ['k1', 'k2', 'k3']:
                if random.random() < 0.5:
                    child1[key] = p1[key]
                    child2[key] = p2[key]
                else:
                    child1[key] = p2[key]
                    child2[key] = p1[key]
            return child1, child2
        else:
            return copy.deepcopy(p1), copy.deepcopy(p2)

    def _mutate(self, gene):
        mutated = copy.deepcopy(gene)
        for key in ['k1', 'k2', 'k3']:
            if random.random() < self.mutation_rate:
                noise = random.gauss(0, 0.5) 
                mutated[key] += noise
                low, high = self.bounds[key]
                mutated[key] = max(low, min(high, mutated[key]))
        return mutated

    def run(self):
        self._init_population()
        
        best_gene_overall = None
        best_fitness_overall = -float('inf') # 음수이므로 매우 작은 값으로 초기화
        
        for gen in range(self.generations):
            fitness_scores = []
            for i, gene in enumerate(self.population):
                fit = self._evaluate_fitness(gene)
                fitness_scores.append(fit)
            
            max_fit = max(fitness_scores)
            best_idx = fitness_scores.index(max_fit)
            
            if max_fit > best_fitness_overall:
                best_fitness_overall = max_fit
                best_gene_overall = self.population[best_idx]
                
            # 점수가 음수이므로 보기 좋게 -부호를 붙여 Cost로 출력
            print(f"Gen {gen+1}/{self.generations} | Best Cost: {-max_fit:.1f} (Lower is better) | Overall Best: {-best_fitness_overall:.1f}")
            print(f"  -> Best Gene: {self.population[best_idx]}")

            new_population = []
            new_population.append(self.population[best_idx]) # Elitism
            
            while len(new_population) < self.pop_size:
                p1, p2 = self._select_parents(fitness_scores)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)
            
            self.population = new_population
            
        return best_gene_overall, best_fitness_overall

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    ga = GeneticOptimizer(
        pop_size=10, 
        generations=10, 
        mutation_rate=0.8,
        crossover_rate=0.8,
        param_bounds={
            'k1': (0.1, 10.0), 
            'k2': (0.0001, 1.0), 
            'k3': (0.1, 10.0) 
        }
    )
    
    print("=== Starting Genetic Algorithm Optimization (Minimizing Setups with Penalty) ===")
    best_gene, best_fitness = ga.run()
    
    print("\n=== Optimization Complete ===")
    print(f"Final Best Parameter Set: {best_gene}")
    print(f"Best Total Cost (Setups + Penalty): {-best_fitness}")