import random
import time
import numpy as np
import math

from Env import Environment
from utills import result_explain, new_datarecoder


# ============================================================
# Decode chromosome to schedule  (only used for final output)
# ============================================================
def decode(jobs, machine_assignment, sequence, n_machines, setup_time=3):
    """
    jobs: list of (p, f, d)
    returns schedule formatted as:
    [
    [(job_id, start, end, deadline, family), ...], # machine 0
    [(...), ...], # machine 1
    ...
    ]
    """
    machine_jobs = [[] for _ in range(n_machines)]
    for job in sequence:
        m = machine_assignment[job]
        machine_jobs[m].append(job)

    schedules = [[] for _ in range(n_machines)]
    total_setups = 0
    late_jobs = []

    for m in range(n_machines):
        current_time = 0
        prev_family = None

        for job in machine_jobs[m]:
            p, f, d = jobs[job]

            # setup check
            if prev_family is not None and prev_family != f:
                current_time += setup_time
                total_setups += 1

            start = current_time
            end = start + p
            current_time = end

            # Append schedule including deadline
            schedules[m].append((job, start, end, d, f))

            # late job check
            if end > d:
                late_jobs.append(job)

            prev_family = f

    return schedules, total_setups, late_jobs


# ============================================================
# Fast fitness evaluation with numpy vectorization
# ============================================================
def evaluate_batch(pop_seq, pop_ma, jobs, n_machines, setup_time,
                   spec_cdf, scaled_v, Gamma, Lambda,
                   late_penalty=1000, yield_penalty=1000, var_penalty=1000):

    n_pop = len(pop_seq)
    n_jobs = len(jobs)

    durations = np.array([p for (p, f, d) in jobs])
    deadlines = np.array([d for (p, f, d) in jobs])
    families = np.array([f for (p, f, d) in jobs])

    total_job_time = durations.sum()

    fitness = np.zeros(n_pop, dtype=float)

    for idx in range(n_pop):
        seq = pop_seq[idx]
        ma = np.array(pop_ma[idx])

        # ==== Compute tardiness violation with setup time ====
        machine_time = np.zeros(n_machines, dtype=float)
        setup_count = 0

        last_fam = np.full(n_machines, -1)

        for job in seq:
            m = ma[job]
            p = durations[job]
            f = families[job]

            if last_fam[m] != -1 and last_fam[m] != f:
                machine_time[m] += setup_time
                setup_count += 1

            start = machine_time[m]
            end = start + p
            machine_time[m] = end

            last_fam[m] = f

        # Late jobs count
        job_end_times = np.zeros(n_jobs)
        machine_time_tmp = np.zeros(n_machines)
        last_fam_tmp = np.full(n_machines, -1)

        for job in seq:
            m = ma[job]
            p = durations[job]
            f = families[job]

            if last_fam_tmp[m] != -1 and last_fam_tmp[m] != f:
                machine_time_tmp[m] += setup_time

            start = machine_time_tmp[m]
            end = start + p
            machine_time_tmp[m] = end

            job_end_times[job] = end
            last_fam_tmp[m] = f

        late_count = np.sum(job_end_times > deadlines)

        obj = setup_count + late_penalty * late_count

        # === Expected Yield ===
        exp_yield = np.sum(durations * spec_cdf[np.arange(n_jobs), ma])
        exp_var = np.sum(durations * scaled_v[np.arange(n_jobs), ma])

        if exp_yield < total_job_time * Gamma:
            obj += yield_penalty

        if exp_var > total_job_time * Lambda:
            obj += var_penalty

        fitness[idx] = obj

    return fitness


# ============================================================
# Tournament Selection
# ============================================================
def tournament_select(pop, fits, k=3):
    candidates = random.sample(range(len(pop)), k)
    best = min(candidates, key=lambda i: fits[i])
    return pop[best]


def ox_crossover(seq1, seq2):
    n = len(seq1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = seq1[a:b]
    pos = b
    for x in seq2:
        if x not in child:
            if pos >= n:
                pos = 0
            child[pos] = x
            pos += 1
    return child


def uniform_crossover(vec1, vec2):
    return [vec1[i] if random.random() < 0.5 else vec2[i] for i in range(len(vec1))]


def swap_mutation(seq, rate=0.1):
    if random.random() < rate:
        a, b = random.sample(range(len(seq)), 2)
        seq[a], seq[b] = seq[b], seq[a]
    return seq


def reassign_mutation(ma, n_machines, rate=0.1):
    if random.random() < rate:
        j = random.randint(0, len(ma)-1)
        ma[j] = random.randint(0, n_machines-1)
    return ma

def greedy_repair(seq, ma, jobs, n_machines, setup_time, deadlines):
    """
    One-pass greedy repair:
    If job exceeds deadline, try swapping with next job once.
    """
    # basic decoding to compute end times
    machine_time = np.zeros(n_machines, dtype=float)
    last_fam = np.full(n_machines, -1)
    job_end_times = np.zeros(len(seq), dtype=float)
    durations = np.array([p for (p,f,d) in jobs])
    families  = np.array([f for (p,f,d) in jobs])

    # simulate schedule
    for pos, job in enumerate(seq):
        m = ma[job]
        p = durations[job]
        f = families[job]

        if last_fam[m] != -1 and last_fam[m] != f:
            machine_time[m] += setup_time

        start = machine_time[m]
        end   = start + p
        machine_time[m] = end
        last_fam[m] = f

        job_end_times[job] = end

        # repair if late
        if end > deadlines[job] and pos < len(seq)-1:
            # try swap with next job
            seq[pos], seq[pos+1] = seq[pos+1], seq[pos]

            # stop after a single repair attempt
            break

    return seq

def heuristic_initialization(pop_size, jobs, n_machines, capable_machines):
    population = []
    n_jobs = len(jobs)
    
    # (capable_machines 계산 로직 제거 - 인자로 받음)

    for _ in range(pop_size):
        # 1. Sequence: EDD (Earliest Due Date) + Random noise
        base_seq = sorted(range(n_jobs), key=lambda x: jobs[x][2])
        
        # 부분적으로 섞어서 다양성 확보 (Swap 10%)
        for _ in range(int(n_jobs * 0.1)):
            a, b = random.sample(range(n_jobs), 2)
            base_seq[a], base_seq[b] = base_seq[b], base_seq[a]
            
        # 2. Machine Assignment: Load Balancing
        ma = [0] * n_jobs
        machine_loads = [0] * n_machines
        
        for job_idx in base_seq:
            p = jobs[job_idx][0]
            # 미리 계산된 리스트 사용
            candidates = capable_machines[job_idx]
            best_m = min(candidates, key=lambda m: machine_loads[m])
            
            ma[job_idx] = best_m
            machine_loads[best_m] += p
            
        population.append((base_seq, ma))
        
    return population

def family_grouping_mutation(seq, jobs, rate=0.1):
    """
    임의의 작업을 선택해서, 같은 Family를 가진 작업의 바로 옆으로 이동시킴
    """
    if random.random() > rate:
        return seq
    
    n = len(seq)
    target_idx = random.randint(0, n-1)
    target_job = seq[target_idx]
    target_fam = jobs[target_job][1] # Family of target job
    
    # 같은 Family를 가진 다른 작업들의 위치 찾기
    same_fam_indices = [i for i, j in enumerate(seq) if jobs[j][1] == target_fam and i != target_idx]
    
    if same_fam_indices:
        # 그 중 하나 옆으로 이동
        dest_idx = random.choice(same_fam_indices)
        
        # 리스트에서 제거 후 삽입
        new_seq = seq[:]
        new_seq.pop(target_idx)
        
        # pop으로 인해 인덱스가 밀렸을 수 있으므로 조정 필요하지만
        # 간단하게 insert 사용 (dest_idx가 target보다 뒤면 -1 보정 필요)
        if target_idx < dest_idx:
            dest_idx -= 1
            
        new_seq.insert(dest_idx + 1, target_job) # 바로 뒤에 붙임
        return new_seq
        
    return seq

# ============================================================
# GA
# ============================================================
def run_ga(jobs, n_machines, setup_time=3,
           pop_size=1000, time_limit=3600,
           max_generation=100000,
           crossover_rate=0.8, mutation_rate=0.8,
           spec_cdf=None, scaled_v=None,
           Gamma=0.9, Lambda=0.2, log_viz=True,
           patience=1000):

    start_time = time.time()
    n_jobs = len(jobs)
    generation = 0
    
    # Counter for generations without improvement
    no_improvement_count = 0 

    capable_machines = []
    for j in range(n_jobs):
        candidates = [m for m in range(n_machines) if spec_cdf[j, m] >= Gamma]
        if not candidates: 
             candidates = [np.argmax(spec_cdf[j])]
        capable_machines.append(candidates)

    # Initial population
    population = []
    
    # 50% Heuristic Initialization (한 번에 생성하도록 변경하여 속도 향상)
    n_heuristic = int(pop_size * 0.5)
    population += heuristic_initialization(n_heuristic, jobs, n_machines, capable_machines)

    # 50% Random/EDD Initialization
    n_random = pop_size - n_heuristic
    for _ in range(n_random):
        jobs_arr = np.array(jobs)
        families  = jobs_arr[:,1]
        deadlines = jobs_arr[:,2]
        seq = sorted(range(n_jobs), key=lambda j: (families[j], deadlines[j]))
        
        # 약간의 셔플 추가 (완전 정렬된 상태는 Local Optima에 빠지기 쉬움)
        if random.random() < 0.3:
             swap_mutation(seq, rate=1.0) # 강제 스왑

        ma = [random.randint(0, n_machines-1) for _ in range(n_jobs)]
        population.append((seq, ma))

    # Evaluate initial population
    pop_seq = [ind[0] for ind in population]
    pop_ma = [ind[1] for ind in population]

    fits = evaluate_batch(
        pop_seq, pop_ma, jobs, n_machines, setup_time,
        spec_cdf, scaled_v, Gamma, Lambda
    )

    best_idx = np.argmin(fits)
    best_fit = fits[best_idx]
    best_ind = population[best_idx]

    while generation < max_generation:

        new_pop = [best_ind]

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)

            seq1, ma1 = p1
            seq2, ma2 = p2

            if random.random() < crossover_rate:
                cseq = ox_crossover(seq1, seq2)
                cseq = family_grouping_mutation(cseq, jobs, mutation_rate)
                cma = uniform_crossover(ma1, ma2)
            else:
                cseq = seq1[:]
                cma = ma1[:]

            cseq = swap_mutation(cseq, mutation_rate)
            # cma = reassign_mutation(cma, n_machines, mutation_rate)
            if random.random() < mutation_rate:
                j = random.randint(0, len(cma)-1)
                # 단순히 랜덤 기계가 아니라, 해당 작업의 Yield가 높은 기계들 중에서 랜덤 선택
                candidates = [m for m in range(n_machines) if spec_cdf[j, m] >= Gamma] 
                if not candidates: candidates = range(n_machines)
                cma[j] = random.choice(candidates)

            deadlines = np.array([d for (p,f,d) in jobs])
            cseq = greedy_repair(cseq, cma, jobs, n_machines, setup_time, deadlines)

            new_pop.append((cseq, cma))

        population = new_pop

        pop_seq = [ind[0] for ind in population]
        pop_ma  = [ind[1] for ind in population]

        fits = evaluate_batch(
            pop_seq, pop_ma, jobs, n_machines, setup_time,
            spec_cdf, scaled_v, Gamma, Lambda
        )

        bi = np.argmin(fits)
        
        # Check for improvement
        if fits[bi] < best_fit:
            best_fit = fits[bi]
            best_ind = population[bi]
            no_improvement_count = 0 # Reset counter if solution improved
        else:
            no_improvement_count += 1 # Increment counter if no improvement

        # Check early stopping condition
        if no_improvement_count >= patience:
            if log_viz:
                print(f"\n[Early Stopping] No improvement for {patience} generations. Stopping at Gen {generation}.")
            break

        if not log_viz:
            generation += 1 # Increment generation before continue
            # Note: Logic adjusted to ensure generation increments properly even if log_viz is False
            if time.time() - start_time > time_limit:
                break
            continue

        if generation % 100 == 0:

            # Logging quality evaluation for best chromosome
            seq_best, ma_best = best_ind
            ma_best = np.array(ma_best)

            durations = np.array([p for (p, f, d) in jobs])
            deadlines = np.array([d for (p, f, d) in jobs])
            families = np.array([f for (p, f, d) in jobs])

            total_dur = durations.sum()
            n_jobs = len(jobs)

            # ===== Re-simulate end times to count late jobs =====
            machine_time_tmp = np.zeros(n_machines)
            last_fam_tmp = np.full(n_machines, -1)
            job_end_times = np.zeros(n_jobs)

            for job in seq_best:
                m = ma_best[job]
                p = durations[job]
                f = families[job]

                if last_fam_tmp[m] != -1 and last_fam_tmp[m] != f:
                    machine_time_tmp[m] += setup_time

                start = machine_time_tmp[m]
                end = start + p
                machine_time_tmp[m] = end

                job_end_times[job] = end
                last_fam_tmp[m] = f

            late_count = np.sum(job_end_times > deadlines)

            # ===== Quality Checks =====
            exp_yield = np.sum(durations * spec_cdf[np.arange(n_jobs), ma_best])
            exp_var   = np.sum(durations * scaled_v[np.arange(n_jobs), ma_best])

            yield_ok = exp_yield >= total_dur * Gamma
            var_ok   = exp_var <= total_dur * Lambda

            print(
                f"[Progress] Generation={generation} Time={int(time.time() - start_time)}s | "
                f"Best={best_fit:.3f} | "
                f"LateJobs={late_count} | "
                f"ExpectedYield={exp_yield/total_dur:.3f} ({'OK' if yield_ok else 'VIOL'}) | "
                f"Variance={exp_var/total_dur:.3f} ({'OK' if var_ok else 'VIOL'})"
            )

        generation += 1

        if time.time() - start_time > time_limit:
            break

    return best_ind, best_fit

# ============================================================
# Integration
# ============================================================
def run_ga_with_env(env: Environment, time_limit=10, Gamma=0.9, Lambda=0.2, log_viz=True):
    st = time.time()

    spec_cdf = np.array(env.spec_cdf)
    scaled_v = np.array(env.scaled_v)

    ga_jobs = [(int(env.duration[j]), int(env.job_to_family[j]), int(env.deadline[j]))
               for j in range(env.job_num)]

    (best_seq, best_ma), best_obj = run_ga(
        jobs=ga_jobs,
        n_machines=env.machine_num,
        setup_time=env.setup_time,
        time_limit=time_limit,
        spec_cdf=spec_cdf,
        scaled_v=scaled_v,
        Gamma=Gamma,
        Lambda=Lambda,
        log_viz=log_viz
    )

    schedules, setups, late = decode(ga_jobs, best_ma, best_seq, env.machine_num, env.setup_time)

    # Quality check
    durations = np.array([p for (p, f, d) in ga_jobs])
    total_dur = durations.sum()

    exp_yield = np.sum(durations * spec_cdf[np.arange(len(ga_jobs)), best_ma])
    exp_var = np.sum(durations * scaled_v[np.arange(len(ga_jobs)), best_ma])
    yield_ok = exp_yield >= total_dur * Gamma
    var_ok = exp_var <= total_dur * Lambda
    schedule_ok = result_explain(env, Lambda, schedule=schedules, solution=setups, stop_if_wrong=False)
    feasible_TF = schedule_ok and yield_ok and var_ok

    print("\n=== Final GA Result ===")
    print(f"Objective Value        = {best_obj:.3f}")
    print(f"Total Setups           = {setups}")
    print(f"Late Jobs              = {len(late)}")
    print(f"Yield Ratio            = {exp_yield/total_dur:.3f} (Target >= {Gamma}: {'OK' if yield_ok else 'VIOL'})")
    print(f"Variance Ratio         = {exp_var/total_dur:.3f} (Target <= {Lambda}: {'OK' if var_ok else 'VIOL'})")
    print(f"Schedule Check         = {feasible_TF}")

    return [setups, feasible_TF, round(time.time()-st, 3)]


# ----------------------------------------
# Execution block
# ----------------------------------------
if __name__ == "__main__":
    result = new_datarecoder()
    result.create_csv('ga_result2.csv', new_col_names=['Lambda', 'objective value', 'feasible_TF', 'total time'])
    
    for job_num in [100, 200]:
        for machine_num in [20, 25]:
            for seed in range(110, 210, 10):

                family_num = math.floor(job_num/8)+1 

                Gamma = 0.95
                Lambda = 1.15

                print(f"\nJob: {job_num}, Machine: {machine_num}, Family: {family_num}, Seed: {seed}")

                environment = Environment(job_num, machine_num, family_num, setup_time=10)
                env = environment.reset(seed=seed, tight='base', quality='high')

                row = run_ga_with_env(env, time_limit=60*60, Gamma=Gamma, Lambda=Lambda, log_viz=True)

                result.write_result_to_csv([job_num, machine_num, seed, Lambda] + row)
