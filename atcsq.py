import numpy as np
import math
import time

from Env import Environment
from utills import result_explain, new_datarecoder

class ATCSQ_Scheduler:
    def __init__(self, env: Environment, Gamma=0.9, Lambda=1.15, k1=1.0, k2=1.0, k3=1.0):
        self.env = env
        self.n_jobs = env.job_num
        self.n_machines = env.machine_num
        self.setup_time = env.setup_time
        
        # Target quality thresholds
        self.Gamma = Gamma
        self.Lambda = Lambda
        
        # Load problem data from environment
        self.durations = np.array(env.duration)
        self.deadlines = np.array(env.deadline)
        # spec_cdf represents per-job-per-machine expected yield
        self.spec_cdf = np.array(env.spec_cdf)
        # scaled_v represents per-job-per-machine risk/variance metric
        self.scaled_v = np.array(env.scaled_v)
        
        # Handle family mapping (job -> family)
        if isinstance(env.job_to_family, dict):
            # If mapping provided as dict, convert to array indexed by job id
            self.families = np.array([env.job_to_family[j] for j in range(self.n_jobs)])
        else:
            self.families = np.array(env.job_to_family)

        # Sensitivity parameters
        self.k1 = k1  # due-date sensitivity (slack scaling)
        self.k2 = k2  # setup sensitivity
        self.k3 = k3  # quality-violation sensitivity (penalty scale)
        
        # Scaling factors used in exponential terms to make terms unit-consistent
        self.avg_p = np.mean(self.durations)
        # avg_s is a heuristic average setup for normalization; currently set to half of setup_time
        self.avg_s = self.setup_time / 2.0 

    def calculate_priority_index(self, job_idx, machine_idx, current_time, last_fam):
        """
        Compute the ATCSQ priority index for a candidate job on a given machine.

        Formula (threshold-based):
            index = (1 / p) * exp(-slack / (k1 * avg_p)) * exp(-setup / (k2 * avg_s)) * exp(-violation_score / k3)

        Terms:
        - 1/p: favors short processing times
        - slack term: favors jobs with small slack (tight deadlines)
        - setup term: favors jobs that do not require family change
        - quality term: penalizes assignments that violate yield/risk thresholds
        
        Parameters:
            job_idx: index of the job being evaluated
            machine_idx: index of the machine considered
            current_time: current available time of the machine
            last_fam: family id of the last job processed on the machine (-1 if none)
        """
        p = self.durations[job_idx]
        d = self.deadlines[job_idx]
        
        # 1) Due date term (slack). Slack is how much time remains after processing.
        slack = max(d - p - current_time, 0)
        term_due = math.exp(-slack / (self.k1 * self.avg_p))
        
        # 2) Setup term. If family changes and there was a previous family, add setup penalty.
        fam = self.families[job_idx]
        setup = 0 if (last_fam == fam or last_fam == -1) else self.setup_time
        term_setup = math.exp(-setup / (self.k2 * self.avg_s))
        
        # 3) Quality term. Check whether assignment violates yield (Gamma) or risk (Lambda).
        #    spec_cdf and scaled_v are assumed indexed [job, machine].
        yield_val = self.spec_cdf[job_idx, machine_idx]
        risk_val = self.scaled_v[job_idx, machine_idx]
        
        violation_score = 0
        # Penalty if yield is below target
        if yield_val < self.Gamma:
            violation_score += 1
        # Penalty if risk exceeds allowed threshold
        if risk_val > self.Lambda:
            violation_score += 1
        
        # If no violations, this term is 1. Larger violation_score reduces the index.
        term_quality = math.exp(-violation_score / self.k3)
        
        # Final index: shorter processing time increases priority
        index = (1.0 / p) * term_due * term_setup * term_quality
        
        return index

    def solve(self):
        """
        Greedy scheduling loop using the ATCSQ index.
        At each step choose the machine that becomes available earliest, evaluate
        all unscheduled jobs for that machine, and assign the job with maximum index.

        Returns:
            schedules: list of per-machine schedules; each entry is a list of tuples
                       (job_id, start_time, end_time, deadline, family)
            total_setups: total number of family-change setups performed
        """
        machine_times = np.zeros(self.n_machines)
        machine_last_fam = np.full(self.n_machines, -1)
        
        unscheduled_jobs = set(range(self.n_jobs))
        schedules = [[] for _ in range(self.n_machines)]
        total_setups = 0
        
        while unscheduled_jobs:
            # Select the machine that gets free the earliest
            m_curr = np.argmin(machine_times)
            t_curr = machine_times[m_curr]
            last_fam = machine_last_fam[m_curr]
            
            best_job = -1
            max_index = -1.0
            
            # Evaluate all remaining jobs for this machine
            for job in unscheduled_jobs:
                idx = self.calculate_priority_index(job, m_curr, t_curr, last_fam)
                if idx > max_index:
                    max_index = idx
                    best_job = job
            
            # Assign the selected job (if any)
            if best_job != -1:
                fam = self.families[best_job]
                p = self.durations[best_job]
                d = self.deadlines[best_job]
                
                # Apply setup time if family changes
                if last_fam != -1 and last_fam != fam:
                    machine_times[m_curr] += self.setup_time
                    total_setups += 1
                
                start = machine_times[m_curr]
                end = start + p
                
                schedules[m_curr].append((best_job, start, end, d, fam))
                
                machine_times[m_curr] = end
                machine_last_fam[m_curr] = fam
                unscheduled_jobs.remove(best_job)
            else:
                # No feasible job found (should not typically happen) -> break to avoid infinite loop
                break
                
        return schedules, total_setups


# ----------------------------------------
# Example execution and basic result logging
# ----------------------------------------
if __name__ == "__main__":
    result = new_datarecoder()
    result.create_csv('atcsq_result2.csv', new_col_names=['Lambda', 'objective value', 'feasible_TF', 'total time'])

    for job_num in [100, 200]:
        for machine_num in [20, 25]:
            for seed in range(110, 210, 10):

                family_num = math.floor(job_num/8)+1 

                Gamma = 0.95
                Lambda = 1.15

                print(f"\nJob: {job_num}, Machine: {machine_num}, Family: {family_num}, Seed: {seed}")

                environment = Environment(job_num, machine_num, family_num, setup_time=10)
                env = environment.reset(seed=seed, tight='base', quality='high')

                start_time = time.time()
                scheduler = ATCSQ_Scheduler(env, Gamma=Gamma, Lambda=Lambda, k1=0.354, k2=0.069, k3=3.135)
                schedules, total_setups = scheduler.solve()
                total_time = time.time() - start_time

                # === Detailed analysis ===
                durations = np.array(env.duration)
                total_dur = durations.sum()
                
                # Compute cumulative yield and risk and count late jobs
                current_yield_sum = 0
                current_var_sum = 0
                late_jobs = []
                
                for m_idx, m_sched in enumerate(schedules):
                    for job_info in m_sched:
                        j_id, start, end, d, f = job_info
                        p = durations[j_id]
                        
                        # Accumulate quality metrics weighted by processing time
                        current_yield_sum += p * env.spec_cdf[j_id][m_idx]
                        current_var_sum += p * env.scaled_v[j_id][m_idx]
                        
                        # Check lateness
                        if end > d:
                            late_jobs.append(j_id)
                            
                # Check constraints
                exp_yield = current_yield_sum
                exp_var = current_var_sum
                
                yield_ok = exp_yield >= total_dur * Gamma
                var_ok = exp_var <= total_dur * Lambda
                
                # Validate schedule with environment helper
                schedule_ok = result_explain(env, Lambda, schedule=schedules, solution=total_setups, stop_if_wrong=False)
                feasible_TF = schedule_ok and yield_ok and var_ok

                # Final log
                print("\n=== Final ATCSQ Result ===")
                print(f"Total Setups           = {total_setups}")
                print(f"Late Jobs              = {len(late_jobs)}")
                print(f"Yield Ratio            = {exp_yield/total_dur:.3f} (Target >= {Gamma}: {'OK' if yield_ok else 'VIOL'})")
                print(f"Variance Ratio         = {exp_var/total_dur:.3f} (Target <= {Lambda}: {'OK' if var_ok else 'VIOL'})")
                print(f"Schedule Check         = {feasible_TF}")

                result.write_result_to_csv([job_num, machine_num, seed, Lambda, total_setups, feasible_TF, total_time])

