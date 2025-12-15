import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from ortools.sat.python import cp_model
import time

from Env import Env
from Assignment import Assignment, check_min_setup


class Pattern:
    def __init__(self):
        """Initialize the Pattern object with an empty dictionary."""
        self.patterns = {}  # Stores patterns where keys are the first elements of new patterns
        self.pattern_schedule = {}

    def append(self, new_pattern: list):
        """
        Add a new pattern if its first element is not already present.

        Args:
            new_pattern (tuple): A tuple containing:
                - new_pattern[0]: A key representing the pattern category.
                - new_pattern[1]: Upper bound.
                - new_pattern[2]: Lower bound.
                - new_pattern[3]: Schedule of the pattern

        Returns:
            bool: Always returns True (pattern is either added or ignored).
        """        
        # Check if the first element of the new pattern is already stored
        if new_pattern[0] not in self.patterns:
            # If not, add the new pattern with its associated values
            self.patterns[new_pattern[0]] = (new_pattern[1], new_pattern[2])
            self.pattern_schedule[new_pattern[0]] = new_pattern[3]

        return True  # Always return True regardless of addition

    def __len__(self):
        """Return the number of stored patterns."""
        return len(self.patterns)
    
class Guide:
    def __init__(self):
        """Initialize the Guide object with an empty dictionary to store patterns."""
        self.guides = {}  # Stores patterns as keys and their corresponding values

    def append(self, new_guide):
        """
        Add a new pattern if it is not already covered by an existing pattern.

        Args:
            new_guide (tuple): A tuple where:
                - new_guide[0] is a list representing a pattern (binary sequence).
                - new_guide[1] is an integer value associated with the pattern.

        Returns:
            bool: True if the new pattern was added, False if it was redundant.
        """
        # Step 1: Check if the new pattern is already covered by an existing one
        for guide in self.guides:
            if guide[1] >= new_guide[1] and (self.subset(new_guide[0], guide) or guide[0] == new_guide[0]):
                return False  # If redundant, do not add

        # Step 2: Identify existing patterns that should be removed
        delete_keys = []        
        for guide in self.guides:
            if self.subset(guide, new_guide[0]) and guide[1] <= new_guide[1]:  # If new_guide is a better version
                delete_keys.append(guide)
            elif guide[0] == new_guide[0] and guide[1] < new_guide[1]:  # If same pattern but with a better value
                delete_keys.append(guide)

        # Step 3: Remove outdated patterns
        for key in delete_keys:
            self.guides.pop(key)

        # Step 4: Add the new pattern
        self.guides[new_guide[0]] = new_guide[1]
        return True

    def subset(self, A, B):
        """
        Check if pattern A is a subset of pattern B (A <= B).

        Args:
            A (list): The smaller pattern.
            B (list): The larger pattern.

        Returns:
            bool: True if A is a subset of B, False otherwise.
        """
        if sum(B) >= sum(A):  # If B has more or equal 1s than A, it's not a subset
            return False
        
        # Ensure that every 1 in A also exists in B
        for i, e in enumerate(B):
            if e == 1 and A[i] == 0:
                return False

        return True
    
    def __len__(self):
        """Return the number of patterns stored."""
        return len(self.guides)

def pattern_to_infomation(machine_num, allocation, pattern: dict):
    """
    Convert pattern data into structured information for each machine.

    Args:
        machine_num (int): Number of machines.
        allocation (np.ndarray): Allocation matrix where each column represents jobs assigned to a machine.
        pattern (dict): Dictionary containing job patterns with:
            - Keys (tuple): Job allocation pattern.
            - Values (tuple): (upper bound, lower bound) or (None, None) if unknown.

    Returns:
        list: A list where each index corresponds to a machine and stores:
            - False: If no matching pattern exists.
            - [tuple]: If the pattern is solved optimally (upper bound == lower bound).
            - [tuple, ub, lb]: If the pattern is feasible but not optimal.
            - [tuple, None]: If the pattern is known but still unresolved.
    """
    # Initialize pattern information for each machine with default value 'False'
    pattern_information = [False for _ in range(machine_num)]  

    # Iterate through each machine
    for m in range(machine_num):
        for pi, p in pattern.items():  # Iterate over stored patterns
            pi = np.array(pi)  # Convert pattern key to numpy array

            # Check if the machine's allocation exactly matches the pattern
            if (allocation[:, m] == pi).all():
                if p[0] is not None:  # If the pattern has a known solution
                    if p[0] == p[1]:  # If upper bound == lower bound (solved optimally)
                        pattern_information[m] = [tuple(pi)]  # Store the pattern only
                        break
                    else:  # If feasible but not optimal, store pattern with ub and lb
                        pattern_information[m] = [tuple(pi), p[0], p[1]]
                        break
                else:  # If the pattern is known but still unresolved (unknown bounds)
                    pattern_information[m] = [tuple(pi), None]

    return pattern_information



class Sequence:
    def __init__(self, env: Env, fix=None) -> None:
        self.env = env
        self.viz = False
        self.count_dict = [{} for _ in range(env.machine_num)]
        self.job_to_family = env.job_to_family
        self.machine_family_num = [0 for _ in range(env.machine_num)]
        self.fix = fix # machine 별 리스트인데 각 리스트 안에는 튜플 혹은 값

    def reset(self, env, assignment):
        # Reset for re-assignment
        self.allocation = assignment

        self.machine_jobs = [[] for _ in range(env.machine_num)]
        for machine_index in range(env.machine_num):
            for job_index in range(env.job_num):
                if assignment[job_index, machine_index]:
                    self.machine_jobs[machine_index].append(job_index)

        self.temp_schedule = [[] for _ in range(env.machine_num)]

    def use_EDD(self):
        alloc = Assignment(self.env)
        alloc.EDD(self.allocation)
        return alloc.schedule
    
    def dominant(self, m):
        """
        jobs_dict: dict
            key: family_id
            value: list of job IDs belonging to that family

        return: list of tuples (j, k)
            조건을 만족하는 모든 (j, k) 쌍을 반환
        """
        relations = []
        for j in self.machine_jobs[m]:
            for k in self.machine_jobs[m]:
                if j == k:
                    continue

                if self.job_to_family[j] == self.job_to_family[k]:

                    dj, dk = self.env.deadline[j], self.env.deadline[k]
                    pj, pk = self.env.duration[j], self.env.duration[k]

                    # 기본 조건: deadline[j] ≤ deadline[k] and duration[j] ≤ duration[k]
                    if dj <= dk and pj <= pk:
                        # tie-breaker: 둘 다 같을 땐 인덱스 작은 쪽만 (j < k) 저장
                        if dj == dk and pj == pk and j > k:
                            continue
                        relations.append((j, k))
        return relations

    def cp_parallel_sequence(self, pattern: Pattern, time_limit: int, framework_remaining_time, parallel:bool=True, generation:bool=True):
            """
            Solves the job scheduling problem in parallel for multiple machines while considering pattern reuse.

            Args:
                time_limit (float): The base time limit for solving a single machine problem.
                framework_remaining_time (float): The total time limit remaining for the framework.
                pattern (dict): A dictionary containing known scheduling patterns and their objective values.

            Returns:
                tuple:
                    - temp_pattern (list): Computed scheduling patterns for each machine.
                    - pattern_information (list): Information about patterns used for decision-making.
                    - schedule (list): The generated schedules for each machine.
            """
            start_time = time.time()
            n_machines = self.env.machine_num
            patterns = pattern.patterns

            # Prepare pattern information
            pattern_information = []
            if patterns:
                pattern_information = pattern_to_infomation(n_machines, self.allocation, patterns)
            
            # Log optimal pattern selection
            if pattern_information:
                selection = []
                for p in pattern_information:
                    if p is False:
                        selection.append(False)
                    elif len(p) == 1:
                        selection.append(True)
                    elif len(p) == 2:
                        selection.append('Unknown')
                    else:
                        selection.append('Feasible')
                print(f"Optimal pattern selection: {selection}")
            else:
                print(f"Optimal pattern selection: {[False for _ in range(n_machines)]}")

            # EDD heuristic and job-family counts
            self.EDD = self.use_EDD()
            for m in range(n_machines):
                self.count_dict[m] = Counter(job[-1] for job in self.EDD[m])
                self.machine_family_num[m] = len(self.count_dict[m])
                if self.viz:
                    print(f"Machine: {m}, Assigned jobs: {self.machine_jobs[m]}")
            print(f"Family number per machine: {self.machine_family_num}")

            temp_pattern = [[] for _ in range(n_machines)]
            schedule = [[] for _ in range(n_machines)]
            lower_bounds = [0 for _ in range(n_machines)] 

            # Helper to decide reuse cases
            def should_skip(m):
                temp_schedule = []
                info = pattern_information[m] if pattern_information else False
                if info and len(info) == 1: # reuse optimal solution
                    return True, patterns[info[0]][0], pattern.pattern_schedule[info[0]] 
                
                # prepare assignment tuple
                if generation:
                    assigned = tuple(1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num))
                    # family-based lower bound
                    family_counter = defaultdict(list)
                    for i, b in enumerate(assigned):
                        if b:
                            family_counter[self.env.job_to_family[i]].append(i)
                    fam_num = len(family_counter)
                    # trivial min-setup case
                    if fam_num > 1:
                        if check_min_setup(self.env, family_counter, self.env.setup_time, temp_schedule):
                            return True, fam_num - 1, temp_schedule
                        else:
                            lower_bounds[m] = fam_num
        
                return False, None, None

            # Worker function for parallel solving
            def solve_machine(m):
                # set up hints and fix
                ub_hint = None
                lb_hint = lower_bounds[m]
                info = pattern_information[m] if pattern_information else False
                if info and len(info) == 3: # feasible but not optimal
                    lb_hint, ub_hint = info[2], info[1]
                
                # solve with CP solver
                if info is not False and len(info) > 1:
                    temp_time_limit = framework_remaining_time - (time.time() - start_time)
                    print(f"Machine: {m} has not opimally solved pattern, allocating time {round(temp_time_limit)}s")
                else:
                    temp_time_limit = time_limit
                
                fix = [f[m] for f in self.fix] if self.fix is not None else None

                prec = self.dominant(m)
                result = self.cp_sequence_one_machine(
                    m, temp_time_limit, ub_lb=(ub_hint, lb_hint), prec=prec, fix=fix, first_only=False
                )
                return m, result

            # Identify machines that need solving
            to_solve = []
            for m in range(n_machines):
                skip, ub, temp_schedule = should_skip(m)
                if skip:
                    temp_pattern[m] = [tuple([1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num)]), ub, ub, temp_schedule]
                    schedule[m] = temp_schedule
                else:
                    temp_pattern[m].append(tuple([1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num)]))
                    to_solve.append(m)

            # Parallel solve
            if to_solve:
                if parallel:
                    with ThreadPoolExecutor(max_workers=len(to_solve)) as executor:
                        futures = {executor.submit(solve_machine, m): m for m in to_solve}
                        for future in as_completed(futures):
                            m, result = future.result()
                            if result is None:
                                temp_pattern[m].append(None)
                                schedule[m] = []
                            elif result is False:
                                temp_pattern[m].append(False)
                                schedule[m] = []
                            else:
                                ub, lb = result
                                temp_pattern[m].append(ub)
                                temp_pattern[m].append(lb)
                                temp_pattern[m].append(self.temp_schedule[m])
                                schedule[m] = self.temp_schedule[m]
                else:
                    futures = [solve_machine(m) for m in to_solve]
                    for future in futures:
                        m, result = future
                        if result is None:
                            temp_pattern[m].append(None)
                            schedule[m] = []
                        elif result is False:
                            temp_pattern[m].append(False)
                            schedule[m] = []
                        else:
                            ub, lb = result
                            temp_pattern[m].append(ub)
                            temp_pattern[m].append(lb)
                            temp_pattern[m].append(self.temp_schedule[m])
                            schedule[m] = self.temp_schedule[m]

            return temp_pattern, pattern_information, schedule

    def cp_sequence_one_machine(self, machine_index, time_limit, ub_lb=None, prec=None, fix=None, first_only=False):
        # parameters
        job_list = self.machine_jobs[machine_index]
        dummy_job_list = [-1] + job_list
        if len(job_list) == 0:
            return None
        duration = self.env.duration
        deadline = self.env.deadline
        setup_time = self.env.setup_time
        T = sum(duration[j] for j in job_list) + len(job_list) - 1

        model = cp_model.CpModel()

        # Variables
        relation = {}
        setup = {}
        start_time = {}

        for i in job_list:
            start_time[i] = model.NewIntVar(0, T, f'st_{i}')
            for j in job_list:
                relation[i, j] = model.NewBoolVar(f'rel_{i}_{j}')
                setup[i, j] = model.NewBoolVar(f'setup_{i}_{j}')

        for i in job_list:
            relation[i, -1] = model.NewBoolVar(f'rel_{i}_{-1}')
            relation[-1, i] = model.NewBoolVar(f'rel_{-1}_{i}')

        if fix is not None:
            fix_relation, fix_setup, fix_st = fix
            for j, st in fix_st.items():
                model.add(start_time[j] == st)
            for j, k in fix_relation:
                model.add(relation[j, k] == True)
            for j, k in fix_setup:
                model.add(setup[j, k] == True)   

        # objective
        # model.minimize(T * len(job_list) * sum([setup[i, j] for i in job_list for j in job_list]) + sum([start_time[i] for i in job_list]))
        model.minimize(sum([setup[i, j] for i in job_list for j in job_list]))
        
        # optional ub, lb hint
        ub, lb = ub_lb
        if ub is not None: model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= ub)
        if lb is not None: model.add(sum([setup[i, j] for i in job_list for j in job_list]) >= lb)
        
        # deadline
        for j in job_list:
            assert duration[j] <= deadline[j]
            model.add(start_time[j] + duration[j] <= deadline[j])

        # Domninant property
        if prec is not None: 
            for j, k in prec:
                model.add(start_time[j] + duration[j] <= start_time[k])
                model.add(relation[k, j] == 0)

        # 각 행의 합이 1이 되도록 제약 조건 추가
        for i in job_list:
            model.add(sum(relation[i, j] for j in dummy_job_list) == 1) # 10, 11

        # 각 열의 합이 1이 되도록 제약 조건 추가
        for j in job_list:
            model.add(sum(relation[i, j] for i in dummy_job_list) == 1) # 7, 8

        # Ensure exactly one earliest & latest start
        model.add(sum(relation[-1, i] for i in job_list) == 1)
        model.add(sum(relation[i, -1] for i in job_list) == 1)

        # 대각 예외 제약 조건 추가
        for j in job_list:
            model.add(relation[j, j] == 0)
            for i in job_list:
                if j < i:
                    model.add(relation[i, j] + relation[j, i] <= 1)

        # setup 정의
        for i in job_list:
            for j in job_list:
                if self.job_to_family[i] != self.job_to_family[j]:
                    model.add(setup[i, j] >= relation[i, j])
                else:
                    model.add(setup[i, j] == 0)

        # predecessor-successor
        for i in job_list:
            for j in job_list:
                model.add(start_time[i] + duration[i] + setup_time * setup[i, j] <= start_time[j]).only_enforce_if(relation[i, j])

        # EDD를 힌트로 사용, 더 많은 변수에 값 추가
        if getattr(self, "EDD"): 
            setup_bound = 0
            previous_job = None
            previous_fam = None
            for job, eddst, dur, fam in self.EDD[machine_index]:
                model.add_hint(start_time[job], eddst)
                if previous_job is not None:
                    model.add_hint(relation[previous_job, job], True)
                    if fam != previous_fam:
                        model.add_hint(setup[previous_job, job], True)
                        setup_bound += 1
                previous_job = job
                previous_fam = fam

            model.add_hint(relation[-1, self.EDD[machine_index][0][0]], True) # for earliest start time
            model.add_hint(relation[self.EDD[machine_index][-1][0], -1], True) # for latest due date

            # LB, UB
            model.add(sum([setup[i, j] for i in job_list for j in job_list]) >= len(self.count_dict[machine_index].keys()) - 1) # trivial lower bound

            model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= setup_bound) # EDD upper bound
            # value_list = sorted(count_dict.values(), reverse=True)
            # total_sum = sum(value_list)
            # largest = value_list[0]
            # if len(value_list) > 1: second_largest = value_list[1]
            # else: second_largest = 0
            # model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= total_sum - largest + second_largest) # trivial upper bound

        solver = cp_model.CpSolver()
        if first_only:
            solver.parameters.stop_after_first_solution = True
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.solve(model) # solver가 해결하도록

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if self.viz:
                print("Machine:", machine_index, "status:", solver.status_name(status), "objective value:", solver.objective_value,  
                "lower bound:", solver.best_objective_bound, "time:", round(solver.WallTime(), 2))
            # job index, start time, completion time, deadline, family
            schedules = [(j, int(solver.Value(start_time[j])), int(solver.Value(start_time[j])) + int(duration[j]), int(deadline[j]), self.job_to_family[j]) for j in job_list]
            self.temp_schedule[machine_index] = sorted(schedules, key=lambda x: x[1])
            if status == cp_model.OPTIMAL:
                return int(solver.objective_value), int(solver.objective_value)
            return int(solver.objective_value), int(solver.best_objective_bound)
            # if status == cp_model.OPTIMAL:
            #     answer = sum([int(solver.Value(setup[i, j])) for i in job_list for j in job_list])
            #     return answer, answer
            # return answer, answer
        else:
            if self.viz:
                print("Machine:", machine_index, "status:", solver.status_name(status), "time:", round(solver.WallTime(), 2))
            if status == cp_model.INFEASIBLE:
                return False
            else:
                return None
