import numpy as np
from itertools import permutations
from collections import Counter, defaultdict
from ortools.sat.python import cp_model

from Env import Env
from Assignment import Assignment


class Pattern:
    def __init__(self):
        """Initialize the Pattern object with an empty dictionary."""
        self.patterns = {}  # Stores patterns where keys are the first elements of new patterns

    def append(self, new_pattern):
        """
        Add a new pattern if its first element is not already present.

        Args:
            new_pattern (tuple): A tuple containing:
                - new_pattern[0]: A key representing the pattern category.
                - new_pattern[1]: First associated value.
                - new_pattern[2]: Second associated value.

        Returns:
            bool: Always returns True (pattern is either added or ignored).
        """        
        # Check if the first element of the new pattern is already stored
        if new_pattern[0] not in self.patterns:
            # If not, add the new pattern with its associated values
            self.patterns[new_pattern[0]] = (new_pattern[1], new_pattern[2])
            
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

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, viz):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.viz = viz
        self.__solution_count = 0
        self.first_solution_time = 0
        self.first_solution = 0
        self.obj_values = []
        self.lbd_values = []

    def on_solution_callback(self):
        if self.viz:
            print(
                "Solution %i, time = %f s, objective = %i, lower bound = %i"
                % (self.__solution_count, self.WallTime(), self.ObjectiveValue(), self.best_objective_bound)
            )
        if self.__solution_count == 0: 
            self.first_solution_time = round(self.WallTime(), 2)
            self.first_solution = self.ObjectiveValue()
        self.__solution_count += 1
        self.obj_values.append(self.ObjectiveValue())
        self.lbd_values.append(self.best_objective_bound)

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

def check_min_setup(env: Env, families, setup_time=1):
    """
    Check if there exists at least one ordering of job families that meets all deadlines.

    Args:
        env (Env): The environment containing job deadlines and durations.
        families (dict): A dictionary where keys are family labels, and values are lists of job indices.
                         Example: {'a': [0,1,2], 'b': [3,4], 'c': [5,6], ...}
        setup_time (int, optional): Waiting time when switching between job families. Default is 1.

    Returns:
        bool: True if at least one valid ordering exists where all jobs finish before their deadlines, 
              otherwise False.
    """
    # Pre-sort jobs within each family based on their deadlines
    sorted_families = {
        family: sorted(jobs, key=lambda job: env.deadline[job])
        for family, jobs in families.items()
    }

    # Check all possible orderings of job families
    for order in permutations(sorted_families.keys()):
        time = 0  # Track the total elapsed time
        valid = True  # Flag to check if the current order is feasible

        # Process jobs in the given order of families
        for family in order:
            for job in sorted_families[family]:
                time += env.duration[job]  # Accumulate job execution time
                if time > env.deadline[job]:  # Check if the job exceeds its deadline
                    valid = False  # Mark as invalid and break out of inner loop
                    break
            time += setup_time  # Add setup time when switching families

        if valid:
            return True  # If at least one valid order exists, return True

    return False  # No valid order found


class Sequence:
    def __init__(self, env, viz=False) -> None:
        self.env = env
        self.viz = viz
        self.count_dict = [{} for _ in range(env.machine_num)]
        self.job_to_family = env.job_to_family
        self.machine_family_num = [0 for _ in range(env.machine_num)]

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
    
    def branch(self, m):
        """
        Selects representative jobs from each family assigned to machine `m` based on 
        the earliest deadline and shortest duration within each family.

        Args:
            m (int): The machine index.

        Returns:
            list: A list of selected job indices. If some families do not have a job,
                the last element will be a tuple of the families that do have a job.
                If no jobs are found, returns [None].
        """
        # Select one job from each family with the earliest deadline and shortest duration
        cases = {
            fam: min(
                (j for j in self.machine_jobs[m] if self.job_to_family[j] == fam 
                and self.env.deadline[j] == min(self.env.deadline[x] for x in self.machine_jobs[m] if self.job_to_family[x] == fam) 
                and self.env.duration[j] == min(self.env.duration[x] for x in self.machine_jobs[m] if self.job_to_family[x] == fam)),
                key=lambda x: self.env.deadline[x],  # Select the job with the earliest deadline in case of a tie
                default=None  # If no job is found for a family, assign None
            ) for fam in self.count_dict[m].keys()
        }

        # Separate non-None values from None values
        non_none_values = [value for value in cases.values() if value is not None]

        if not non_none_values:  
            # Case 1: If there are no valid jobs at all, return [None]
            cases = [None]
        elif len(non_none_values) < len(self.count_dict[m]):  
            # Case 2: If at least one job is found, but not for all families
            # Store the valid jobs and append a tuple of the families that have a job
            non_none_keys = tuple(key for key, value in cases.items() if value is not None)
            cases = non_none_values + [non_none_keys]
        else:  
            # Case 3: If every family has at least one job, return the selected jobs
            cases = non_none_values

        return cases

    def cp_parallel_sequence(self, time_limit, framework_time_limit, framework_elapsed_time, pattern: dict):
        """
        Solves the job scheduling problem in parallel for multiple machines while considering pattern reuse.

        Args:
            time_limit (float): The base time limit for solving a single machine problem.
            framework_time_limit (float): The total time limit for the framework.
            framework_elapsed_time (float): The elapsed time in the framework.
            pattern (dict): A dictionary containing known scheduling patterns and their objective values.

        Returns:
            tuple: 
                - temp_pattern (list): Stores the computed scheduling patterns for each machine.
                - pattern_information (list): Information about patterns used for decision-making.
                - schedule (list): The generated schedules for each machine.
        """

        pattern_information = []
        if len(pattern) > 0:  # If there are existing patterns
            # Extract pattern information for each machine
            # Information format: [False, False, [index], [index, ub, lb], ...]
            pattern_information = pattern_to_infomation(self.env.machine_num, self.allocation, pattern)
            print(f"Pattern selection: {[False if p is False else True for p in pattern_information]}")

        # Compute the Earliest Due Date (EDD) heuristic
        self.EDD = self.use_EDD()
        for m in range(self.env.machine_num):
            self.count_dict[m] = Counter(tuple_item[-1] for tuple_item in self.EDD[m])
            self.machine_family_num[m] = len(self.count_dict[m].keys())
            print(f"Machine: {m}, Assigned jobs: {self.machine_jobs[m]}")
        print(f"Family number per machine: {self.machine_family_num}")

        # Main processing loop
        temp_pattern = [[] for _ in range(self.env.machine_num)]  # Stores patterns for each machine
        schedule = [[] for _ in range(self.env.machine_num)]  # Stores the computed schedules

        for m in range(self.env.machine_num):
            # Create a binary tuple representing job assignments for this machine
            assigned_job_tuple = tuple([1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num)])
            temp_pattern[m].append(assigned_job_tuple)  # First element in pattern

            # If a previously solved optimal pattern exists, reuse it
            if len(pattern_information) > 0 and pattern_information[m] is not False and len(pattern_information[m]) < 2:
                pi = pattern_information[m][0]
                ub, lb = pattern[pi][0], pattern[pi][1]
                print(f"Machine: {m}, Status: OPTIMAL, Objective value: {ub}, Lower bound: {lb}")
                temp_pattern[m].append(ub)
                temp_pattern[m].append(lb)
                continue

            else:
                ub_hint, lb_hint = None, None  # Upper and lower bound hints

                # Count jobs per family
                family_counter = defaultdict(list)
                for i, b in enumerate(assigned_job_tuple):
                    if b == 1:
                        family_counter[self.env.job_to_family[i]].append(i)
                
                fam_num = len(family_counter)  # Number of families on this machine

                # If there are multiple families, check if a minimum setup sequence exists
                if fam_num > 1:
                    if check_min_setup(self.env, family_counter):
                        print(f"Machine: {m}, Status: OPTIMAL, Objective value: {fam_num-1}, Lower bound: {fam_num-1}")
                        temp_pattern[m].append(fam_num - 1)
                        temp_pattern[m].append(fam_num - 1)
                        continue
                    else:
                        lb_hint = fam_num  # Provide a lower bound hint

                # If no previous solution is applicable, solve the problem
                found_unknown, found_solution = False, False

                # Generate branching cases for optimization
                if fam_num > 1:
                    cases = self.branch(m)
                else:
                    cases = [None]

                if len(cases) > 1 and len(self.count_dict[m]) > 1:
                    print(f"Machine: {m}, Now using {len(cases)} branches!")

                # Adjust the time limit based on framework constraints
                temp_time_limit = time_limit
                if len(pattern_information) > 0 and pattern_information[m] is not False:  # If pattern reuse is possible, allocate more time
                    temp_time_limit = max(
                        (framework_time_limit - framework_elapsed_time) / (len(cases) * (self.env.machine_num - m)), 
                        time_limit
                    ) # Try to provide as much time as possible
                    print("Check for additional time:", temp_time_limit, 
                        (framework_time_limit - framework_elapsed_time) / (len(cases) * (self.env.machine_num - m)))

                    # If the previous solution was feasible but not optimal, reuse its bounds
                    if len(pattern_information[m]) == 3:
                        ub_hint, lb_hint = pattern_information[m][1], pattern_information[m][2]

                # Solve the scheduling problem for this machine
                pre_result = len(self.machine_jobs[m])  # Initialize with the worst-case objective value
                for _, case in enumerate(cases):
                    result = self.cp_sequence_one_machine(m, temp_time_limit, ub_lb=(ub_hint, lb_hint), case=case)  

                    if result is None:
                        found_unknown = True  # Time limit exceeded
                        continue

                    if result is not False and result[0] < pre_result:  # Keep track of the best found solution
                        pre_result = result[0]
                        temp = result
                        found_solution = True
                        schedule[m] = self.temp_schedule[m]

                        if result[0] == self.machine_family_num[m] - 1:  # If we reach the minimum possible objective, stop early
                            break

                # Store the computed result in temp_pattern
                if found_solution:
                    ub, lb = temp  # A feasible solution was found
                    temp_pattern[m].append(ub)
                    temp_pattern[m].append(lb)  # Second element in pattern
                elif found_unknown:
                    schedule[m] = []  # Time limit exceeded, result is unknown
                    temp_pattern[m].append(None)
                else:
                    temp_pattern[m].append(False)  # No feasible solution found

        return temp_pattern, pattern_information, schedule

    def cp_sequence_one_machine(self, machine_index, time_limit, ub_lb=None, case=None):
        # parameters
        job_list = self.machine_jobs[machine_index]
        dummy_job_list = [-1] + job_list
        if len(job_list) == 0:
            return None
        duration = self.env.duration
        deadline = self.env.deadline
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

        # objective
        model.minimize(sum([setup[i, j] for i in job_list for j in job_list]))
        
        # optional ub, lb hint
        ub, lb = ub_lb
        if ub is not None: model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= ub)
        if lb is not None: model.add(sum([setup[i, j] for i in job_list for j in job_list]) >= lb)
        
        # deadline
        for j in job_list:
            assert duration[j] <= deadline[j]
            model.add(start_time[j] + duration[j] <= deadline[j])
 
        # Ensure exactly one latest start
        model.add(sum(relation[i, -1] for i in job_list) == 1)

        if case is not None and not isinstance(case, tuple):
            model.add(relation[-1, case] == 1) # 가장 먼저 시작하는 job 고정
        else:
            model.add(sum(relation[-1, i] for i in job_list) == 1) # Ensure exactly one earliest start
            if isinstance(case, tuple):
                temp_ban_job = [i for i in job_list if self.job_to_family[i] in case]
                model.add(sum(relation[-1, i] for i in temp_ban_job) == 0)

        # 각 행의 합이 1이 되도록 제약 조건 추가
        for i in job_list:
            model.add(sum(relation[i, j] for j in dummy_job_list) == 1) # 10, 11

        # 각 열의 합이 1이 되도록 제약 조건 추가
        for j in job_list:
            model.add(sum(relation[i, j] for i in dummy_job_list) == 1) # 7, 8

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
                model.add(start_time[i] + duration[i] + setup[i, j] <= start_time[j]).only_enforce_if(relation[i, j])

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
        solver.parameters.max_time_in_seconds = time_limit
        solution_printer = SolutionPrinter(self.viz)
        status = solver.solve(model, solution_printer) # solver가 해결하도록

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("machine:", machine_index, "status:", solver.status_name(status), "objective value:", solver.objective_value,  
            "lower bound:", solver.best_objective_bound, "time:", round(solver.WallTime(), 2))

            start_times = [(j, int(solver.Value(start_time[j])), int(solver.Value(start_time[j])) + int(duration[j]), int(deadline[j]), self.job_to_family[j]) for j in job_list]
            self.temp_schedule[machine_index] = sorted(start_times, key=lambda x: x[1])
            if status == cp_model.OPTIMAL:
                return int(solver.objective_value), int(solver.objective_value)
            return int(solver.objective_value), int(solver.best_objective_bound)
        else:
            print("machine:", machine_index, "status:", solver.status_name(status), "time:", round(solver.WallTime(), 2))
            if status == cp_model.INFEASIBLE:
                return False
            else:
                return None
      
def global_cp(env: Env, Gamma, Lambda, time_limit, viz=True):
    # parameters
    job_num = env.job_num
    machine_num = env.machine_num
    duration = env.duration
    deadline = env.deadline
    T = sum(duration[j] for j in range(job_num)) + job_num - 1 - min(duration)
    SPEC_CDF = env.spec_cdf
    SCALED_V = env.scaled_v

    model = cp_model.CpModel()

    # Variables
    relation = {}
    setup = {}
    start_time = {}
    allocation = {}

    for i in range(job_num):
        start_time[i] = model.NewIntVar(0, T, f'st_{i}')
        for j in range(job_num):
            relation[i, j] = model.NewBoolVar(f'rel_{i}_{j}')
            setup[i, j] = model.NewBoolVar(f'setup_{i}_{j}')
        for m in range(machine_num):
            allocation[i, m] = model.NewBoolVar(f'alloc_{i}_{m}')

    for i in range(job_num):
        relation[-1, i] = model.NewBoolVar(f'rel_{-1}_{i}')
        relation[i, -1] = model.NewBoolVar(f'rel_{i}_{-1}')
        
    # objective
    model.minimize(sum([setup[i, j] for i in range(job_num) for j in range(job_num)]))

    # deadline
    for j in range(job_num):
        assert duration[j] <= deadline[j]
        model.add(start_time[j] + duration[j] <= deadline[j]) # 5

    # allocation
    for i in range(job_num):
        model.add(sum(allocation[i, m] for m in range(machine_num)) == 1) # 1

    # Ensure exactly machine_num earliest start
    model.add(sum(relation[-1, i] for i in range(job_num)) == machine_num) # 17

    # Ensure exactly machine_num latest start
    model.add(sum(relation[i, -1] for i in range(job_num)) == machine_num) # 18

    # 각 행의 합이 1이 되도록 제약 조건 추가
    for i in range(job_num):
        model.add(sum(relation[i, j] for j in range(-1, job_num)) == 1) # 10, 11

    # 각 열의 합이 1이 되도록 제약 조건 추가
    for j in range(job_num):
        model.add(sum(relation[i, j] for i in range(-1, job_num)) == 1) # 7, 8

    # 대각 예외 제약 조건 추가
    for j in range(job_num):
        model.add(relation[j, j] == 0) # 12
        for i in range(job_num):
            if j < i:
                model.add(relation[i, j] + relation[j, i] <= 1) # 13

    # setup 정의
    for i in range(job_num):
        for j in range(job_num):
            if env.job_to_family[i] != env.job_to_family[j]:
                model.add(setup[i, j] >= relation[i, j]) # 15
            else:
                model.add(setup[i, j] == 0) # 16
            if i != j:
                for m in range(machine_num):
                    condition = model.NewBoolVar(f"temp_{i}_{j}_{m}")

                    model.add(condition <= (allocation[i, m] + allocation[j, m]))
                    model.add(condition <= 1 - (allocation[i, m] + allocation[j, m] - 1))
                    model.add(condition >= (allocation[i, m] - allocation[j, m]))
                    model.add(condition >= (allocation[j, m] - allocation[i, m]))

                    model.add(relation[i, j] == 0).only_enforce_if(condition) # 4
                    model.add(relation[j, i] == 0).only_enforce_if(condition) # 4        

    # predecessor-successor
    for i in range(job_num):  
        for j in range(job_num):
            model.add(start_time[i] + duration[i] + setup[i, j] <= start_time[j]).only_enforce_if(relation[i, j]) # 14

    # Constraint: Expected yield
    SCALING_FACTOR = 1000
    model.add(
        sum(
            duration[j] * sum(allocation[j, m] * int(SPEC_CDF[j][m] * SCALING_FACTOR) for m in range(machine_num))
            for j in range(job_num)
        ) >= int(sum(duration) * Gamma * SCALING_FACTOR),
    ) # 2

    # Constraint: Expected variance
    model.add(
        sum(
            duration[j] * sum(allocation[j, m] * int(SCALED_V[j][m] * SCALING_FACTOR) for m in range(machine_num))
            for j in range(job_num)
        ) <= int(sum(duration) * Lambda * SCALING_FACTOR),
    ) # 3

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(viz)
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model, solution_printer) # solver가 해결하도록

    print("status:", solver.status_name(status), "objective value:", solver.objective_value, 
"lower bound:", solver.best_objective_bound, "time:", round(solver.WallTime(), 2))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if status == cp_model.OPTIMAL:
            return solver.objective_value, solver.best_objective_bound, round(solver.WallTime(), 2), True, solution_printer.first_solution, solution_printer.first_solution_time
        else:
            return solver.objective_value, solver.best_objective_bound, round(solver.WallTime(), 2), -1, solution_printer.first_solution, solution_printer.first_solution_time
    elif status == cp_model.INFEASIBLE:
        return False, -1, round(solver.WallTime(), 2), -1, -1, -1
    else:
        print("time limit over")
        return solver.objective_value, solver.best_objective_bound, round(solver.WallTime(), 2), -1, solution_printer.first_solution, solution_printer.first_solution_time
        