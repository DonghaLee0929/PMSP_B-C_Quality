import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import List, Dict
from gurobipy import Model, GRB, quicksum

from Env import Env

GRB_status = {2: 'OPTIMAL', 3: 'INFEASIBLE', 5: 'UNBOUNDED', 9: 'TIME_LIMIT'}

class Ban:
    def __init__(self):
        """Initialize an empty dictionary to store banned job configurations."""
        self.bans: Dict[int, int] = {}  # Key: Bitmask (int), Value: Count of 1s

    @staticmethod
    def to_bitmask(job: List[int]) -> int:
        """Convert a job (list of 0s and 1s) to a bitmask (integer).
        
        Example:
        job = [1, 0, 1] -> bitmask = 5 (binary: 101)
        """
        bitmask = 0
        for bit in job:
            bitmask = (bitmask << 1) | bit
        return bitmask

    @staticmethod
    def count_ones(bitmask: int) -> int:
        """Count the number of 1s in the bitmask."""
        return bin(bitmask).count('1')
    
    @staticmethod
    def to_list(bitmask: int, length: int = 60) -> List[int]:
        """Convert a bitmask (integer) back to a list of 0s and 1s.
        
        Args:
            bitmask: The integer representation of a binary sequence.
            length: The expected length of the output list (default: 60).

        Returns:
            A list of 0s and 1s representing the bitmask.
        """
        return [1 if bitmask & (1 << i) else 0 for i in range(length - 1, -1, -1)]

    def append(self, job: List[int]):
        """Add a new job configuration while ensuring no redundant entries.

        The method:
        - Converts the job list into a bitmask.
        - Removes any existing configurations that are subsets of the new one.
        - Avoids adding new configurations that are already covered by existing ones.
        """
        new_bitmask = self.to_bitmask(job)
        new_count = self.count_ones(new_bitmask)

        to_remove = []

        for existing_bitmask, existing_count in self.bans.items():
            # Check if the new job includes an existing job
            if (new_bitmask & existing_bitmask) == existing_bitmask and new_count <= existing_count:
                to_remove.append(existing_bitmask)
            # Check if the existing job includes the new job
            elif (existing_bitmask & new_bitmask) == new_bitmask and existing_count <= new_count:
                return  # The new job is already covered; do not add it

        # Remove included jobs
        for bitmask in to_remove:
            del self.bans[bitmask]

        # Add the new job configuration
        self.bans[new_bitmask] = new_count
    
    def __len__(self):
        """Return the number of banned job configurations stored."""
        return len(self.bans)

def check_min_setup(env: Env, families, setup_time=10, schedule:list=None):
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
            if schedule is not None:
                time = 0
                for family in order:
                    for job in sorted_families[family]:
                        schedule.append((job, time, time + env.duration[job], env.deadline[job], env.job_to_family[job]))
                        time += env.duration[job]
                    time += setup_time
            return True  # If at least one valid order exists, return True

    return False  # No valid order found


class Assignment:
    def __init__(self, env: Env, fix=None, viz=False, edd_additional=True, edd_with_setup=True) -> None:
        self.env = env
        self.viz = viz
        self.ban_list = None
        self.job_to_family = env.job_to_family
        self.fix = fix # list of tuple (machine, job)
        self.edd_additional = edd_additional
        self.edd_with_setup = edd_with_setup

    def insert_initial_ban(self):
        """Initialize the ban list by identifying infeasible job sequences.

        The function processes each job family separately and determines 
        which jobs violate their deadlines when scheduled sequentially.

        Process:
        - Jobs within each family are sorted by their deadlines.
        - Jobs are scheduled one by one, keeping track of the cumulative processing time.
        - If a job exceeds its deadline, all jobs scheduled up to that point 
        are added as a banned configuration.

        Returns:
            self.ban_list: A list of banned job sequences, where each entry is a binary list 
                        representing jobs that cannot be scheduled together.
        """
        for f in range(self.env.family_num):      
            # Step 1: Sort jobs within the family based on their deadlines (earliest first)
            sorted_jobs = sorted(self.env.family[f], key=lambda x: self.env.deadline[x])
            
            current_time = 0  # Tracks cumulative processing time

            for i, job in enumerate(sorted_jobs):
                end_time = current_time + self.env.duration[job]  # Calculate job completion time
                
                if end_time > self.env.deadline[job]:
                    # Step 2: If job violates its deadline, add all jobs up to this point to ban_list
                    self.ban_list.append([1 if j in sorted_jobs[:i+1] else 0 for j in range(self.env.job_num)])
                    break  # Stop checking further jobs in this family
                    
                current_time = end_time  # Update current time with completed job's end time
        
        return self.ban_list

    def EDD(self, assignment=None, additional=None, with_setup=None):
        """Earliest Due Date (EDD) Scheduling with optional setup consideration.

        This method assigns jobs to machines based on their deadlines using the EDD rule. 
        Jobs with earlier deadlines are scheduled first. If a job cannot be completed before
        its deadline, the corresponding job sequence is added to the ban list.

        Args:
            assignment (numpy.ndarray, optional): A job-machine assignment matrix where each row 
                                                represents a job and each column represents a machine.
                                                Defaults to self.assignment.
            additional (bool, optional): If True, additional infeasible job sequences are added to the ban list.
            with_setup (bool, optional): If True, considers setup times when switching between job families.

        Returns:
            bool: True if all scheduled jobs meet their deadlines, False otherwise.
        """
        additional = self.edd_additional if additional is None else additional
        with_setup = self.edd_with_setup if with_setup is None else with_setup
        assignment = assignment if assignment is not None else self.assignment

        result = np.ones(self.env.machine_num, dtype=bool)  # Stores feasibility of each machine's schedule
        self.schedule = [[] for _ in range(self.env.machine_num)]  # Stores the final schedule for each machine
        self.setup = [0 for _ in range(self.env.machine_num)]  # Tracks the number of setup operations per machine
        
        # Step 1: Iterate over each machine
        for m in range(self.env.machine_num):
            # Extract job indices assigned to this machine
            jobs_assigned_to_machine = np.where(assignment[:, m])[0]

            if with_setup:
                family_duration_sums = defaultdict(int)

                # Step 2: Sum up durations per job family
                for job in jobs_assigned_to_machine:
                    family = self.job_to_family[job]
                    duration = self.env.duration[job]
                    family_duration_sums[family] += duration

                # Determine max family duration (T) for setup considerations
                if len(family_duration_sums) <= 1:
                    T = -1  # No setup needed if there's only one family
                else:
                    T = max(family_duration_sums.values())

            # Step 3: Sort jobs by their deadline (earliest first)
            sorted_jobs = jobs_assigned_to_machine[np.argsort(self.env.deadline[jobs_assigned_to_machine])]

            if len(sorted_jobs) == 0:
                continue  # Skip empty machine assignments

            count = False  # Used to track if an infeasible job has been encountered
            current_time = 0  # Tracks cumulative processing time
            machine_results = []  # Stores feasibility of each job on this machine
            pre_job = sorted_jobs[0]  # Keeps track of the previous job for setup considerations

            # Step 4: Process jobs in deadline order
            for i, job in enumerate(sorted_jobs):
                # Check if a setup is needed when switching job families
                if pre_job != job and self.job_to_family[pre_job] != self.job_to_family[job]:
                    end_time = current_time + self.env.duration[job]  # Setup occurs
                    self.setup[m] += 1  # Increase setup count
                else:
                    end_time = current_time + self.env.duration[job]  # Normal processing time
                
                # Check if the job meets its deadline
                meets_deadline = end_time <= self.env.deadline[job]

                # Adjust deadline check if setup times (T) are considered
                if with_setup and T > -1 and current_time >= T:
                    meets_deadline = (end_time + self.env.setup_time) <= self.env.deadline[job]

                # If the job cannot be scheduled within its deadline
                if not meets_deadline and not count:
                    # Add current job sequence up to this point to ban list
                    self.ban_list.append([1 if j in sorted_jobs[:i+1] else 0 for j in range(self.env.job_num)])
                    # print([self.env.deadline[j] for j in sorted_jobs[:i+1]], [self.env.job_to_family[j] for j in sorted_jobs[:i+1]])

                    # Step 5: Additional infeasibility checks (if enabled)
                    if additional:
                        sorted_set = set(sorted_jobs[:i])  # Convert list to set for faster lookup
                        for j in range(self.env.job_num):
                            if j not in sorted_set and j != job: 
                                # if self.job_to_family[job] == self.job_to_family[j] and self.env.deadline[job] <= self.env.deadline[j]: # job를 j로 교체하고자함
                                if self.env.deadline[job] <= self.env.deadline[j]: # job를 j로 교체하고자함
                                    # Check if job j violates deadline when added
                                    if self.env.deadline[j] < current_time + self.env.duration[j]:
                                        job_list = [1 if jo in sorted_set else 0 for jo in range(self.env.job_num)]
                                        job_list[j] = 1 
                                        self.ban_list.append(job_list)
                                # elif self.env.deadline[j] < current_time + self.env.duration[j] + 1:
                                #     self.ban_list.append([1 if jo in sorted_jobs[:i] or jo == j else 0 for jo in range(self.env.job_num)])

                    count = True  # Mark that a deadline violation has been encountered
                
                # Store each machine's result
                machine_results.append(meets_deadline)

                # Store job execution details (job_id, start_time, duration, family_id)
                self.schedule[m].append((int(job), current_time, int(self.env.duration[job]), int(self.job_to_family[job])))

                # Update previous job and current time
                pre_job = job
                current_time = end_time  

            # Step 6: Sort scheduled jobs by start time
            self.schedule[m] = sorted(self.schedule[m], key=lambda x: x[1])

            # Check if all jobs on this machine met their deadlines
            result[m] = all(machine_results)

        return all(result)  # Return True if all machines have feasible schedules
    
    def EDD_for_pattern(self, jobs_assigned_to_machine, additional=None, with_setup=None):
        additional = self.edd_additional if additional is None else additional
        with_setup = self.edd_with_setup if with_setup is None else with_setup
        if with_setup:
            family_duration_sums = defaultdict(int)

            # Step 2: Sum up durations per job family
            for job in jobs_assigned_to_machine:
                family = self.job_to_family[job]
                duration = self.env.duration[job]
                family_duration_sums[family] += duration

            # Determine max family duration (T) for setup considerations
            if len(family_duration_sums) <= 1:
                T = -1  # No setup needed if there's only one family
            else:
                T = max(family_duration_sums.values())

        # Step 3: Sort jobs by their deadline (earliest first)
        sorted_jobs = jobs_assigned_to_machine[np.argsort(self.env.deadline[jobs_assigned_to_machine])]

        if len(sorted_jobs) == 0:
            return True

        count = False  # Used to track if an infeasible job has been encountered
        current_time = 0  # Tracks cumulative processing time
        machine_results = []  # Stores feasibility of each job on this machine
        pre_job = sorted_jobs[0]  # Keeps track of the previous job for setup considerations

        # Step 4: Process jobs in deadline order
        for i, job in enumerate(sorted_jobs):
            # Check if a setup is needed when switching job families
            if pre_job != job and self.job_to_family[pre_job] != self.job_to_family[job]:
                end_time = current_time + self.env.duration[job]  # Setup occurs
            else:
                end_time = current_time + self.env.duration[job]  # Normal processing time
            
            # Check if the job meets its deadline
            meets_deadline = end_time <= self.env.deadline[job]

            # Adjust deadline check if setup times (T) are considered
            if with_setup and T > -1 and current_time >= T:
                meets_deadline = (end_time + self.env.setup_time) <= self.env.deadline[job]

            # If the job cannot be scheduled within its deadline
            if not meets_deadline and not count:
                # Add current job sequence up to this point to ban list
                self.ban_list.append([1 if j in sorted_jobs[:i+1] else 0 for j in range(self.env.job_num)])
                # print([self.env.deadline[j] for j in sorted_jobs[:i+1]], [self.env.job_to_family[j] for j in sorted_jobs[:i+1]])

                # Step 5: Additional infeasibility checks (if enabled)
                if additional:
                    sorted_set = set(sorted_jobs[:i])  # Convert list to set for faster lookup
                    for j in range(self.env.job_num):
                        if j not in sorted_set and j != job: 
                            # if self.job_to_family[job] == self.job_to_family[j] and self.env.deadline[job] <= self.env.deadline[j]: # job를 j로 교체하고자함
                            if self.env.deadline[job] <= self.env.deadline[j]: # job를 j로 교체하고자함
                                # Check if job j violates deadline when added
                                if self.env.deadline[j] < current_time + self.env.duration[j]:
                                    job_list = [1 if jo in sorted_set else 0 for jo in range(self.env.job_num)]
                                    job_list[j] = 1 
                                    self.ban_list.append(job_list)
                            # elif self.env.deadline[j] < current_time + self.env.duration[j] + 1:
                            #     self.ban_list.append([1 if jo in sorted_jobs[:i] or jo == j else 0 for jo in range(self.env.job_num)])

                count = True  # Mark that a deadline violation has been encountered
            
            # Store each machine's result
            machine_results.append(meets_deadline)

            # Update previous job and current time
            pre_job = job
            current_time = end_time  
            
        return all(machine_results)  # Return True if all machines have feasible schedules

    def gurobi_assignment_min_setup(self, Gamma:float, Lambda:float, time_limit:int, pattern:dict={}, guide:dict={}, lb: int=None, ub: int=None, lazy: bool=True):
        # parameters
        job_num = self.env.job_num
        machine_num = self.env.machine_num
        family_num = self.env.family_num
        family = self.env.family
        duration = self.env.duration
        deadline = self.env.deadline
        SPEC_CDF = self.env.spec_cdf
        SCALED_V = self.env.scaled_v

        model = Model("assignment")
        model.setParam('OutputFlag', self.viz)
        model.setParam("TimeLimit", time_limit)

        # Variables
        assignments = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="assignments")
        if self.fix is not None:
            for m, j in self.fix:
                model.addConstr(assignments[j, m] == 1, name=f"fix_{m}{j}")
        s_min = model.addVars(machine_num, lb=0, ub=job_num, vtype=GRB.CONTINUOUS, name="s_min")
        machine_family = model.addVars(machine_num, family_num, vtype=GRB.BINARY, name="machine_check_family")
        
        # Objective: Minimize
        model.setObjective(quicksum(s_min[m] for m in range(machine_num)), GRB.MINIMIZE)

        # Optional Bound Constraint
        if lb is not None: model.addConstr(quicksum(s_min[m] for m in range(machine_num)) >= lb, name="lb")
        if ub is not None: model.addConstr(quicksum(s_min[m] for m in range(machine_num)) <= ub, name="ub")

        # Constraint: Job uniqueness & assignment
        for j in range(job_num):
            model.addConstr(
                quicksum(assignments[j, m] for m in range(machine_num)) == 1,
                name=f"job_uniqueness_{j}"
            ) # 6

        # Constraint: machine_family_relation
        for m in range(machine_num):
            for f in range(family_num):
                model.addGenConstrIndicator(machine_family[m, f], True, quicksum(assignments[j, m] for j in family[f]) >= 1, name=f"machine_family_exist_{m}{f}") # 6
                model.addGenConstrIndicator(machine_family[m, f], False, quicksum(assignments[j, m] for j in family[f]) <= 0, name=f"machine_family_not_exist_{m}{f}") # 7

        # Valid Inequality: too many jobs in a machine
        for m in range(machine_num):
            model.addConstr(quicksum(assignments[j, m] * duration[j] for j in range(job_num)) + 
                            self.env.setup_time * (quicksum(machine_family[m, f] for f in range(family_num)) - 1) <= max(deadline), name=f"machine_job_limit_{m}")
            for f in range(family_num):
                model.addConstr(quicksum(assignments[j, m] * duration[j] for j in family[f]) <= max([deadline[j] for j in family[f]]), name=f"machine_family_job_limit_{m}")

        # Optional Column and Constraint: Optimal guide
        pattern_num = len(pattern)
        if pattern_num > 0:
            job_assignments = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="job_assignments")
            PATTERN_INFORM, PATTERN_SETUP = zip(*[(pi, p[0]) for pi, p in pattern.items()]) # setup은 optimal한 경우만
            patterns = model.addVars(machine_num, pattern_num, vtype=GRB.BINARY, name="patterns")
            patterns_machine = model.addVars(machine_num, vtype=GRB.BINARY, name="patterns_machine")
        
            # Constraint: pattern_job relation
            for m in range(machine_num):
                model.addConstr(patterns_machine[m] == quicksum(patterns[m, p] for p in range(pattern_num)), name=f"pattern_machine_{m}") # 1
                model.addGenConstrIndicator(patterns_machine[m], True, quicksum(job_assignments[j, m] for j in range(job_num)) <= 0, name=f"pattern_job_exist_{m}") # 2
                model.addGenConstrIndicator(patterns_machine[m], False, quicksum(job_assignments[j, m] for j in range(job_num)) >= 1, name=f"pattern_job_not_exist_{m}") # 3

            # Constraint: pattern_selection
            for p in range(pattern_num):
                model.addConstr(quicksum(patterns[m, p] for m in range(machine_num)) <= 1, name=f"pattern_selection_{p}") # 4

            for j in range(job_num):
                for m in range(machine_num):
                    model.addConstr(assignments[j, m] == quicksum(PATTERN_INFORM[p][j] * patterns[m, p] for p in range(pattern_num)) + job_assignments[j, m], name=f"assignment_{j}{m}") # 5

            # Pattern Ban
            len_decision = model.addVars(pattern_num, machine_num, job_num, vtype=GRB.BINARY, name="len_decision")
            for p, ban in enumerate(pattern.keys()):
                for m in range(machine_num):
                    for j in range(job_num):
                        model.addConstr(len_decision[p, m, j] >= (job_assignments[j, m] - ban[j]),
                            name=f"len_decision_lb1_{p}{m}{j}")
                        model.addConstr(len_decision[p, m, j] >= (ban[j] - job_assignments[j, m]),
                            name=f"len_decision_lb2_{p}{m}{j}")
                        model.addConstr(len_decision[p, m, j] <= (ban[j] + job_assignments[j, m]),
                            name=f"len_decision_ub1_{p}{m}{j}")
                        model.addConstr(len_decision[p, m, j] <= 1 - (ban[j] + job_assignments[j, m] - 1),
                            name=f"len_decision_ub2_{p}{m}{j}")
                        
                    model.addConstr(quicksum(len_decision[p, m, j] for j in range(job_num)) >= 1,
                            name=f"pattern_ban_list_{p}{m}"
                    )

            # Constraint: s_min
            for m in range(machine_num):
                model.addGenConstrIndicator(patterns_machine[m], True, quicksum(PATTERN_SETUP[p] * patterns[m, p] for p in range(pattern_num)) <= s_min[m], name=f"s_min_machine_pattern_{m}") # 8
                model.addGenConstrIndicator(patterns_machine[m], False, quicksum(machine_family[m, f] for f in range(family_num)) - 1 <= s_min[m], name=f"s_min_machine_job_{m}")
                
        else:
            for m in range(machine_num):
                model.addConstr(quicksum(assignments[j, m] for j in range(job_num)) >= 1, name=f"machine_job_min_limit_{m}") # 3
                model.addConstr(quicksum(machine_family[m, f] for f in range(family_num)) - 1 <= s_min[m], name=f"s_min_machine_job_{m}")
                model.addConstr(s_min[m] <= family_num - 1, name=f"s_min_machine_ub_{m}") # trivial upper bound, but required?

        # Constraint: Expected yields
        model.addConstr(
            quicksum(
                duration[j] * quicksum(assignments[j, m] * SPEC_CDF[j][m] for m in range(machine_num))
                for j in range(job_num)
            ) >= sum(duration) * Gamma,
            name="expected_yield"
        )

        # Constraint: Expected variance
        model.addConstr(
            quicksum(
                duration[j] * quicksum(assignments[j, m] * SCALED_V[j][m] for m in range(machine_num))
                for j in range(job_num)
            ) <= sum(duration) * Lambda,
            name="expected_variance"
        )

        # Optional Constraint: Ban
        for bitmask, sum_ban in self.ban_list.bans.items():
            ban = self.ban_list.to_list(bitmask, job_num)
            for m in range(machine_num):
                model.addConstr(
                    quicksum(assignments[j, m] * ban[j] for j in range(job_num)) <= sum_ban - 1,
                        name=f"ban_list_{m}"
                )

        # Optional Constraint: Guide
        guide_num = len(guide)
        if guide_num > 0:
            GUIDE_INFORM = list(guide.keys())
            GUIDE_SETUP = [guide[p] for p in GUIDE_INFORM] # lower bound

            big_M = True
            if not big_M:
                machine_guide = model.addVars(machine_num, guide_num, vtype=GRB.BINARY, name="machine_guide")
                GUIDE_SUM = [quicksum(p) for p in GUIDE_INFORM]
                for m in range(machine_num):
                    for g, INFORM in enumerate(GUIDE_INFORM):
                        model.addGenConstrIndicator(machine_guide[m, g], True, 
                                                    quicksum(INFORM[j] * assignments[j, m] for j in range(job_num)) >= GUIDE_SUM[g],    
                                name=f"guide_machine_exist_{m}{g}"
                        )
                        model.addGenConstrIndicator(machine_guide[m, g], False, 
                                                    quicksum(INFORM[j] * assignments[j, m] for j in range(job_num)) <= GUIDE_SUM[g] - 1,    
                                name=f"guide_machine_not_exist_{m}{g}"
                        )
                        model.addGenConstrIndicator(machine_guide[m, g], True, GUIDE_SETUP[g] <= s_min[m],
                                name=f"guide_machine_setup_{m}{g}"
                        )
            else:
                for m in range(machine_num):
                    for g, INFORM in enumerate(GUIDE_INFORM): # job_num - machine_num = big-M
                        model.addConstr(GUIDE_SETUP[g] - (job_num-machine_num) * quicksum(INFORM[j] * (1-assignments[j, m]) for j in range(job_num)) <= s_min[m])

        if lazy:
            model.setParam("LazyConstraints", 1)

            def lazy_constraint_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    current_assignments = model.cbGetSolution(assignments)
                    for m in range(machine_num):
                        J_i = np.array([
                            j for j in range(job_num) if current_assignments[j, m] > 0.5
                        ])
                        is_edd_feasible = self.EDD_for_pattern(J_i)
                        if not is_edd_feasible:
                            model.cbLazy(
                                quicksum(assignments[j, m] for j in J_i) <= len(J_i) - 1
                            )

            model.optimize(lazy_constraint_callback)
        else:
            # Solve the model
            model.optimize()
    
        # Check the solution
        self.status = model.Status

        if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
            if self.viz:
                print("Assign Time:", round(model.Runtime, 2), "Status: INFEASIBLE")
            return False

        if model.SolCount == 0:
            if self.viz:        
                print("Assign Time:", round(model.Runtime, 2), "Status:", GRB_status[self.status])        
            return None
        
        self.lb = model.ObjBound
        
        try:
            # Fill the numpy array with variable values
            self.assignment = np.zeros((job_num, machine_num), dtype=bool)
            self.pattern_selection = np.zeros(machine_num, dtype=bool)

            for m in range(machine_num):
                if pattern_num > 0:
                    self.pattern_selection[m] = bool(round(patterns_machine[m].X))
                else:
                    self.pattern_selection[m] = False

                for j in range(job_num):
                    self.assignment[j, m] = bool(round(assignments[j, m].X))
                    
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        if model.Status == GRB.OPTIMAL:
            return round(model.ObjVal)
        
        return round(model.ObjVal)