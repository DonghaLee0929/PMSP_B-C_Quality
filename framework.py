import time
from collections import defaultdict
from itertools import combinations
from gurobipy import GRB

from Env import Env
from Assignment import Assignment, Ban, check_min_setup
from Sequence import Sequence, Pattern, Guide
from utills import result_explain

def pattern_manager(env: Env, temp_pattern: list, temp_pattern_information: list, pattern: Pattern, ban: Ban, guide: Guide, guide_generation: bool):
    """
    Updates the pattern list, ban list, and guide based on temporary patterns and their feasibility status.

    This function processes each temporary pattern in 'temp_pattern' and updates the main 'pattern' container
    accordingly. For each temporary pattern:
      - If the pattern is marked as infeasible (tp[1] is False), it is added to the ban list.
      - Otherwise, the function counts job families for entries in the pattern (where a job is active if its value is 1).
        - If the pattern contains jobs from two or more families, it checks whether a minimum setup is possible using
          the 'check_min_setup' function.
            - If the minimum setup is not possible:
                - If exactly two families are present, it calls 'guide_manager' to update the guide.
                - Otherwise, it appends a tuple containing the pattern and the number of families to the guide.
            - If a minimum setup is possible, the pattern is added to the pattern list (with a ValueError raised if the feasibility indicator is None).
      - If the feasibility indicator is None (unknown), the pattern is added with unknown bounds.
      - For patterns that are feasible or optimal:
            - If there is no previous pattern information or this is the first occurrence of the pattern, the pattern and its lower bound (as guide) are appended.
            - If the pattern has been seen before (with previous information available), the function updates the stored pattern bounds if the current results show improvement.
    """
    # Update the patterns
    for m, tp in enumerate(temp_pattern):
        # tp[0]: pattern tp[1]: upper bound or status tp[2]: lower bound tp[3]: schedule
        if tp[1] is False:  # If the pattern is infeasible
            ban.append(tp[0])
            continue
        else:
            family_counter = defaultdict(list)
            for i, b in enumerate(tp[0]):
                if b == 1:
                    family_counter[env.job_to_family[i]].append(i)
            if guide_generation:
                # Initialize a family counter for guide-related processing
                if len(family_counter) >= 2:  # If the pattern involves two or more families
                    temp_schedule = []
                    min_possible = check_min_setup(env, family_counter, env.setup_time, temp_schedule)  # Check if a minimum setup is possible
                    if min_possible is False:  # If not possible, update the guide
                        if len(family_counter) == 2:
                            family_list = list(family_counter.values())
                            find_min_subset_two_family(env, family_list[0], family_list[1], guide)
                        else:
                            guide.append((tp[0], len(family_counter)))  # Append to guide with the count of families
                    else:  # If a minimum setup is possible, add the pattern
                        if tp[1] is None:
                            raise ValueError
                        pattern.append([tp[0], tp[1], tp[2], temp_schedule])

            if tp[1] is None:  # If the feasibility status is unknown
                pattern.append([tp[0], None, None, None])  # Append the pattern with unknown bounds
                continue

            # For patterns that are feasible or optimal:
            elif (len(temp_pattern_information) == 0) or (temp_pattern_information[m] is False):
                # If this is the first occurrence of the pattern
                pattern.append([tp[0], tp[1], tp[2], tp[3]])
                if len(family_counter) - 1 < tp[2]:
                    guide.append((tp[0], tp[2]))  # Append the lower bound to the guide (if optimal, lb equals ub)
                continue

            # For a pattern that has been seen before and previously was feasible or unknown, but now is optimal or feasible:
            elif len(temp_pattern_information[m]) >= 2:
                tp_ub, tp_lb = pattern.patterns[temp_pattern_information[m][0]]
                if (tp_ub is not None) and (tp_lb is not None):  # If the previous pattern was feasible
                    if tp_ub > tp[1]:  # If the new upper bound is better, update it
                        temp1, temp2 = pattern.patterns[temp_pattern_information[m][0]]
                        pattern.patterns[temp_pattern_information[m][0]] = (tp[1], temp2)  # Update the pattern
                    if tp_lb < tp[2]:
                        temp1, temp2 = pattern.patterns[temp_pattern_information[m][0]]
                        pattern.patterns[temp_pattern_information[m][0]] = (temp1, tp[2])  # Update the pattern
                        guide.append((tp[0], tp[2]))  # Append the updated lower bound to the guide (if optimal, lb equals ub)
                else:  # If the previous pattern was unknown
                    pattern.patterns[temp_pattern_information[m][0]] = (tp[1], tp[2])  # Update the pattern
                    if len(family_counter) - 1 < tp[2]:
                        guide.append((tp[0], tp[2]))  # Append the lower bound to the guide (if optimal, lb equals ub)


def find_min_subset_two_family(env: Env, family_a: list, family_b: list, guide: Guide):
    """
    Determines and updates guide information based on two job families by constructing minimal subsets
    from each family. The function identifies the job in each family with the smallest start time 
    (computed as deadline minus duration), then builds subsets by iteratively adding jobs in descending order 
    of duration until the cumulative duration (plus one) exceeds the smallest start time of the opposite family.
    Finally, it appends to the guide a tuple consisting of a bitmask indicating the union of these subsets and a guide value of 2.
    """
    # Identify the job in family A with the smallest start time (start time = deadline - duration)
    s_a, s_a_index = min((env.deadline[i] - env.duration[i], i) for i in family_a)  # Find smallest start time in family A
    
    # Identify the job in family B with the smallest start time
    s_b, s_b_index = min((env.deadline[i] - env.duration[i], i) for i in family_b)  # Find smallest start time in family B

    # Initialize subset A with the job having the smallest start time in family A
    subset_a = [s_a_index]
    current_duration = env.duration[s_a_index]
    
    # Build subset A: iterate over family A in descending order of duration
    # Add jobs until the cumulative duration + 1 exceeds the start time of the selected job from family B (s_b)
    for job in sorted(family_a, key=lambda x: env.duration[x], reverse=True):
        if job != s_a_index:
            current_duration += env.duration[job]
            if current_duration + env.setup_time > s_b:
                break
            subset_a.append(job)

    # Initialize subset B with the job having the smallest start time in family B
    subset_b = [s_b_index]
    current_duration = env.duration[s_b_index]
    
    # Build subset B: iterate over family B in descending order of duration
    # Add jobs until the cumulative duration + 1 exceeds the start time of the selected job from family A (s_a)
    for job in sorted(family_b, key=lambda x: env.duration[x], reverse=True):
        if job != s_b_index:
            current_duration += env.duration[job]
            if current_duration + env.setup_time > s_a:
                # If both subsets currently have only one job, add this job to subset B to avoid a guide value of 2 for only two jobs
                if len(subset_a) == 1 and len(subset_b) == 1:
                    subset_b.append(job)
                break
            subset_b.append(job)

    # Append to guide a tuple:
    # - The first element is a bitmask (tuple) indicating which jobs (from 0 to env.job_num - 1)
    #   are included in the union of subset A and subset B.
    # - The second element is the guide value (2).
    guide.append((tuple([1 if j in subset_a + subset_b else 0 for j in range(env.job_num)]), 2))


def insert_initial_guide(env: Env, guide: Guide, pattern: Pattern):
    """
    Inserts initial guidance and pattern information based on job families in the environment.

    This function calculates a "reasonable" maximum number of families that can be assigned to a machine,
    based on the ratio of total families to machines. If this maximum is at least 2 (since cases with only one
    family per machine are not considered), the function proceeds as follows:
    
      1. Constructs a dictionary that maps each family to the list of job indices belonging to that family.
      2. Iterates over all possible combinations of families (from 2 up to the computed reasonable maximum).
      3. For each combination:
           - Builds a combined dictionary (comb_dict) for the selected families.
           - Creates a bitmask (all_jobs_tuple) representing all jobs belonging to these families.
           - Checks if a minimum setup is possible for this combination using the 'check_min_setup' function.
               * If a minimum setup is not possible:
                   - For combinations of three or more families, appends the bitmask and the family count (n)
                     to the guide.
                   - For combinations of exactly two families, calls 'find_min_subset_two_family' to update the guide.
               * If a minimum setup is possible, updates the pattern dictionary with both lower and upper bounds set to n-1.
    """
    # Calculate the maximum number of families that can reasonably be assigned to a machine.
    # reasonable = min(int(env.family_num / env.machine_num) + 2, env.family_num)  # Maximum families per machine if assignment is reasonable
    # reasonable = env.family_num-env.machine_num+1  # Maximum families per machine if assignment is reasonable
    reasonable = env.family_num//env.machine_num+2  # Maximum families per machine if assignment is reasonable
    
    if reasonable >= 2:  # Only consider cases where at least 2 families can be assigned to a machine
        # Build a dictionary mapping each family to its corresponding job indices.
        family_counter = defaultdict(list)
        for i in range(env.job_num):
            family_counter[env.job_to_family[i]].append(i)
        families = list(family_counter.keys())

        # Iterate over possible numbers of families assigned (from 2 up to the reasonable maximum)
        for n in range(2, reasonable + 1):
            if n <= len(families):
                # For each combination of 'n' families, try to restrict the scenario where these families are assigned together.
                for comb in combinations(families, n):
                    # Build a dictionary for the current combination of families.
                    comb_dict = {family: family_counter[family] for family in comb}
                    # Create a combined bitmask for all jobs belonging to the selected families.
                    all_jobs = [job for job_list in comb_dict.values() for job in job_list]
                    all_jobs_tuple = tuple([1 if j in all_jobs else 0 for j in range(env.job_num)])
                    
                    # Check if a minimum setup is possible for this combination of families.
                    temp_schedule = []
                    if check_min_setup(env, comb_dict, env.setup_time, temp_schedule) is False:
                        # If not possible, update the guide based on the number of families in the combination.
                        if n >= 3:
                            guide.append((all_jobs_tuple, n))
                        else:
                            # For the case with exactly two families, call the helper function to find the minimal subset.
                            family_list = list(comb_dict.values())
                            find_min_subset_two_family(env, family_list[0], family_list[1], guide)
                    else:
                        # If a minimum setup is possible, update the pattern with both lower and upper bounds set to n-1.
                        # pattern.patterns[all_jobs_tuple] = (n - 1, n - 1) # Do not append it to guide because it is trivial.
                        pattern.append([all_jobs_tuple, n - 1, n - 1, temp_schedule])


class Framework:
    def __init__(self, env: Env, Gamma: float, Lambda: float, delta: float, asn_viz: bool = False, gantt_viz: bool = False):
        self.env = env
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.delta = delta
        self.asn_viz=asn_viz
        self.gantt_viz=gantt_viz

        self.fixed = None

        fixed_keys = ['Job num', 'Machine num', 'Seed', 'Lambda', 'objective value', 'lower bound', 'total time']
        self.result = {key:-1 for key in fixed_keys}
        self.result['Job num'] = self.env.job_num
        self.result['Machine num'] = self.env.machine_num

    def fix_schedule(self, fixed=None):
        print(f"Fixed job num: {len(fixed[0])}")
        self.fixed = fixed

    # alloc -> EDD -> Sequence framework
    def run(self, asn_max_time=180, seq_max_time=180, max_time=60*60, 
            initial_guide_TF=True, guide_generation_TF=True, optimal_guide_TF=True, lazy_TF=False, feasible_only=False):
        """
        Main framework that integrates assignment (via an EDD approach) and sequencing (using a CP parallel method).

        This function orchestrates the overall procedure to improve the assignment of jobs by iteratively:
        1. Setting up initial ban, guide, and pattern information.
        2. Running the assignment procedure to obtain an assignment result and verify its EDD feasibility.
        3. If the assignment is EDD feasible, it resets and invokes the sequencing procedure to obtain an improved schedule
            along with new pattern information.
        4. Evaluating the new solution based on the sum of upper bounds (ub) from the temporary patterns.
        5. Updating the best found solution if an improvement is detected.
        6. Managing and updating the guide, pattern, and ban lists via the pattern_manager.
        7. Terminating if a solution matching the lower bound is found, if the EDD fails, or if the overall maximum time is exceeded.
        """
        self.asn_time = asn_max_time
        self.seq_time = seq_max_time
        self.max_time = max_time

        self.start_time = time.time()

        best = self.env.job_num - self.env.machine_num  # Initialize the best objective value as trivial upper bound
        best_schedule = None

        self.optimal_lb = 0 # assign의 lower bound
        self.asn_ub = best # assign의 ub: seq에서 구한 objective value

        self.guide = Guide()  # Guide information: holds minimal setup value details
        self.pattern = Pattern()  # Pattern information: holds objective values and lower bounds (equal to the objective value if optimal),
                            # for feasible and optimal and unknown cases (None, None if unknown)

        if self.fixed is not None:
            x, y, s, t = self.fixed
            seq_fixed = [y,s,t]
        else:
            x, seq_fixed = None, None

        self.asn = Assignment(env=self.env, fix=x, viz=self.asn_viz, edd_additional=True, edd_with_setup=True)
        self.seq = Sequence(env=self.env, fix=seq_fixed)
        self.asn.ban_list = Ban() # Ban information: stores patterns deemed infeasible
        self.asn.ban_list = self.asn.insert_initial_ban()
        self.result['initial ban num'] = len(self.asn.ban_list)
        if initial_guide_TF and guide_generation_TF:
            insert_initial_guide(self.env, self.guide, self.pattern)

        print(f"Initial Ban: {len(self.asn.ban_list)}, Initial Pattern: {len(self.pattern)}, Initial Guide: {len(self.guide)}")

        # Iterate for a maximum of 200 iterations
        for count in range(200):
            print(f"\nFramework iteration {count}")

            if len(self.pattern) > 0:
                optimal_pattern = {pi: p for pi, p in self.pattern.patterns.items() if (p[0] is not None) and (p[0] == p[1])}
                print(f"Optimal pattern num: {len(optimal_pattern)}, Guide num: {len(self.guide)}")

            pre_len = len(self.asn.ban_list)  # Record the length of the ban list before assignment

            # Run the assignment procedure with current parameters and updated ban, pattern, and guide information
            assign_result, edd_feasible, edd_iteration = self.assignment_edd(self.pattern.patterns, self.guide.guides, 
                                                                             cg_TF= optimal_guide_TF, lazy_TF=lazy_TF)

            # Terminate if the total execution time exceeds the maximum allowed time
            if time.time() - self.start_time > self.max_time: # stop to go to sequencing
                print(f"Execution exceeded {self.max_time} seconds. Terminating.")
                break

            if edd_feasible:
                print(f"EDD feasible, EDD iteration num: {edd_iteration}, EDD new ban num: {len(self.asn.ban_list) - pre_len}, Ban num: {len(self.asn.ban_list)}")
                print(f"Assign result! OPT: {assign_result}, LB: {round(self.optimal_lb)}, Best: {best}")

                # Check if the current assignment result matches the best known objective
                if best <= assign_result: # (exact) Lower bound increases + already best known schedule exists
                    result_explain(env=self.env, Lambda=self.Lambda, solution=best, schedule=best_schedule, viz=self.gantt_viz)

                    self.result['objective value'] = best
                    if self.asn.status == GRB.OPTIMAL:
                        print("Optimal solution!")
                        break
                    else:
                        gap = 100.0 if round(self.optimal_lb) == 0 else round((best / round(self.optimal_lb) - 1) * 100, 2)
                        self.asn_time = int(self.max_time - (time.time() - self.start_time))
                        print(f"Promising feasible solution! Lower bound: {round(self.optimal_lb)}, Gap: {gap}%, More asn time: {self.asn_time}")
                        if feasible_only: break
                        continue

                # Sequence phase: reset the sequence with the current assignment and obtain a new schedule
                self.seq.reset(self.env, self.asn.assignment)
                temp_pattern, temp_pattern_information, schedule = self.seq.cp_parallel_sequence(pattern=self.pattern,
                    time_limit=self.seq_time, framework_remaining_time=self.max_time - (time.time() - self.start_time), 
                    parallel=True, generation=guide_generation_TF)

                temp_solution = [p[1] for p in temp_pattern]  # Upper bound values from temporary patterns
                # Only proceed if all values in temp_solution are integers (skip if any value is infeasible or unknown)
                if all(isinstance(x, int) and not isinstance(x, bool) for x in temp_solution):
                    current_solution = sum(temp_solution)

                    result_explain(env=self.env, Lambda=self.Lambda, solution=current_solution, schedule=schedule, viz=self.gantt_viz)

                    print(f"Feasible solution! Solution: {current_solution}, Time: {round(time.time() - self.start_time, 3)}")
                    if current_solution < best:
                        best = current_solution
                        best_schedule = schedule
                        self.asn_ub = best
                        self.result['objective value'] = best
        
                    if best <= assign_result:
                        self.result['objective value'] = best
                        if self.asn.status == GRB.OPTIMAL: # If the best known objective equals the assignment result, optimality is guaranteed
                            print("Optimal solution!")
                            break
                        else:
                            gap = 100.0 if round(self.optimal_lb) == 0 else round((best / round(self.optimal_lb) - 1) * 100, 2)
                            self.asn_time = int(self.max_time - (time.time() - self.start_time))
                            print(f"Promising feasible solution! Lower bound: {round(self.optimal_lb)}, Gap: {gap}%, More asn time: {self.asn_time}")
                            if feasible_only: break
                else:
                    print(f"Sequencing infeasible or time limited! Time: {round(time.time() - self.start_time, 3)}")

                # Update guide and pattern information based on the new schedule and temporary patterns
                pattern_manager(self.env, temp_pattern, temp_pattern_information, self.pattern, self.asn.ban_list, self.guide, guide_generation=guide_generation_TF)

            else:
                print("Assignment Fail")
                break

            # Terminate if the total execution time exceeds the maximum allowed time
            if time.time() - self.start_time > self.max_time: # seq에서 assign으로 가는걸 막음
                print(f"Execution exceeded {self.max_time} seconds. Terminating.")
                break

        self.result['total ban num'] = len(self.asn.ban_list)
        self.result['total iter num'] = count + 1
        self.result['total time'] = round(time.time() - self.start_time, 3)

        return best, best_schedule

    def assignment_edd(self, pattern: dict, guide: dict, cg_TF: bool, lazy_TF: bool):
        edd_feasible = False
        edd_iteration_num = 0
        optimal_pattern = {pi: p for pi, p in pattern.items() if (p[0] is not None) and (p[0] == p[1])} if cg_TF else {}

        loop_count = 100 if lazy_TF else 1000
        for _ in range(loop_count):
            if time.time() - self.start_time > self.max_time: 
                return None, False, edd_iteration_num

            assign_result = self.asn.gurobi_assignment_min_setup(self.Gamma, self.Lambda, self.asn_time, 
                                                            optimal_pattern, guide, 
                                                            self.optimal_lb, self.asn_ub, lazy=lazy_TF)
            if assign_result is None:
                print("Warning: Timeout for assignment")
                self.asn_time += 360
                continue

            assign_infeasible = self.asn.status == GRB.INFEASIBLE
            if assign_infeasible:
                self.Lambda = round(self.Lambda + self.delta, 2)
                self.result['Lambda'] = self.Lambda
                print(f"Lambda is now {self.Lambda}")
                if self.Lambda > 2:
                    print(f"{self.Lambda} is too large. Terminating.")
                    return None, False, 1
                continue
                
            self.optimal_lb = self.asn.lb
            self.result['lower bound'] = round(self.optimal_lb)
            edd_iteration_num += 1
            if not self.asn.EDD():  # Adding to the ban list
                if lazy_TF:
                    raise ValueError(f"Error: EDD failed even though lazy constraint was added.")
                else:
                    if self.asn_viz:
                        print(f"EDD added Ban: {len(self.asn.ban_list)}")
                    continue
            else:
                edd_feasible = True
                break

        return assign_result, edd_feasible, edd_iteration_num