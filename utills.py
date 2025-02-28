import time
import csv
from collections import defaultdict
from itertools import combinations
from gurobipy import GRB

from Env import Env
from Assignment import Assignment, Ban
from Sequence import Sequence, Pattern, Guide, check_min_setup

class Datarecoder:
    def __init__(self) -> None:
        pass

    def reset(self):
        self.first_feasible_sol = -1
        self.first_feasible_time = -1
        self.total_ban_num = -1
        self.objective_value = -1
        self.total_iter_num = -1
        self.total_time = -1
        self.feasible_count = 0
        self.assign_count = 0

        self.cp_feasible_sol = -1
        self.cp_feasible_time = -1
        self.cp_status = -1
        self.cp_value = -1
        self.cp_lb = -1
        self.cp_time = -1

    def _convert_value(self, value):
        """
        Returns an empty string if the value is -1, otherwise returns the original value.
        """
        return '' if value == -1 else value
    
    def create_csv(self, file_name='result.csv'):
        self.file_name = 'result/' + file_name
        with open(self.file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['first_feasible_sol', 'first_feasible_time', 
                            'total_ban_num', 'objective_value', 'total_iter_num', 'total_time', 
                            'cp_feasible_sol', 'cp_feasible_time', 'optimal', 'cp_value', 'cp_lb', 'cp_time'])

    def write_result_to_csv(self):
        if self.total_time > 3600: self.total_time = 3600
        if self.cp_time > 3600: self.cp_time = 3600
        if self.first_feasible_time > 3600: self.first_feasible_time = 3600
        with open(self.file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self._convert_value(self.first_feasible_sol),
                self._convert_value(self.first_feasible_time),
                self._convert_value(self.total_ban_num),
                self._convert_value(self.objective_value),
                self._convert_value(self.total_iter_num),
                self._convert_value(self.total_time),
                self._convert_value(self.cp_feasible_sol),
                self._convert_value(self.cp_feasible_time),
                self._convert_value(self.cp_status),
                self._convert_value(self.cp_value),
                self._convert_value(self.cp_lb),
                self._convert_value(self.cp_time)
            ])


# alloc_result = alloc.gurobi_assignment_min_setup_NN(Gamma, Lambda, assign_max_time, optimal_ub)
def assignment_edd(assign: Assignment, optimal_lb: int, Gamma: float, Lambda: float, delta: float, recoder: Datarecoder, ban: Ban, pattern: dict, guide: dict, assign_max_time: int, max_time: int):
    """
    Performs an assignment procedure using an EDD (Earliest Due Date) approach in conjunction with a Gurobi solver.
    The function iteratively attempts to obtain a feasible assignment that satisfies both the assignment and EDD conditions.
    If the assignment method fails to produce a result within the allotted assignment time (assign_max_time),
    a TimeoutError is raised. Additionally, if the total elapsed time exceeds max_time or if Lambda reaches 2 
    (indicating that the instance might be infeasible), the function terminates early.
    
    When an infeasible assignment is detected (i.e., alloc.status equals GRB.INFEASIBLE), Lambda is increased by delta.
    The function also tracks the number of assignments and records the first assignment time via the recoder object.
    After obtaining an optimal assignment, it verifies the EDD feasibility. If the EDD check fails, the ban list is 
    updated and the process continues until both assignment and EDD conditions are met.
    
    Parameters:
        alloc (assignment): The assignment object that integrates the EDD mechanism.
        optimal_lb (int): The optimal lower bound for the assignment result.
        Gamma (float): A parameter used in the Gurobi assignment setup.
        Lambda (float): A parameter that is adjusted (increased by delta) if the assignment is infeasible.
        delta (float): The incremental value added to Lambda upon detecting an infeasible assignment.
        recoder (datarecoder): An object for recording assignment statistics (e.g., assignment count, first assignment time).
        ban (Ban): The ban list that is used and updated during the assignment process.
        pattern (dict): A dictionary of patterns; only entries with non-None keys where both values are equal are used.
        guide (dict): A guidance dictionary used in the assignment process.
        assign_max_time (int): Maximum time allowed for each assignment attempt.
        max_time (int): Maximum overall time allowed for the entire assignment procedure.
    
    Returns:
        tuple: A tuple containing:
            - alloc_result: The final assignment result from the Gurobi solver.
            - edd_feasible (bool): True if the EDD feasibility condition is satisfied; otherwise, False.
            - Lambda (float): The final value of Lambda after adjustments.
            - alloc.ban_list: The ban list used during the assignment process.
            - edd_iteration_num (int): The number of iterations performed to satisfy the EDD condition.
            - optimal_lb (int): The updated optimal lower bound (which does not decrease from previous assignments).
    
    Raises:
        TimeoutError: If the assignment cannot be completed within the allowed assign_max_time.
    """
    start_time = time.time()

    edd_feasible = False
    edd_iteration_num = 0
    assign.ban_list = ban
    optimal_pattern = {pi: p for pi, p in pattern.items() if p[0] is not None and p[0] == p[1]}
    for _ in range(1000):  # Termination condition: both assignment and EDD are feasible
        if len(optimal_pattern) == 0:
            assign_result = assign.gurobi_assignment_min_setup(Gamma, Lambda, assign_max_time, guide, optimal_lb)
            if assign_result is None: 
                raise TimeoutError  # In case assignment cannot be completed within the time limit (excluding feasible cases)
        else:
            assign_result = assign.gurobi_assignment_min_setup_pattern(Gamma, Lambda, assign_max_time, optimal_pattern, guide, optimal_lb)
            if assign_result is None: 
                raise TimeoutError  # In case assignment cannot be completed within the time limit (excluding feasible cases)
                
        if time.time() - start_time > max_time or Lambda >= 2:  # If the time limit is exceeded or the instance seems infeasible
            print(f"Execution exceeded {max_time} seconds or {Lambda} is too large. Terminating.")
            return None, False, Lambda, assign.ban_list, edd_iteration_num, optimal_lb
        
        if recoder.assign_count == 0: 
            recoder.first_assign_time = round(time.time() - start_time, 2)
        recoder.assign_count += 1

        alloc_infeasible = assign.status == GRB.INFEASIBLE
        if alloc_infeasible:
            Lambda = round(Lambda + delta, 2)
            print(f"Lambda is now {Lambda}")
            continue
        else:
            if assign.status == GRB.OPTIMAL: 
                optimal_lb = assign_result  # Cannot be lower than the previous assignment result
            if not assign.EDD(additional=False):  # Adding to the ban list
                print(f"E{len(assign.ban_list)}", end=' ')
                edd_iteration_num += 1
                continue 
            else:
                edd_feasible = True
                break
    return assign_result, edd_feasible, Lambda, assign.ban_list, edd_iteration_num, optimal_lb


def pattern_manager(env: Env, schedule: list, temp_pattern: list, temp_pattern_information: list, pattern: Pattern, ban: Ban, guide: Guide):
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
    
    Parameters:
        env (Env): The environment object containing job-to-family mappings and other configuration details.
        schedule (list): A list representing the schedule (its usage is implied within the environment).
        temp_pattern (list): A list of temporary patterns, where each entry is a list or tuple with at least three elements.
        temp_pattern_information (list): A list containing additional information about each temporary pattern.
        pattern (Pattern): The primary container for patterns that is updated by this function.
        ban (Ban): A list or container that holds patterns deemed infeasible.
        guide (Guide): A list used to store guidance information, such as lower bounds for feasible patterns.
    
    Returns:
        None. This function updates the 'pattern', 'ban', and 'guide' objects in place.
    
    Raises:
        ValueError: If an optimal or feasible pattern has a None value for its feasibility indicator.
    """
    # Update the patterns
    for m, tp in enumerate(temp_pattern):
        if tp[1] is False:  # If the pattern is infeasible
            ban.append(tp[0])
            continue
        else:
            # Initialize a family counter for guide-related processing
            family_counter = defaultdict(list)
            for i, b in enumerate(tp[0]):
                if b == 1:
                    family_counter[env.job_to_family[i]].append(i)
            if len(family_counter) >= 2:  # If the pattern involves two or more families
                min_possible = check_min_setup(env, family_counter)  # Check if a minimum setup is possible
                if min_possible is False:  # If not possible, update the guide
                    if len(family_counter) == 2:
                        family_list = list(family_counter.values())
                        find_min_subset_two_family(env, family_list[0], family_list[1], guide)
                    else:
                        guide.append((tp[0], len(family_counter)))  # Append to guide with the count of families
                else:  # If a minimum setup is possible, add the pattern
                    if tp[1] is None:
                        raise ValueError
                    pattern.append([tp[0], tp[1], tp[2]])

            if tp[1] is None:  # If the feasibility status is unknown
                pattern.append([tp[0], None, None])  # Append the pattern with unknown bounds
                continue

            # For patterns that are feasible or optimal:
            elif (len(temp_pattern_information) == 0) or (temp_pattern_information[m] is False):
                # If there is no previous information or this is the first occurrence of the pattern
                pattern.append([tp[0], tp[1], tp[2]])
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
                    guide.append((tp[0], tp[2]))  # Append the lower bound to the guide (if optimal, lb equals ub)


def find_min_subset_two_family(env: Env, family_a: list, family_b: list, guide: Guide):
    """
    Determines and updates guide information based on two job families by constructing minimal subsets
    from each family. The function identifies the job in each family with the smallest start time 
    (computed as deadline minus duration), then builds subsets by iteratively adding jobs in descending order 
    of duration until the cumulative duration (plus one) exceeds the smallest start time of the opposite family.
    Finally, it appends to the guide a tuple consisting of a bitmask indicating the union of these subsets and a guide value of 2.
    
    Parameters:
        env (Env): Environment object containing job-related data (deadlines, durations, job count, and family mappings).
        family_a (list): List of job indices for family A.
        family_b (list): List of job indices for family B.
        guide (Guide): The guide list that will be updated with the resulting bitmask and guide value.
        
    Returns:
        None. The guide list is modified in place.
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
            if current_duration + 1 > s_b:
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
            if current_duration + 1 > s_a:
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
    
    Parameters:
        env (Env): The environment object containing job, family, and machine details.
        guide (Guide): A list to be updated with guidance tuples, each containing a bitmask and a numeric guide value.
        pattern (Pattern): An object storing pattern information; its 'patterns' attribute is updated with new patterns.
    
    Returns:
        None. This function updates 'guide' and 'pattern.patterns' in place.
    """
    # Calculate the maximum number of families that can reasonably be assigned to a machine.
    reasonable = min(int(env.family_num / env.machine_num) + 2, env.family_num)  # Maximum families per machine if assignment is reasonable
    
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
                    if check_min_setup(env, comb_dict) is False:
                        # If not possible, update the guide based on the number of families in the combination.
                        if n >= 3:
                            guide.append((all_jobs_tuple, n))
                        else:
                            # For the case with exactly two families, call the helper function to find the minimal subset.
                            family_list = list(comb_dict.values())
                            find_min_subset_two_family(env, family_list[0], family_list[1], guide)
                    else:
                        # If a minimum setup is possible, update the pattern with both lower and upper bounds set to n-1.
                        pattern.patterns[all_jobs_tuple] = (n - 1, n - 1)


# alloc -> EDD -> Sequence framework
def framework(env: Env, Gamma: float, Lambda: float, delta: float, recoder: Datarecoder, assign_max_time=180, seq_max_time=180, max_time=60*60):
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

    Parameters:
        env (Env): The environment object containing jobs, families, machines, deadlines, durations, etc.
        Gamma (float): A parameter used in the assignment procedure.
        Lambda (float): A parameter used in the assignment procedure that is adjusted when an assignment is infeasible.
        delta (float): The incremental value added to Lambda upon detecting an infeasible assignment.
        recoder (datarecoder): An object used to record statistics such as assignment counts and execution times.
        assign_max_time (int, optional): Maximum time allowed for each assignment attempt. Defaults to 180 seconds.
        seq_max_time (int, optional): Maximum time allowed for each sequencing attempt. Defaults to 180 seconds.
        max_time (int, optional): Maximum total execution time for the entire framework. Defaults to 3600 seconds (1 hour).

    Returns:
        tuple: A tuple containing:
            - best (int): The best objective value (lower bound of the sum of setups) found.
            - best_schedule: The best schedule (sequence) corresponding to the best objective value.
            - Lambda (float): The final value of Lambda after all adjustments.
    """
    start_time = time.time()

    best = env.job_num - env.machine_num  # Initialize the best objective value as trivial upper bound
    best_schedule = None
    best_assign_lb = None
    guide = Guide()  # Guide information: holds minimal setup value details
    pattern = Pattern()  # Pattern information: holds objective values and lower bounds (equal to the objective value if optimal),
                        # for feasible and optimal and unknown cases (None, None if unknown)
    ban = Ban()  # Ban information: stores patterns deemed infeasible

    assign = Assignment(env, False)
    seq = Sequence(env, False)

    # For the first iteration, initialize ban and guide information
    assign.ban_list = ban
    ban = assign.insert_initial_ban()
    insert_initial_guide(env, guide, pattern)
    print(f"Initial Ban: {len(ban)}, Initial Pattern: {len(pattern)}, Initial Guide: {len(guide)}, Reasonable: {int(env.family_num/env.machine_num) + 2}")

    # Iterate for a maximum of 200 iterations
    for count in range(200):
        print(f"\nFramework iteration {count}")

        pre_len = len(ban)  # Record the length of the ban list before assignment

        # Run the assignment procedure with current parameters and updated ban, pattern, and guide information
        alloc_result, edd_feasible, Lambda, ban, edd_iteration, best_assign_lb = assignment_edd(
            assign, best_assign_lb, Gamma, Lambda, delta, recoder, ban, pattern.patterns, guide.guides,
            assign_max_time=assign_max_time, max_time=max_time
        )

        if edd_feasible:
            print(f"\nEDD feasible, OPT: {alloc_result}, Best: {best}, EDD result: {assign.setup}")
            print(f"EDD iteration num: {edd_iteration}, EDD new ban num: {len(ban) - pre_len}")
            if len(pattern) > 0:
                optimal_patterns = {pi: p for pi, p in pattern.patterns.items() if p[0] == p[1]}
                print(f"Optimal pattern num: {len(optimal_patterns)}, Guide num: {len(guide)}, Ban num: {len(ban)}")

            # Check if the current assignment result matches the best known objective
            if best <= alloc_result:
                recoder.objective_value = best
                print("Lower bound of sum of setup is equal to the best solution!")
                break

            # Sequence phase: reset the sequence with the current assignment and obtain a new schedule
            seq.reset(env, assign.assignment)
            temp_pattern, temp_pattern_information, schedule = seq.cp_parallel_sequence(
                seq_max_time, max_time, time.time() - start_time, pattern.patterns
            )

            temp_solution = [p[1] for p in temp_pattern]  # Upper bound values from temporary patterns
            # Only proceed if all values in temp_solution are integers (skip if any value is infeasible or unknown)
            if all(isinstance(x, int) and not isinstance(x, bool) for x in temp_solution):
                temp_solution_sum = sum(temp_solution)
                if temp_solution_sum < best:
                    best = temp_solution_sum
                    best_schedule = schedule

                # Record the first feasible solution if not already recorded
                if recoder.feasible_count == 0:
                    recoder.first_feasible_sol = temp_solution_sum
                    recoder.first_feasible_time = round(time.time() - start_time, 2)
                recoder.feasible_count += 1
                print(f"Solution: {temp_solution}, Sum: {temp_solution_sum}, Time: {round(time.time() - start_time, 2)}")

            # If the best known objective equals the assignment result, optimality is guaranteed
            if best <= alloc_result:
                recoder.objective_value = best
                print("Lower bound of sum of setup is equal to the best solution!")
                break

            # Update guide and pattern information based on the new schedule and temporary patterns
            pattern_manager(env, schedule, temp_pattern, temp_pattern_information, pattern, ban, guide)

        else:
            print("EDD Fail or Assign Terminated")
            break

        # Terminate if the total execution time exceeds the maximum allowed time
        if time.time() - start_time > max_time:
            print(f"Execution exceeded {max_time} seconds. Terminating.")
            break

    recoder.total_ban_num = len(ban)
    recoder.total_iter_num = count + 1
    recoder.total_time = round(time.time() - start_time, 2)
    return best, best_schedule, Lambda
