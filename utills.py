import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from ortools.sat.python import cp_model
import numbers

from Env import Env

class Datarecoder:
    def __init__(self) -> None:
        self.base_columns = ['Job num', 'Machine num', 'Seed']
        self.row = {key: -1 for key in self.base_columns}
        self.created_TF = False

    # def reset(self):
    #     self.Lambda = -1
    #     self.initial_ban_num = -1
    #     self.total_ban_num = -1
    #     self.total_iter_num = -1
    #     self.objective_value = -1
    #     self.lower_bound = 0
    #     self.total_time = -1

    #     self.feasible_count = 0
    #     self.assign_count = 0

    #     self.cp_value = -1
    #     self.cp_lb = 0
    #     self.cp_time = -1
    #     self.cp_status = -1

    def reset(self):
        self.row = {key: -1 for key in self.row}

    def _convert_value(self, key, value):
        if 'time' in key and value > 3600:
            return 3600
        if value == -1:
            return ''
        if isinstance(value, numbers.Number):
            return round(value, 3)
        
        return value
    
    def create_csv(self, file_name='result.csv'):
        self.file_name = 'result/' + file_name
        with open(self.file_name, 'w', newline='') as f:
            pass

    def write_result_to_csv(self, row: dict):
        with open(self.file_name, 'a', newline='') as f:
            writer = csv.writer(f)

            if not self.created_TF:
                keys = [key for key in row]
                writer.writerow(keys)
                self.created_TF = True

            values = [self._convert_value(key, value) for key, value in row.items()]
            writer.writerow(values)

class new_datarecoder:
    def __init__(self) -> None:
        self.base_columns = ['Job num', 'Machine num', 'Seed']

    def create_csv(self, file_name='result.csv', new_col_names=[]):
        self.file_name = 'result/' + file_name
        self.columns = self.base_columns + new_col_names
        with open(self.file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)

    def write_result_to_csv(self, row=[]):
        """
        row: list of values corresponding to self.columns
        columns: ['Job num', 'Machine num', 'Seed', ...]
        """
        new_row = [
            round(r, 3) if isinstance(r, numbers.Number) and r != -1 else r
            for r in row
        ]
        with open(self.file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

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

# def draw_gantt_chart(machine_schedules):
#     """
#     간트 차트를 그리는 함수
    
#     Parameters:
#     machine_schedules: list of lists
#         각 기계별 작업 스케줄 리스트
#         각 작업은 (job_id, start_time, end_time, deadline, family_id) 형태
#     figsize: tuple
#         그래프 크기
#     title: str
#         차트 제목
#     """
#     all_families = set()
#     for machine_jobs in machine_schedules:
#         for job in machine_jobs:
#             all_families.add(job[4])  # family_id
    
#     family_count = len(all_families)
#     if family_count <= 10:
#         colormap = 'tab10'
#     elif family_count <= 20:
#         colormap = 'tab20'
#     else:
#         colormap = 'nipy_spectral'
#     cmap = cm.get_cmap(colormap)
    
#     # family별 색상 매핑
#     family_colors = {}
#     for i, family in enumerate(sorted(all_families)):
#         if colormap in ['tab10', 'tab20']:
#             color = cmap(i)  # 인덱스 직접 사용
#         else:
#             color = cmap(i / family_count)  # 0~1 사이 값으로 정규화
        
#         family_colors[family] = color  # RGB 튜플 그대로 사용
    
#     # 그래프 설정
#     figsize=(15, 10)
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # 기계 수
#     num_machines = len(machine_schedules)
    
#     # 전체 시간 범위 계산
#     max_time = 0
#     for machine_jobs in machine_schedules:
#         for job in machine_jobs:
#             max_time = max(max_time, job[2], job[3])  # end_time, deadline
    
#     # 각 기계별로 작업 그리기
#     for machine_idx, machine_jobs in enumerate(machine_schedules):
#         y_pos = machine_idx
        
#         # 각 작업 그리기
#         for job in machine_jobs:
#             job_id, start_time, end_time, deadline, family_id = job
            
#             # 작업 바 그리기
#             duration = end_time - start_time
#             rect = patches.Rectangle(
#                 (start_time, y_pos - 0.4), duration, 0.8,
#                 linewidth=1, edgecolor='black', 
#                 facecolor=family_colors[family_id], alpha=0.7
#             )
#             ax.add_patch(rect)
            
#             # 작업 ID 텍스트 추가
#             ax.text(start_time + duration/2, y_pos, f'J{job_id}',
#                    ha='center', va='center', fontsize=9, fontweight='bold')
            
#     # 그래프 설정
#     ax.set_xlim(0, max_time * 1.1)
#     ax.set_ylim(-0.5, num_machines - 0.5)
#     ax.set_xlabel('Time', fontsize=12)
#     ax.set_ylabel('Machine', fontsize=12)
#     ax.set_title("Job Schedule Gantt Chart", fontsize=14, fontweight='bold')
    
#     # Y축 라벨 설정
#     ax.set_yticks(range(num_machines))
#     ax.set_yticklabels([f'Machine {i+1}' for i in range(num_machines)])
    
#     # 격자 추가
#     ax.grid(True, alpha=0.3)
    
#     # 범례 추가 (family별)
#     legend_elements = []
#     for family_id in sorted(all_families):
#         legend_elements.append(
#             patches.Patch(color=family_colors[family_id], alpha=0.7, 
#                          label=f'Family {family_id}')
#         )
    
#     ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
#     plt.tight_layout()
#     return fig, ax

def draw_gantt_chart(machine_schedules, setup_time=10):
    """
    간트 차트를 그리는 함수 (Setup Time 시각화 포함)
    
    Parameters:
    machine_schedules: list of lists
        각 기계별 작업 스케줄 리스트
        각 작업은 (job_id, start_time, end_time, deadline, family_id) 형태
    setup_time: int or float, optional
        Setup 시간 값. 이 값이 주어지고 조건(family 변경, 시간 간격 일치)이 맞으면 차트에 표시됨.
    """
    all_families = set()
    for machine_jobs in machine_schedules:
        for job in machine_jobs:
            all_families.add(job[4])  # family_id
    
    family_count = len(all_families)
    if family_count <= 10:
        colormap = 'tab10'
    elif family_count <= 20:
        colormap = 'tab20'
    else:
        colormap = 'nipy_spectral'
    
    # Matplotlib 버전에 따라 get_cmap 사용법 대응
    try:
        cmap = plt.get_cmap(colormap)
    except:
        cmap = cm.get_cmap(colormap)
    
    # family별 색상 매핑
    family_colors = {}
    sorted_families = sorted(list(all_families))
    for i, family in enumerate(sorted_families):
        if colormap in ['tab10', 'tab20']:
            color = cmap(i)
        else:
            color = cmap(i / family_count)
        family_colors[family] = color
    
    figsize=(15, 10)
    fig, ax = plt.subplots(figsize=figsize)
    num_machines = len(machine_schedules)
    
    # 전체 시간 범위 계산
    max_time = 0
    for machine_jobs in machine_schedules:
        for job in machine_jobs:
            max_time = max(max_time, job[2], job[3]) 
    
    # 각 기계별로 작업 그리기
    for machine_idx, machine_jobs in enumerate(machine_schedules):
        y_pos = machine_idx
        
        # 작업을 시작 시간 순으로 정렬 (연속된 작업 비교를 위해 필수)
        sorted_jobs = sorted(machine_jobs, key=lambda x: x[1])
        
        for i, job in enumerate(sorted_jobs):
            job_id, start_time, end_time, deadline, family_id = job
            
            # 1. 작업 바 그리기
            duration = end_time - start_time
            rect = patches.Rectangle(
                (start_time, y_pos - 0.4), duration, 0.8,
                linewidth=1, edgecolor='black', 
                facecolor=family_colors[family_id], alpha=0.7
            )
            ax.add_patch(rect)
            
            # 작업 ID 텍스트
            ax.text(start_time + duration/2, y_pos, f'J{job_id}',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            
            # 2. Setup Time 그리기 (조건부)
            # 마지막 작업이 아니고 setup_time이 설정되어 있을 때만 확인
            if setup_time is not None and i < len(sorted_jobs) - 1:
                next_job = sorted_jobs[i+1]
                next_start = next_job[1]
                next_family = next_job[4]
                
                # 조건: Family가 다르고, 갭이 Setup 시간과 일치할 때 (부동소수점 오차 고려하여 abs 사용)
                time_gap = next_start - end_time
                if (family_id != next_family) and (abs(time_gap - setup_time) < 1e-5):
                    
                    # Setup 바 그리기 (회색, 빗금 처리로 은은하게)
                    setup_rect = patches.Rectangle(
                        (end_time, y_pos - 0.4), setup_time, 0.8,
                        linewidth=0.5, edgecolor='gray',
                        facecolor='lightgray', hatch='////', alpha=0.5
                    )
                    ax.add_patch(setup_rect)
                    
                    # 옵션: 좁은 공간에 'S' 글자 표시 (공간이 충분할 때만)
                    if setup_time > 1: # 시각적으로 너무 좁지 않을 때만 텍스트 표시
                        ax.text(end_time + setup_time/2, y_pos, 'S',
                                ha='center', va='center', fontsize=7, color='gray')

    # 그래프 설정
    ax.set_xlim(0, max_time * 1.1)
    ax.set_ylim(-0.5, num_machines - 0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title("Job Schedule Gantt Chart", fontsize=14, fontweight='bold')
    
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(num_machines)])
    
    ax.grid(True, alpha=0.3)
    
    # 범례 추가
    legend_elements = []
    for family_id in sorted_families:
        legend_elements.append(
            patches.Patch(color=family_colors[family_id], alpha=0.7, 
                          label=f'Family {family_id}')
        )
    
    # Setup 범례 추가 (setup이 그려진 경우에만 의미가 있으므로)
    if setup_time is not None:
         legend_elements.append(
            patches.Patch(facecolor='lightgray', edgecolor='gray', hatch='////', alpha=0.5,
                          label='Setup Time')
        )

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig, ax

def global_cp(env: Env, Gamma, Lambda, time_limit, log_viz=True, gantt_viz=False):
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
        for m in range(machine_num):
            allocation[m, i] = model.NewBoolVar(f'alloc_{m}_{i}')
            for j in range(job_num):
                relation[m, i, j] = model.NewBoolVar(f'rel_{m}_{i}_{j}')
                setup[m, i, j] = model.NewBoolVar(f'setup_{m}_{i}_{j}')

    for m in range(machine_num):
        for i in range(job_num):
            relation[m, -1, i] = model.NewBoolVar(f'rel_{m}_{-1}_{i}')
            relation[m, i, -1] = model.NewBoolVar(f'rel_{m}_{i}_{-1}')
        
    # objective
    model.minimize(sum([setup[m, i, j] for i in range(job_num) for j in range(job_num) if i != j for m in range(machine_num)])) # 1
    model.add(sum([setup[m, i, j] for i in range(job_num) for j in range(job_num) if i != j for m in range(machine_num)]) <= job_num - machine_num) 

    # Constraint: Expected yield
    SCALING_FACTOR = 1000
    model.add(
        sum(
            duration[j] * sum(allocation[m, j] * int(SPEC_CDF[j][m] * SCALING_FACTOR) for m in range(machine_num))
            for j in range(job_num)
        ) >= int(sum(duration) * Gamma * SCALING_FACTOR),
    ) # 2

    # Constraint: Expected variance
    model.add(
        sum(
            duration[j] * sum(allocation[m, j] * int(SCALED_V[j][m] * SCALING_FACTOR) for m in range(machine_num))
            for j in range(job_num)
        ) <= int(sum(duration) * Lambda * SCALING_FACTOR),
    ) # 3

    # allocation
    for m in range(machine_num):
        model.add(sum(relation[m, -1, i] for i in range(job_num)) <= 1) # 5 # Ensure exactly machine_num earliest start 
        model.add(sum(relation[m, i, -1] for i in range(job_num)) <= 1) # 6 # Ensure exactly machine_num latest start
        for j in range(job_num):
            model.add(sum(relation[m, i, j] for i in range(-1, job_num) if i != j) == allocation[m, j]) # 4
            model.add(sum(relation[m, i, j] for i in range(-1, job_num) if i != j) == sum(relation[m, j, i] for i in range(-1, job_num) if i != j)) # 9

    for i in range(job_num):
        model.add(sum(relation[m, i, j] for j in range(-1, job_num) if i != j for m in range(machine_num)) == 1) # 7
        model.add(sum(relation[m, j, i] for j in range(-1, job_num) if i != j for m in range(machine_num)) == 1) # 8

    # setup
    for m in range(machine_num):
        for i in range(job_num):
            for j in range(job_num):
                if env.job_to_family[i] != env.job_to_family[j]:
                    model.add(setup[m, i, j] >= relation[m, i, j]) # 9

    # deadline
    for j in range(job_num):
        assert duration[j] <= deadline[j]
        model.add(start_time[j] + duration[j] <= deadline[j]) # 10

    # predecessor-successor
    for m in range(machine_num):
        for i in range(job_num):  
            for j in range(job_num):
                if i == j: continue
                model.add(start_time[i] + duration[i] + env.setup_time * setup[m, i, j] <= start_time[j]).only_enforce_if(relation[m, i, j]) # 11

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(log_viz)
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model, solution_printer) # solver가 해결하도록

    print("status:", solver.status_name(status), "objective value:", solver.objective_value, 
    "lower bound:", solver.best_objective_bound, "time:", round(solver.WallTime(), 2))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

        schedule = [[(j, int(solver.Value(start_time[j])), int(solver.Value(start_time[j])) + int(duration[j]), int(deadline[j]), env.job_to_family[j]) 
                     for j in [i for i in range(job_num) if solver.Value(allocation[m, i])]] for m in range(machine_num)]
        schedule = [sorted(machine_jobs, key=lambda x: x[1]) for machine_jobs in schedule]
        result_explain(env, Lambda, schedule, int(solver.objective_value), gantt_viz)

        if status == cp_model.OPTIMAL:
            return solver.objective_value, solver.best_objective_bound, round(solver.WallTime(), 2), True
        else:
            return solver.objective_value, solver.best_objective_bound, round(solver.WallTime(), 2), -1

    elif status == cp_model.INFEASIBLE:
        return False, -1, round(solver.WallTime(), 2), -1, -1, -1
    else:
        print("time limit over")
        return solver.objective_value, solver.best_objective_bound, round(solver.WallTime(), 2), -1

def validate_schedule(env: Env, schedule:list, setup_count:int):
    """
    스케줄이 유효한지 검증하는 함수
    
    Args:
        schedule: [[(job_id, start_time, end_time, deadline, family), ...], ...] 형태의 리스트
        job_num: 전체 작업 수
        setup_time: 셋업 시간 (dictionary 또는 function)
        duration: 작업 기간 리스트
        deadline: 데드라인 리스트
        env: 환경 객체 (job_to_family 포함)
    
    Returns:
        bool: 유효하면 True, 그렇지 않으면 False
    """
    job_num = env.job_num
    duration = env.duration
    setup_time = env.setup_time
    
    # 1. 모든 작업이 스케줄되어 있는지 확인
    scheduled_jobs = set()
    for machine_jobs in schedule:
        for job in machine_jobs:
            job_id = job[0]
            if job_id in scheduled_jobs:
                print(f"Error: Job {job_id} is scheduled on multiple machines")
                return False
            scheduled_jobs.add(job_id)
    
    if len(scheduled_jobs) != job_num:
        missing_jobs = set(range(job_num)) - scheduled_jobs
        print(f"Error: Missing jobs in schedule: {missing_jobs}")
        return False
    
    total_setup_count = 0
    
    # 각 기계별 검증
    for machine_id, machine_jobs in enumerate(schedule):
        if not machine_jobs:  # 빈 기계는 건너뛰기
            continue
            
        # 2. 각 기계에서 작업 시간이 겹치지 않았는지 확인
        for i in range(len(machine_jobs) - 1):
            current_job = machine_jobs[i]
            next_job = machine_jobs[i + 1]
            
            current_end = current_job[2]
            next_start = next_job[1]
            
            # 현재 작업의 종료 시간이 다음 작업의 시작 시간보다 늦으면 겹침
            if current_end > next_start:
                print(f"Error: Jobs {current_job[0]} and {next_job[0]} overlap on machine {machine_id}")
                print(f"Job {current_job[0]} ends at {current_end}, Job {next_job[0]} starts at {next_start}")
                return False
        
        # 3. 데드라인 확인
        for job in machine_jobs:
            job_id, start_time, end_time, deadline_value, family = job
            if end_time > deadline_value:
                print(f"Error: Job {job_id} misses deadline. Ends at {end_time}, deadline is {deadline_value}")
                return False
        
        # 4. 셋업 시간 확인
        machine_setup_count = 0
        for i in range(len(machine_jobs) - 1):
            current_job = machine_jobs[i]
            next_job = machine_jobs[i + 1]
            
            current_family = current_job[4]
            next_family = next_job[4]
            
            # 연속된 작업이 다른 family인 경우 셋업 시간 확인
            if current_family != next_family:
                machine_setup_count += 1
                current_end = current_job[2]
                next_start = next_job[1]
                
                # 셋업 시간 가져오기               
                actual_gap = next_start - current_end
                if actual_gap < setup_time:
                    print(f"Error: Insufficient setup time between jobs {current_job[0]} and {next_job[0]} on machine {machine_id}")
                    print(f"Required setup time: {setup_time}, Actual gap: {actual_gap}")
                    return False
        
        total_setup_count += machine_setup_count
    
    # 5. 추가 검증: 작업의 시작시간 + 지속시간이 종료시간과 일치하는지
    for machine_jobs in schedule:
        for job in machine_jobs:
            job_id, start_time, end_time, deadline_value, family = job
            expected_end = start_time + duration[job_id]
            if end_time != expected_end:
                print(f"Error: Job {job_id} time calculation error. Start: {start_time}, Duration: {duration[job_id]}, Expected end: {expected_end}, Actual end: {end_time}")
                return False
    
    if total_setup_count != setup_count:
        print(f"Error: Total setup operations: {total_setup_count}, input setup count: {setup_count}")
        return False
    
    return True

def result_explain(env: Env, Lambda, schedule, solution, viz=False, stop_if_wrong=True):
    yild, risk = 0, 0
    for m, mr in enumerate(schedule):
        for j, _, _, _, _ in mr:
            yild += env.spec_cdf[j][m] * env.duration[j]
            risk += env.scaled_v[j][m] * env.duration[j]

    valid_TF = validate_schedule(env, schedule, solution)

    if not valid_TF and stop_if_wrong:
        fig, ax = draw_gantt_chart(schedule)
        plt.show()
        raise ValueError('Stop when schedule is wrong')
    
    if valid_TF:
        print(f'Valid schedule! Solution: {solution}, Lambda: {Lambda}, Yield: {round(yild/sum(env.duration), 3)}, risk: {round(risk/sum(env.duration), 3)}')
    
    if viz:
        fig, ax = draw_gantt_chart(schedule)
        plt.show()

    return valid_TF
