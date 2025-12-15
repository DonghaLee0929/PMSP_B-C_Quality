import math

from Env import Environment
from framework import Framework
from utills import Datarecoder
from config import experiments

def main():
    target_experiments = ['large_size']

    for exp_name in target_experiments:
        exp = next((exp for exp in experiments if exp['name'] == exp_name), None)
        if exp is None:
            continue

        exp_name = exp['name']
        env_conf = exp['env_config']
        run_args = exp['run_args']

        mode = exp.get('mode', 'framework') 
        job_list = exp.get('jobs', [30, 60])
        machine_list = exp.get('machines', [2, 3, 4, 5])
        seed_list = exp.get('seeds', range(10, 110, 10))

        recorder = Datarecoder()
        recorder.create_csv(f'{exp_name}.csv')

        for job_num in job_list:
            for machine_num in machine_list:
                for seed in seed_list:        
                    family_num = math.floor(job_num/8)+1
                    environment = Environment(job_num, machine_num, family_num, setup_time=10)
                    env = environment.reset(seed=seed, tight=env_conf['tight'], quality=env_conf['quality'])

                    print(f"\nJob: {job_num}, Machine: {machine_num}, Family: {family_num}, Seed: {seed}")

                    Gamma = 0.95
                    Lambda = 1
                    delta = 0.05
                    
                    framework = Framework(env, Gamma, Lambda, delta, asn_viz=False, gantt_viz=False)
                    framework.result['Seed'] = seed
                    if mode in ['framework']:
                        best, best_schedule = framework.run(**run_args)
                        print(f'Iter: {framework.result['total iter num']}, Time: {framework.result['total time']}')

                    recorder.write_result_to_csv(framework.result)

if __name__ == "__main__":
    main()