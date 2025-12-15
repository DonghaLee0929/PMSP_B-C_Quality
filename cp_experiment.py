import math

from Env import Environment
from utills import Datarecoder, global_cp
from config import experiments

def main():
    target_experiments = ['cp_only2']

    for exp in experiments:
        if 'all' not in target_experiments and exp['name'] not in target_experiments:
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
                    if job_num == 60 and machine_num == 2: continue # TODO

                    family_num = math.floor(job_num/8)+1
                    environment = Environment(job_num, machine_num, family_num, setup_time=10)
                    env = environment.reset(seed=seed, tight=env_conf['tight'], quality=env_conf['quality'])

                    print(f"\nJob: {job_num}, Machine: {machine_num}, Family: {family_num}, Seed: {seed}")

                    Gamma = 0.95
                    Lambda = 1.15 # TODO: load Lambda from result csv file
                    
                    if mode in ['cp']:
                        result = {'Job num': job_num, 'Machine num': machine_num, 'Seed': seed, 'Lambda': Lambda}
                        max_time = run_args.get('max_time', 3600)
                        result['cp value'], result['cp lb'], result['cp time'], result['cp status'] = global_cp(
                            env, Gamma, Lambda, max_time, log_viz=True
                        )

                    recorder.write_result_to_csv(result)

if __name__ == "__main__":
    main()