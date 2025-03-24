"""
Created on Tuesday November 23 12:47:42 2021

@author: baltundas
"""

import os
import random
import argparse

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

def generate_task_allocation_problems(folder, num_humans, num_robots, num_tasks, start, finish):
    os.makedirs(folder, exist_ok=True)

    makespan_list = []
    passed_list = []
    time_list = []

    for i in range(start, finish+1):
        num_tasks_chosen = num_tasks
        if isinstance(num_tasks, list):
            num_tasks_chosen = random.randint(num_tasks[0], num_tasks[1])   # 任务数 [min, max]
        num_humans_chosen = num_humans
        if isinstance(num_humans, list):
            num_humans_chosen = random.randint(num_humans[0], num_humans[1])
        num_robots_chosen = num_robots
        if isinstance(num_robots, list):
            num_robots_chosen = random.randint(num_robots[0], num_robots[1])

        file_name = folder + "/problem_" + format(i, '04')
        problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen)
        # success, makespan, passed, elapsed_time = problem.save_to_file(file_name)
        success = problem.save_to_file(file_name)

        # makespan_list.append(makespan)
        # passed_list.append(passed)
        # time_list.append(elapsed_time)

        while not success:
            problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen)
            success = problem.save_to_file(file_name)
            # success, makespan, passed, elapsed_time = problem.save_to_file(file_name)
            # makespan_list[-1] = makespan
            # passed_list[-1] = passed
            # time_list[-1] = elapsed_time

        # while not success:
        #     problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen)
        #     success = problem.save_to_file(file_name)
    
    # output_file = os.path.join(folder, "gurobi_result.txt")
    # with open(output_file, "w") as f:
    #     for mksp, p, t in zip(makespan_list, passed_list, time_list):
    #         f.write(f"{mksp},{p}, {t}\n")

    # # 计算平均 makespan 和 passed 比例
    # avg_makespan = sum(makespan_list) / len(makespan_list) if makespan_list else 0
    # passed_count = sum(1 for x in passed_list if x)
    # passed_ratio = passed_count / len(passed_list) if passed_list else 0
    # avg_time = sum(time_list) / len(time_list) if time_list else 0

    # with open(output_file, "a") as f:
    #     f.write(f"Average Makespan: {avg_makespan}\n")
    #     f.write(f"Passed Ratio: {passed_ratio}\n")
    #     f.write(f"Average Execution Time: {avg_time}\n")

            
def read_test_data(folder, start, finish):
    problems = []
    for i in range(start+1, finish+1):
        file_name = folder + "/problem_" + format(i, '04')
        problem = MRCProblem(fname=file_name)
        problems.append(problem)
    return problems

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--human', type=int, default=2)
    parser.add_argument('--robot', type=int, default=2)
    parser.add_argument('--task-min', type=int, default=9)
    parser.add_argument('--task-max', type=int, default=11)
    parser.add_argument('--start-problem', type=int, default=0)
    parser.add_argument('--end-problem', type=int, default=2000)
    
    args = parser.parse_args()
    generate_task_allocation_problems(args.folder, args.human, args.robot, [args.task_min, args.task_max],
                                      args.start_problem, args.end_problem)
    print('Done')