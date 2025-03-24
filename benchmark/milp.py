import argparse
import time
import os
import sys
import numpy as np
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus

# 获取上一级目录的路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
# 将上一级目录添加到sys.path中
sys.path.append(parent_dir)

from env.scheduling_env import SchedulingEnv

def solve_milp(env):
    # 创建 MILP 问题，目标为最小化 makespan
    prob = LpProblem("Scheduling", LpMinimize)

    # 生成任务分配的决策变量:
    # x[(i, j)]=1 表示任务 i 分配给成员 j，i 的取值为 1..num_tasks
    x = LpVariable.dicts("x", ((i, j) for i in range(1, env.problem.num_tasks + 1)
                                for j in range(len(env.team))), cat='Binary')

    # 构造 reward_matrix，表示任务 i 分配给成员 j 所需的执行时间
    reward_matrix = {}
    for i in range(1, env.problem.num_tasks + 1):
        for j in range(len(env.team)):
            reward_matrix[(i, j)] = env.team.get_duration(i - 1, j)

    # 创建 makespan 变量 M，即最后一个任务完成时间的上界
    M = LpVariable("M", lowBound=0)

    # 目标：最小化 makespan
    prob += M, "Minimize_Makespan"

    # 约束1：每个任务必须且仅被分配给一个团队成员
    for i in range(1, env.problem.num_tasks + 1):
        prob += lpSum(x[i, j] for j in range(len(env.team))) == 1, f"Task_{i}_assignment"

    # 约束2：ddl 约束
    # env.problem.ddl 形如 [(task_id, deadline), ...]，假设每个任务的完成时间为：
    # finish_time = env.start_time[task_id - 1] + env.dur[task_id - 1][j] （当任务分配给 j 时）
    for (task_id, deadline) in env.problem.ddl:
        prob += lpSum(
            (env.start_time[task_id - 1] + env.dur[task_id - 1][j]) * x[task_id, j]
            for j in range(len(env.team))
        ) <= deadline, f"DDL_task_{task_id}"

    # 约束3：wait 约束
    # env.problem.wait 形如 [(t1, t2, wtime), ...] 表示任务 t1 的开始时间必须
    # 不小于任务 t2 的完成时间加上等待时间 wtime
    for (t1, t2, wtime) in env.problem.wait:
        prob += lpSum(
            (env.start_time[t2 - 1] + env.dur[t2 - 1][j]) * x[t2, j]
            for j in range(len(env.team))
        ) + wtime <= env.start_time[t1 - 1], f"Wait_{t1}_after_{t2}"

    # 约束4：链接每个任务的完成时间与 makespan M
    # 确保每个任务 i 的完成时间不超过 M
    for i in range(1, env.problem.num_tasks + 1):
        prob += lpSum(
            (env.start_time[i - 1] + env.dur[i - 1][j]) * x[i, j]
            for j in range(len(env.team))
        ) <= M, f"Makespan_task_{i}"

    # 求解问题
    prob.solve()

    # 提取并返回分配方案和问题求解状态
    solution = [(i, j) for i in range(1, env.problem.num_tasks + 1)
                for j in range(len(env.team)) if x[i, j].varValue == 1]

    return solution, LpStatus[prob.status]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='../data/small_problem_set')
    parser.add_argument('--save-folder', type=str, default='../tmp/milp/small_test_results')
    parser.add_argument('--noise', type=str, default="False")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=200)
    
    args = parser.parse_args()

    noise = False
    if args.noise == 'True' or args.noise == 'true':
        noise = True

    print('Baseline: MILP')
    # save_name = './johnsonU/r2t100_001_milp_v1_2'
    data_folder = args.data_folder
    save_folder = args.save_folder
    start_no = args.start
    end_no = args.end
    total_no = end_no - start_no + 1
    infeasible_coefficient = 1.0
    repeat = args.repeat

    print(data_folder)

    feas_count = [0 for i in range(repeat)]
    efficiency_metric = [[] for i in range(repeat)]
    results = [[] for i in range(repeat)]
    record_time = [[] for i in range(repeat)]
    makespan = [[] for i in range(repeat)]
    infeasible_makespan = [[] for i in range(repeat)]
    task_count = [0 for i in range(repeat)]
    total_tasks = [0 for i in range(repeat)]

    for i in range(repeat):
        for graph_no in range(start_no, end_no+1):
            print('Evaluation on {}/{}'.format(graph_no, total_no))
            start_t = time.time()
            
            fname = data_folder + '/problem_%04d' % graph_no
            env = SchedulingEnv(fname, restrict_same_time_scheduling=True, 
                                infeasible_coefficient=infeasible_coefficient, noise = noise)
            total_tasks[i] += env.problem.num_tasks
            terminate = False
            total_reward = 0
            
            solution, status = solve_milp(env)
            
            if status == 'Optimal':
                for task, worker in solution:
                    task_dur = env.dur[task][worker]
                    rt, reward, done, _ = env.step(task, worker, 1.0)
                    env.team.update_status(task, worker, task_dur, env.current_time)
                    total_reward += reward
                    
                    if rt == False:
                        print('Infeasible after %d insertions' % (len(env.partialw)-1))
                        task_count[i] += (len(env.partialw)-1)
                        results[i].append([graph_no, -1])
                        terminate = True
                        infeasible_makespan[i].append(env.problem.max_deadline)
                        break
                    elif env.partialw.shape[0] == (env.problem.num_tasks + 1):
                        task_count[i] += env.problem.num_tasks
                        feas_count[i] += 1
                        dqn_opt = env.min_makespan
                        print('Feasible solution found, min makespan: %f' % (env.min_makespan))
                        results[i].append([graph_no, dqn_opt])
                        makespan[i].append(env.min_makespan)
                        infeasible_makespan[i].append(env.min_makespan)
                        terminate = True
                        break
            
            end_t = time.time()
            total_time = end_t - start_t
            record_time[i].append(total_time)
            print('Time: {:.4f} s'.format(total_time))    
            print('Num feasible:', feas_count[i])

    record_time_np = np.array(record_time, dtype=np.float32)
    print("Tasks:", task_count, total_tasks)
    feas_count_np = np.array(feas_count)
    print('Feasible solution found: {}/{}'.format(np.sum(feas_count), total_no*repeat))
    print('Average time per instance:  {:.4f}, stdev: {:.4f}'.format(np.mean(record_time_np), np.std(record_time_np)))
    feasible_makespans = [np.mean(np.asarray(m)) for m in makespan]
    print("Feasible Makespan: ", np.mean(feasible_makespans), ", stdev: ", np.std(feasible_makespans))
    infeasible_makespans = [np.mean(np.asarray(im)) for im in infeasible_makespan]
    print("Total Makespan: ", np.mean(infeasible_makespans), ", stdev: ", np.std(infeasible_makespans))
    feas_percentage = 100 * feas_count_np / (total_no)
    print("Feasible Percentage: {}, stdev: {}%".format(np.mean(feas_percentage), np.std(feas_percentage)))

    # 保存结果到txt文件
    results_file_path = os.path.join(save_folder, "eval_results.txt")
    with open(results_file_path, "w") as f:
        f.write("Tasks: {}/{}\n".format(task_count, total_tasks))
        f.write('Feasible solution found: {}/{}\n'.format(np.sum(feas_count), total_no*repeat))
        f.write('Average time per instance: {:.4f}, stdev: {:.4f}\n'.format(np.mean(record_time_np), np.std(record_time_np)))
        f.write("Feasible Makespan: {:.4f}, stdev: {:.4f}\n".format(np.mean(feasible_makespans), np.std(feasible_makespans)))
        f.write("Total Makespan: {:.4f}, stdev: {:.4f}\n".format(np.mean(infeasible_makespans), np.std(infeasible_makespans)))
        f.write("Feasible Percentage: {:.2f}%, stdev: {:.2f}%\n".format(np.mean(feas_percentage), np.std(feas_percentage)))

    print(f"Results saved to {results_file_path}")
    print('Done.')