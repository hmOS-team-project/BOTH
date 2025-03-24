"""
@function: gaussian process scheduler for task scheduling and assigenment.
@author: Hui
"""
import numpy as np
import torch
import time
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from gp_scheduler import GPScheduler

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam
from env.worker import *
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv
from contex_bayes_optimization import ContextualBayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RationalQuadratic, DotProduct, ExpSineSquared

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # np.random.seed(seed)
#     # random.seed(seed)

# Define likelihood of accuracy for humans and machines
robot_accuracies = {0: (0.5, 0.4, 0.3), 1: (0.6, 0.5, 0.3)}
human_accuracies = {2: lambda diff: 0.6 + 0.3 / diff, 3: lambda diff: 0.8 + 0.2 / diff}

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data-folder', type=str, default="data/small_problem_set")
    parser.add_argument('--save-folder', type=str, default="tmp/small_problem_results")
    parser.add_argument('--trained-checkpoint-folder', type=str, default="data/final_trained_models/both/checkpoint_hetgat_small_both.tar")
    parser.add_argument('--infeasible-coefficient', type=float, default=2.0)
    parser.add_argument('--human-noise', action='store_true')
    parser.set_defaults(human_noise=False)
    parser.add_argument('--estimator', action='store_true')
    parser.set_defaults(estimator=False)
    parser.add_argument('--estimator-noise', action='store_true')
    parser.set_defaults(estimator_noise=False)
    parser.add_argument('--human-learning', action='store_true')
    parser.set_defaults(human_learning=False)

    # Test Information
    parser.add_argument('--start-no', default=1, type=int)
    parser.add_argument('--end-no', default=100, type=int)

    # Batch Count
    parser.add_argument('--batch-size', default=8, type=int)
    # Round Count
    parser.add_argument('--num-repeat', default=4, type=int)
    # GP Iteration Count
    parser.add_argument('--num-iterations', default=100, type=int)

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    # random seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    # data folder
    data_folder = args.data_folder
    save_folder = args.save_folder
    trained_checkpoint_folder = args.trained_checkpoint_folder
    
    # Some predefined parameters
    infeasible_coefficient = args.infeasible_coefficient
    real_noise = args.human_noise
    estimator = args.estimator
    est_noise = args.estimator_noise
    hum_learning = args.human_learning

    # problem size
    start_no = args.start_no
    end_no = args.end_no
    total_no = end_no - start_no + 1

    # training parameters
    batch_size = args.batch_size
    num_repeat = args.num_repeat
    num_iterations = args.num_iterations

    # Initializes the output dimension of the embedding vector
    embedding_dim = 64

    # noise = 1e-6

    beta_function = 'const'  # beta_function = 'theor'
    beta_const_val = 2.5

    # Define trade-off factors
    lambda_val = 0
    # Discount factor for reward calculation
    discount = 2.0

    # Define the result file
    print('Save Folder: '+ save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Get the minimum makespan and save it in a .txt file
    # file_min_makespan = save_folder + '/min_makespan_metrics.txt'
    # file_max_accuracy = save_folder + '/max_accuracy_metrics.txt'
    # min_makespan_record = []
    # max_accuracy_record = []

    # Initialize storage for results
    net_problem_makespans = []
    raw_problem_makespans = []
    all_problem_accuracies = []
    all_problem_times = []
    all_problem_feasibilities = []

    device = args.device
    device_id = args.device_id

    '''
    Run multiple episodes on the same environment, with each step changing the human model based on the repetitions of tasks.
    '''
    for count in range(num_repeat):
        raw_makespan_list = []
        net_makespan_list = []
        raw_accuracy_list = []
        raw_time_list = []
        feasible_problem_count = 0

        for i_problem in range(start_no, end_no + 1):
            '''
            Initialize the scheduling environments.
            '''
            # Create Scheduling Environments
            problem_file_name = data_folder + "/problem_" + format(i_problem, '04')
            problem = MRCProblem(fname=problem_file_name)
            # Create a Team
            team = HybridTeam(problem)

            batch_times = []
            batch_success = []
            batch_raw_makespan = []
            batch_raw_accuracy = []
            

            # Create multiple instances of the same environment, since the environments update after every round
            multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(batch_size)]

            for i_b in range(batch_size):
            # print('Batch_size:{},step_count:{}'.format(i_b, step_count))
                
                # Record regret metrics of the problem in current batch
                total_regret_list = []
                
                env = None
                start_t = time.time()
                if estimator:
                    env = multi_round_envs[i_b].get_estimate_environment(est_noise=est_noise)
                else:
                    env = multi_round_envs[i_b].get_actual_environment(human_noise=real_noise)

                # 设置保存路径和文件名
                folder_reward = save_folder + '/reward_metrics_{}_{}.txt'.format(i_problem, i_b)
                save_reward_fig = save_folder + "/reward_fig/reward_metrics_{}_{}.png".format(i_problem, i_b)
                folder_regret = save_folder + '/regret_metrics_{}_{}.txt'.format(i_problem, i_b)
                save_regret_fig = save_folder + "/regret_fig/regret_metrics_{}_{}.png".format(i_problem, i_b)

                # 确保所有必要的目录存在
                os.makedirs(os.path.dirname(folder_reward), exist_ok=True)
                os.makedirs(os.path.dirname(save_reward_fig), exist_ok=True)
                os.makedirs(os.path.dirname(folder_regret), exist_ok=True)
                os.makedirs(os.path.dirname(save_regret_fig), exist_ok=True)

                num_tasks = env.problem.num_tasks  # Unscheduled Tasks
                num_robots = env.team.num_robots
                num_workers = env.team.num_robots + env.team.num_humans

                scheduler = GPScheduler(env, device=torch.device(device, device_id))

                task_order, worker_order = scheduler.generate_action()

                # Initializes the action space
                discvars = {'tasks': task_order, 'workers': worker_order}
                action_dim = len(discvars) * num_tasks

                contexts = {'task': '', 'worker': '', 'state': ''}
                # context_dim = task_embedding_dim * (
                #             num_tasks + 2) + worker_embedding_dim * num_workers + state_embedding_dim
                context_dim = embedding_dim * 3

                # The choice of GP kernel
                length_scale = np.ones(context_dim + action_dim)
                # kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=length_scale)
                kernel = WhiteKernel(noise_level=1) + RationalQuadratic(length_scale=1.0, alpha=1.0)
                # kernel = WhiteKernel(noise_level=1) + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
                # kernel = WhiteKernel(noise_level=1) + ExpSineSquared(length_scale=1.0, periodicity=1.0)

                # Initialize the Bayesian optimizer
                optimizer = ContextualBayesianOptimization(all_actions_dict=discvars, 
                                                            contexts=contexts, kernel=kernel,
                                                            contexts_dim=context_dim, 
                                                            actions_dim=action_dim, 
                                                            embedding_dim=embedding_dim)

                # Initialize Utility Function
                utility = UtilityFunction(kind="ucb", beta_kind=beta_function, beta_const=beta_const_val)

                print('GP decision starts: Repeat_{}/taskID_{}/BatchID_{}'.format(count, i_problem, i_b))
                for i in tqdm(range(0, num_iterations), position=0):
                    # The environment is initialized after each iteration
                    # env.reset()
                    context_output = scheduler.generate_context(trained_checkpoint_folder)
                    pooled_context_output = {k: np.max(np.array(v).reshape(-1, 1), axis=1) for k, v in context_output.items()}

                    context = optimizer.array_to_context(pooled_context_output)
                    # context = optimizer.array_to_context(context_output)
                    action = optimizer.suggest(context, utility)

                    # 保存当前iteration中的临时调度结果
                    # tem_schedule: action = [task_id, worker_id, 1.0]
                    tem_schedule = []   
                    total_accuracy = 0.0

                    # 从 'task' 和 'worker' 中按顺序取出 task_id 和 worker_id，并组成 sample_action
                    for task_id, worker_id in zip(action['tasks'], action['workers']):
                        sample_action = (task_id, worker_id, 1.0)
                        tem_schedule.append(sample_action)

                    #     diff = env.problem.diff[task_id-1]
                    #     if worker_id in range(num_robots):
                    #         accuracy = robot_accuracies[worker_id][diff - 1]
                    #     elif worker_id in range(num_robots, num_workers):
                    #         accuracy = human_accuracies[worker_id](diff)
                    #     else:
                    #         raise ValueError("Invalid resource ID")

                    #     total_accuracy += accuracy

                    # Acquire the likelihood of the accuracy of the lasted scheduling
                    # average_accuracy = total_accuracy / len(tem_schedule)

                    success, reward, done, makespan, average_accuracy = multi_round_envs[i_b].step(tem_schedule, evaluate=estimator, human_noise=real_noise)
                    # reward u_t = r_t + lambda * f_alpha * average_accuracy
                    reward = reward + lambda_val * average_accuracy * env.problem.max_deadline

                    # makespan_accuracy_record.append([makespan, average_accuracy, success])
                    # print('The result of {}-th iteration is: Action:{}, Reward:{}, step_success:{}'.format(i, vAction, reward, step_success))
                    optimizer.register(context, action, reward)

                # # Get the optimization results
                # vReward = []
                # res = optimizer.res
                # for i in range(num_iterations):
                #     vReward.append(res['reward'][i])
                # reward_metrics_f = open(folder_reward, 'a')
                # np.savetxt(reward_metrics_f, np.array(vReward))
                # reward_metrics_f.close()

                # # Plot the reward results and save them to a file
                # plt.figure()
                # plt.plot(vReward)
                # plt.xlabel('Iterations')
                # plt.ylabel('Reward')
                # # 保存图形到指定路径
                # plt.savefig(save_reward_fig)
                # plt.close()

                # # 假设 opt_reward 是最优的 reward 值
                # opt_reward = max(vReward)  # 或者根据具体情况设定

                # # 计算 total regret
                # total_regret = np.cumsum(opt_reward - np.array(vReward))

                # # 将 total regret 值保存到文件
                # total_regret_metrics_f = open(folder_regret, 'a')
                # np.savetxt(total_regret_metrics_f, total_regret)
                # total_regret_metrics_f.close()

                # # 绘制 total regret 曲线并保存到文件
                # plt.figure()
                # plt.plot(total_regret)
                # plt.xlabel('Iterations')
                # plt.ylabel('Total Regret')
                # # 保存图形到指定路径
                # plt.savefig(save_regret_fig)
                # plt.close()

                # Record the inference time of the current iteration
                end_t = time.time()
                print('GP decision-making time: {}'.format(end_t - start_t))
                
                # Record the metrics of the problem instance in current batch
                batch_times.append(end_t - start_t)
                batch_success.append(success)
                batch_raw_makespan.append(makespan)
                batch_raw_accuracy.append(average_accuracy)
                
            # 判断当前problem的可行性
            if any(batch_success):
                feasible_problem_count += 1
                net_min_makespan = min([ms for s, ms in zip(batch_success, batch_raw_makespan) if s])
                raw_min_makespan = min([ms for s, ms in zip(batch_success, batch_raw_makespan) if s])
                accuracy_for_min_makespan = batch_raw_accuracy[batch_raw_makespan.index(raw_min_makespan)]
                net_makespan_list.append(net_min_makespan)
            else:
                raw_min_makespan = min(batch_raw_makespan)
                accuracy_for_min_makespan = batch_raw_accuracy[batch_raw_makespan.index(raw_min_makespan)]
            
            # Get the optimization results
            raw_makespan_list.append(raw_min_makespan)
            raw_accuracy_list.append(accuracy_for_min_makespan)
            raw_time_list.append(min(batch_times))

        # Calculate the staticstic results of all problem instances in current repeat
        net_problem_makespans.append(np.mean(net_makespan_list))
        raw_problem_makespans.append(np.mean(raw_makespan_list))
        all_problem_accuracies.append(np.mean(raw_accuracy_list))
        all_problem_feasibilities.append(100 * feasible_problem_count / total_no)
        all_problem_times.append(np.mean(raw_time_list))
        
    # Compute overall statistics
    net_mean_makespan = np.mean(net_problem_makespans)
    net_std_makespan = np.std(net_problem_makespans)
    raw_mean_makespan = np.mean(raw_problem_makespans)
    raw_std_makespan = np.std(raw_problem_makespans)
    mean_accuracy = np.mean(all_problem_accuracies)
    std_accuracy = np.std(all_problem_accuracies)
    mean_time = np.mean(all_problem_times)
    std_time = np.std(all_problem_times)
    mean_feasibility = np.mean(all_problem_feasibilities)
    std_feasibility = np.std(all_problem_feasibilities)

    print("Total Net Makespan, Total Net Makespan Stdev, Total Raw Makespan, Total Raw Makespan Stdev, Total Accuracy, Total Accuracy Stdev, Feasibility, Feasibility Stdev, Time, Time_Stdev")
    print(net_mean_makespan, net_std_makespan, raw_mean_makespan, raw_std_makespan, mean_accuracy, std_accuracy, mean_feasibility, std_feasibility, mean_time, std_time)
    
    # 将结果打印到txt文件
    results_file_path = os.path.join(save_folder, "both_eval_results.txt")
    with open(results_file_path, "w") as file:
        file.write(f"Mean Net Makespan: {net_mean_makespan}\n")
        file.write(f"Std Net Makespan: {net_std_makespan}\n")
        file.write(f"Mean Raw Makespan: {raw_mean_makespan}\n")
        file.write(f"Std Raw Makespan: {raw_std_makespan}\n")
        file.write(f"Mean Accuracy: {mean_accuracy}\n")
        file.write(f"Std Accuracy: {std_accuracy}\n")
        file.write(f"Mean Inference Time: {mean_time}\n")
        file.write(f"Std Inference Time: {std_time}\n")
        file.write(f"Feasibility Ratio: {mean_feasibility}\n")
        file.write(f"Feasibility Std: {std_feasibility}\n")
    
    print(f"Results saved to {results_file_path}")
    print('Done.')


    # # Get the minimum makespan and success value
    # min_value_makespan = min(makespan_accuracy_record, key=lambda x: x[0])
    # min_makespan_record.append(min_value_makespan)

    # # Get the maximum accuracy and success value
    # max_value_accuracy = max(makespan_accuracy_record, key=lambda x: x[1])
    # max_accuracy_record.append(max_value_accuracy)

    # # # Save the makespan result
    # # makespan_metrics_f = open(fold_makespan, 'a')
    # # np.savetxt(makespan_metrics_f, np.array(makespan_record))
    # # makespan_metrics_f.close()

    # # Get the optimization results
    # vReward = []
    # res = optimizer.res
    # for i in range(num_iterations):
    #     vReward.append(res['reward'][i])
    # reward_metrics_f = open(folder_reward, 'a')
    # np.savetxt(reward_metrics_f, np.array(vReward))
    # reward_metrics_f.close()

    # end_t = time.time()
    # print('Run Time: {:.3f} s'.format(end_t - start_t))
    # # Plot the reward results and save them to a file
    # plt.figure()
    # plt.plot(vReward)
    # plt.xlabel('Iterations')
    # plt.ylabel('Reward')
    # # 保存图形到指定路径
    # plt.savefig(save_reward_fig)

    # # Save the minimum makespan result
    # min_makespan_metrics_f = open(file_min_makespan, 'a')
    # np.savetxt(min_makespan_metrics_f, np.array(min_makespan_record))
    # min_makespan_metrics_f.close()

    # # Save the maximum accuracy result
    # max_accuracy_metrics_f = open(file_max_accuracy, 'a')
    # np.savetxt(max_accuracy_metrics_f, np.array(max_accuracy_record))
    # max_accuracy_metrics_f.close()









