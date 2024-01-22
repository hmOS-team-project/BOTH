from contex_bayes_opt import ContextualBayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RationalQuadratic, ExpSineSquared
import numpy as np
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from gp_scheduler import GPScheduler

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

# Define likelihood of accuracy for humans and machines
robot_accuracies = {0: (0.8, 0.6, 0.4), 1: (0.7, 0.5, 0.3)}
human_accuracies = {2: lambda diff: 0.5 + 0.49 / diff, 3: lambda diff: 0.5 + 0.4 / diff}


if __name__ == '__main__':
    # Some predefined parameters
    estimator = True
    human_noise = True
    estimator_noise = False

    setup_seed(10)  # Set random seeds
    BATCH_SIZE = 1  # batch_size = 5
    TOTAL_PROBLEM_NUM = 100  # total_problem_num = 100

    # Initializes the output dimension of the embedding vector
    task_embedding_dim = 64
    worker_embedding_dim = 64
    state_embedding_dim = 64

    # The choice of GP kernel
    # length_scale = np.ones(context_dim + action_dim)
    # kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=length_scale)
    kernel = WhiteKernel(noise_level=1) + RationalQuadratic(length_scale=1.0, alpha=1.0)
    noise = 1e-6

    beta_function = 'const'  # beta_function = 'theor'
    beta_const_val = 2.5

    # Define trade-off factors
    lambda_val = 0.8
    # Discount factor for reward calculation
    discount = 2.0

    # Define the result file
    folder_result = 'tmp/small_problem_set'
    # Get the minimum makespan and save it in a .txt file
    file_min_makespan = folder_result + '/min_makespan_metrics.txt'
    file_max_accuracy = folder_result + '/max_accuracy_metrics.txt'
    min_makespan_record = []
    max_accuracy_record = []


    for i_problem in range(1, TOTAL_PROBLEM_NUM + 1):
        '''
        Initialize the scheduling environments.
        '''
        # Create Scheduling Environments
        # problem_num = 3  # choose the problem instance
        # folder = 'data/small_training_set/problems'
        folder_data = 'data/small_problem_set'
        # folder = 'data/large_test_set'
        problem_file_name = folder_data + "/problem_" + format(i_problem, '04')
        problem = MRCProblem(fname=problem_file_name)
        team = HybridTeam(problem)

        # Create multiple instances of the same environment, since the environments update after every round
        multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(BATCH_SIZE)]

        for i_b in range(BATCH_SIZE):
            start_t = time.time()
            if estimator:
                env = multi_round_envs[i_b].get_estimate_environment(est_noise=estimator_noise)
            else:
                env = multi_round_envs[i_b].get_actual_environment(human_noise=human_noise)

            # 设置保存路径和文件名
            folder_reward = folder_result + '/reward_metrics_{}_{}.txt'.format(i_problem, i_b)
            save_reward_fig = folder_result + "/reward_fig/reward_metrics_{}_{}.png".format(i_problem, i_b)

            num_tasks = env.problem.num_tasks  # Unscheduled Tasks
            num_workers = env.team.num_robots + env.team.num_humans

            scheduler = GPScheduler(env)

            task_order, worker_order = scheduler.generate_action()

            # Initializes the action space
            discvars = {'tasks': task_order, 'workers': worker_order}
            action_dim = len(discvars) * num_tasks

            contexts = {'task': '', 'worker': '', 'state': ''}
            # context_dim = task_embedding_dim * (
            #             num_tasks + 2) + worker_embedding_dim * num_workers + state_embedding_dim
            context_dim = task_embedding_dim + worker_embedding_dim + state_embedding_dim

            # Initialize the Bayesian optimizer
            optimizer = ContextualBayesianOptimization(all_actions_dict=discvars,
                                                       contexts=contexts,
                                                       kernel=kernel,
                                                       contexts_dim=context_dim,
                                                       actions_dim=action_dim,
                                                       embedding_dim=task_embedding_dim)

            # Initialize Utility Function
            utility = UtilityFunction(kind="ucb", beta_kind=beta_function, beta_const=beta_const_val)

            num_Iterations = 200  # Total number of iterations
            makespan_accuracy_record = []  # Record makespan and accuracy of the problem

            for i in tqdm(range(0, num_Iterations), position=0):
                # The environment is initialized after each iteration
                # env.reset()
                context_output = scheduler.generate_context()
                context = optimizer.array_to_context(context_output)
                action = optimizer.suggest(context, utility)

                total_schedule = []
                total_accuracy = 0.0

                # 从 'task' 和 'worker' 中按顺序取出 task_id 和 worker_id，并组成 sample_action
                for task_id, worker_id in zip(action['tasks'], action['workers']):
                    sample_action = (task_id, worker_id, 1.0)
                    total_schedule.append(sample_action)

                    diff = env.problem.diff[task_id-1]
                    if worker_id in [0, 1]:
                        accuracy = robot_accuracies[worker_id][diff - 1]
                    elif worker_id in [2, 3]:
                        accuracy = human_accuracies[worker_id](diff)
                    else:
                        raise ValueError("Invalid resource ID")

                    total_accuracy += accuracy

                # Acquire the likelihood of the accuracy of the lasted scheduling
                average_accuracy = total_accuracy / len(total_schedule)

                success, reward, done, makespan = multi_round_envs[i_b].step(total_schedule,
                                                                             evaluate=estimator,
                                                                             human_noise=human_noise)
                reward = (lambda_val * reward + (
                            1 - lambda_val) * average_accuracy * makespan / discount) if success else reward

                makespan_accuracy_record.append([makespan, average_accuracy, success])
                # print('The result of {}-th iteration is: Action:{}, Reward:{}, step_success:{}'.format(i, vAction, reward, step_success))
                optimizer.register(context, action, reward)

            # Get the minimum makespan and success value
            min_value_makespan = min(makespan_accuracy_record, key=lambda x: x[0])
            min_makespan_record.append(min_value_makespan)

            # Get the maximum accuracy and success value
            max_value_accuracy = max(makespan_accuracy_record, key=lambda x: x[1])
            max_accuracy_record.append(max_value_accuracy)

            # # Save the makespan result
            # makespan_metrics_f = open(fold_makespan, 'a')
            # np.savetxt(makespan_metrics_f, np.array(makespan_record))
            # makespan_metrics_f.close()

            # Get the optimization results
            vReward = []
            res = optimizer.res
            for i in range(num_Iterations):
                vReward.append(res['reward'][i])
            reward_metrics_f = open(folder_reward, 'a')
            np.savetxt(reward_metrics_f, np.array(vReward))
            reward_metrics_f.close()

            end_t = time.time()
            print('Run Time: {:.3f} s'.format(end_t - start_t))
            # Plot the reward results and save them to a file
            plt.figure()
            plt.plot(vReward)
            plt.xlabel('Iterations')
            plt.ylabel('Reward')
            # 保存图形到指定路径
            plt.savefig(save_reward_fig)

    # Save the minimum makespan result
    min_makespan_metrics_f = open(file_min_makespan, 'a')
    np.savetxt(min_makespan_metrics_f, np.array(min_makespan_record))
    min_makespan_metrics_f.close()

    # Save the maximum accuracy result
    max_accuracy_metrics_f = open(file_max_accuracy, 'a')
    np.savetxt(max_accuracy_metrics_f, np.array(max_accuracy_record))
    max_accuracy_metrics_f.close()









