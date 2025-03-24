"""
Created on Mon Sep  9 13:43:13 2021

@author: baltundas3

Multi Round Scheduling Environment based on OpenAI Code
"""
import gym

import sys
import copy

# setting path
sys.path.append('../')
from env.scheduling_env import SchedulingEnv
from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam
from env.worker import *

class MultiRoundSchedulingEnv(gym.Env):
    """MultiRound Scheduling Environment using OpenAI Gym Environment used for simulation and testing of the ScheduleNet Architecture and Variants.
    
    This Environment models the job shop scheduling problem as a multi-agent problem.

    This Environment is designed to take actions of schedule plans for each time step, update the workers based on the given schedule and produce a new state based on this.
    
    Args:
        problem (MRCProblem): MRCProblem that is passed to the ScheduleNet Environment for
    Returns:
        [type]: [description]
    """
    def __init__(self, problem: MRCProblem, team: HybridTeam, max_num_rounds: int = 500):
        self.problem = problem
        self.team = team
        self.max_num_rounds = max_num_rounds

        # self.team.sample_humans(self.max_num_rounds) # cache human samples for consistency, comment out for stochastic behavior
        self.round = 0   # day for schedule
        
        
    def check_legal(self, schedule):

        pass

    """Reset the Environment to the Starting Conditions

    Returns:
        [type]: [description]
    """
    def reset(self):
    
        self.round = 0
        # reset the team
        self.team.reset()

    def step(self, action, human_learning: bool = False, evaluate = False, human_noise = False, estimator_noise = False):
        """Takes a step using the list of task

        Args:
            schedule (list(tuple(task_id, worker_id, diff))): index of the action being taken, in this case,
            the Schedule that is produced for this timestep
        Returns:
            state (): Worker Team that is updated by the Schedule passed at the step.
            reward (float)
            done (boolean)
            info
        """
    
        reward = 0.0
        # lambda_val = 0.2    # Setup the lambda value
        done = False

        # # Define likelihood of accuracy for humans and machines
        # robot_accuracies = {0: (0.5, 0.4, 0.3), 1: (0.6, 0.5, 0.3)}
        # human_accuracies = {2: lambda diff: 0.6 + 0.3 / diff, 3: lambda diff: 0.8 + 0.2 / diff}


        schedule_env = None
        if not human_noise:  # Then evalutation and actual environment produce the same result
            schedule_env = self.get_single_round()
        elif evaluate:
            schedule_env = self.get_estimate_environment(estimator_noise)
        else:
            schedule_env = self.get_actual_environment(human_noise)
        
        # 获取机器人和人类工作者的数量
        num_robots = schedule_env.team.num_robots
        num_humans = schedule_env.team.num_humans
        num_workers = num_robots + num_humans
    
        # 动态创建robot_accuracies字典
        robot_accuracies = {}
        for i in range(num_robots):
            if i % 2 == 0:  # 为不同类型的机器人设置不同的精度值
                robot_accuracies[i] = (0.5, 0.4, 0.3)
            else:
                robot_accuracies[i] = (0.6, 0.5, 0.3)
    
        # 动态创建human_accuracies字典
        human_accuracies = {}
        for i in range(num_robots, num_workers):
            if (i - num_robots) % 2 == 0:  # 为不同类型的人类工作者设置不同的精度函数
                human_accuracies[i] = lambda diff: 0.6 + 0.3 / diff
            else:
                human_accuracies[i] = lambda diff: 0.8 + 0.2 / diff
        
        total_reward = 0.0
        total_accuracy = 0.0   # Define the total accuracy of the current schedule
        success = True

        # # Get the numbers of robot and human
        # num_robots = schedule_env.team.num_robots
        # num_workers = schedule_env.team.num_robots + schedule_env.team.num_humans

        # Iterate through each action and take a single step in the SingleRoundEnv
        for step in range(len(action)):
            task_id, worker_id, diff = action[step]
            # print(task_id, worker_id, diff)
            step_success, reward, done, info = schedule_env.step(task_id, worker_id, diff)
            success = (success and step_success)
            if not success: # infeasible schedule
                # Get the ramaining tasks that need to be completed
                remaining_task_idx = [task - 1 for task, worker, diff in action[step:]]
                # print(remaining_task_idx)
                # Get the duration of the completion of the tasks at maximum cost by a single user
                reward = schedule_env.get_infeasible_reward(remaining_task_idx)

            diff = schedule_env.problem.diff[task_id-1]
            if worker_id in range(num_robots):
                accuracy = robot_accuracies[worker_id][diff - 1]
            elif worker_id in range(num_robots, num_workers):
                accuracy = human_accuracies[worker_id](diff)
            else:
                raise ValueError("Invalid resource ID")

            total_accuracy += accuracy
            total_reward += reward
            if done:
                break
        # If the steps are infeasible
        # TODO: On Infeasible Schedule Generation, updating worker is a problem.
        # Update Workers
        if human_learning and success:
            for task_id, worker_id, diff in action:
                worker = self.team.get_worker(worker_id)
                # print(worker.type, type(worker.type))
                if worker.type == WorkerType.HUMAN:
                # if worker.type == "Human":
                    worker.add_task(task_id)

        # Get the Total Makespan
        makespan = schedule_env.min_makespan

        # Acquire the likelihood of the accuracy of the lasted scheduling
        average_accuracy = total_accuracy / len(action)
        
        if not success:
            makespan = schedule_env.problem.max_deadline
        # state = action
        # update time and check if done
        # self.round += 1
        # if self.round == self.max_num_rounds:
        #     done = True
        return success, total_reward, done, makespan, average_accuracy

    def get_single_round(self):
        schedule_env = SchedulingEnv(problem=self.problem, team=self.team)
        return schedule_env
    
    def get_actual_environment(self, human_noise=False):
        new_problem = copy.deepcopy(self.problem)
        schedule_env = SchedulingEnv(problem=new_problem, team=self.team, sample_humans=False)
        return schedule_env
    
    def get_estimate_environment(self, est_noise = False):
        new_problem = copy.deepcopy(self.problem)
        # ---------------------------------重构问题---------------------------------------
        new_problem.refactor_problem_from_team(self.team, estimate=True, noise=est_noise)
        # ---------------------------------Modified-------------------------------------
        schedule_env = SchedulingEnv(problem=new_problem, team=self.team, sample_humans=False)
        return schedule_env
        
    def render(self):
        """ [OPTIONAL] Renders model
            TODO: add compatibility for the JssEnv for gif generation for the change of schedule over time, showing optimization.
        
        Returns:
            [type]: [description]
        """
        pass
