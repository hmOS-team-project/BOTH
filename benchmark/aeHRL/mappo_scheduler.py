# -*- coding: utf-8 -*-
"""
Created on Wdn July 10, 2024

@author: Hui

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.optim.lr_scheduler import MultiStepLR
from utils import ReplayMemory, Transition, action_helper_rollout
from utils import hetgraph_node_helper, build_hetgraph
from evolutionary_algorithm import random_gen_schedules, swap_task_allocation, generate_evolution
from benchmark.aeHRL.mappo_policy import MAPPOGRUPolicyNet

class MAPPOScheduler():
    """ Multi-Agent Proximal Policy Optimization Scheduler
    """
    def __init__(self, device=torch.device("cpu"), 
                 gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, lmbda=0.95,
                 milestones=[30, 80], lr_gamma=0.1, 
                 entropy_coefficient=0.0, 
                 eps_clip=0.2,
                 selection_mode='sample',
                 verbose='none',
                 num_agents=2):  # 假设有2个智能体（任务选择器和工作者分配器）
        
        self.device = device
        self.num_agents = num_agents
        
        # 为每个智能体创建单独的策略网络
        # agent_models = [GRUPolicyNet(selection_mode=selection_mode, verbose=verbose, device=device).to(device) for _ in range(num_agents)]
        # 这里我们继续使用原始的GRUPolicyNet模型，但在MAPPO中这部分通常会修改为多个独立的策略网络
        self.model = MAPPOGRUPolicyNet(selection_mode=selection_mode, verbose=verbose, device=device).to(device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.weight_decay = weight_decay
        self.entropy_coefficient = entropy_coefficient
        self.eps_clip = eps_clip

        self.eps = np.finfo(np.float32).eps.item()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=lr_gamma)
        
    def load_checkpoint(self, trained_checkpoint, retain_old=True):
        cp = torch.load(trained_checkpoint, map_location='cuda:0')
        self.model.load_state_dict(cp['policy_net_state_dict'])
        self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        if retain_old:
            self.lr_scheduler.load_state_dict(cp['scheduler_state_dict'])
        else:
            relevant_lr_scheduler_state_dict = self.lr_scheduler.state_dict()
            relevant_lr_scheduler_state_dict['last_epoch'] = cp['scheduler_state_dict']['last_epoch']
            relevant_lr_scheduler_state_dict['_step_count'] = cp['scheduler_state_dict']['_step_count']
            self.lr_scheduler.load_state_dict(relevant_lr_scheduler_state_dict)
            print(self.lr_scheduler.state_dict())
        return cp['i_batch'] + 1

    # 其他辅助方法保持不变
    def get_variables(self, env):
        # 与原PPOScheduler相同
        num_tasks = env.problem.num_tasks
        num_robots = env.team.num_robots
        num_humans = env.team.num_humans
        curr_g = copy.deepcopy(env.halfDG)
        curr_partials = copy.deepcopy(env.partials)
        curr_partialw = copy.deepcopy(env.partialw)
        durs = copy.deepcopy(env.dur)
        act_robot = 0
        unsch_tasks = np.array(action_helper_rollout(num_tasks, curr_partialw), dtype=np.int64)
        g = build_hetgraph(curr_g, num_tasks, num_robots, num_humans, durs, curr_partials, unsch_tasks)
        num_actions = len(unsch_tasks)
        feat_dict = hetgraph_node_helper(curr_g.number_of_nodes(), 
                                       curr_partialw, 
                                       curr_partials,
                                       durs, 
                                       num_robots + num_humans, num_actions)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)
        
        return g, feat_dict_tensor, unsch_tasks

    def select_action(self, env, genetic=False):
        """Generate a Schedule as Action for a MultiRoundEnvironment
        Args:
            env (SingleRoundScheduler): Single-Round Scheduler Environment
        """
        # No Grad
        with torch.no_grad():
            schedule = self.model(env)
        if genetic:
            schedule = self.run_genetic(schedule, env)
        return schedule
    
    def run_genetic(self, schedule, env, generation=10, base_population=90, new_mutation=10, new_random=10):
        # 保持与原PPOScheduler相同的遗传算法逻辑
        # ...与原代码相同...
        unscheduled_tasks = env.get_unscheduled_tasks()
        if len(unscheduled_tasks) > 0:
            worker = env.problem.get_worst_worker(unscheduled_tasks-1)
            for u_task in unscheduled_tasks:
                schedule.append([u_task, worker, 1.0])
                
        env.reset()
        new_random_schedules = random_gen_schedules([schedule], env.team, base_population + new_random - 1)
        if len(new_random_schedules) == 0:
            new_random_schedules = generate_evolution([schedule], base_population + new_random - 1, [0])
        new_mutations = generate_evolution([schedule], new_mutation, [0])
        new_generation = [schedule] + new_random_schedules + new_mutations
        
        scores = [[], []]
        schedules = [[], []]
        infeasible_idx_sorted = []
        feasible_idx_sorted = []
        
        for gen in range(generation):
            for j in range(len(new_generation)):
                env.reset()
                schedule_i = new_generation[j]
                rt = False
                for step in schedule_i:
                    rt, reward, done, _ = env.step(step[0], step[1], step[2])
                    if rt == False:
                        scores[0].append(env.problem.max_deadline)
                        schedules[0].append(schedule_i)
                        break
                if rt:
                    scores[1].append(env.min_makespan)
                    schedules[1].append(schedule_i)
                    
            if len(scores[0]) != 0:
                infeasible_idx_sorted = np.argsort(scores[0])
            if len(scores[1]) != 0:
                feasible_idx_sorted = np.argsort(scores[1])
                
            if gen < generation - 1:
                feasible_top_n = min(base_population, len(scores[1]))
                infeasible_top_n = max(0, base_population - len(scores[1]))
                schedules[1] = np.array(schedules[1], dtype=int)[feasible_idx_sorted][:feasible_top_n].tolist() 
                scores[1] = np.array(scores[1], dtype=int)[feasible_idx_sorted][:feasible_top_n].tolist()
                if infeasible_top_n > 0:
                    schedules[0] = np.array(schedules[0], dtype=int)[infeasible_idx_sorted][:infeasible_top_n].tolist()
                    scores[0] = np.array(scores[0], dtype=int)[infeasible_idx_sorted][:infeasible_top_n].tolist()
                else:
                    schedules[0] = []
                    scores[0] = []
                    
                random_schedules = random_gen_schedules(schedules[0] + schedules[1], env.team, new_random)
                if len(random_schedules) == 0:
                    random_schedules = swap_task_allocation(schedules[0] + schedules[1], new_random)
                new_mutation_schedules = swap_task_allocation(schedules[0] + schedules[1], new_mutation)
                if len(new_mutation_schedules) == 0:
                    new_mutation_schedules = random_gen_schedules(schedules[0] + schedules[1], env.team, new_mutation)
                new_generation = random_schedules + new_mutation_schedules
                
        if len(feasible_idx_sorted) != 0:
            return schedules[1][0]
        else:
            return schedules[0][0]
        return []
    
    def initialize_batch(self, batch_size):
        # 初始化批处理缓冲区
        self.batch_saved_t_log_probs = [[] for i in range(batch_size)]
        self.batch_saved_t_entropy = [[] for i in range(batch_size)]
        self.batch_saved_w_log_probs = [[] for i in range(batch_size)]
        self.batch_saved_w_entropy = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]

    def batch_select_action(self, env, i_b):
        """Batch Selection of Action for Multi-Agent PPO
        Args:
            env: Environment
            i_b: Batch index
        """
        # 重置模型日志概率缓冲区
        self.model.task_classifier.saved_log_probs = []
        self.model.task_classifier.saved_entropy = []
        self.model.worker_classifier.saved_log_probs = []
        self.model.worker_classifier.saved_entropy = []
        
        # 生成调度
        schedule = self.model(env)
        
        # 添加日志概率到批处理数据
        self.batch_saved_t_log_probs[i_b].append(self.model.task_classifier.saved_log_probs[-1])
        self.batch_saved_w_log_probs[i_b].append(self.model.worker_classifier.saved_log_probs[-1])
        
        # 添加熵到批处理数据
        self.batch_saved_t_entropy[i_b].append(self.model.task_classifier.saved_entropy[-1])
        self.batch_saved_w_entropy[i_b].append(self.model.worker_classifier.saved_entropy[-1])
        
        return schedule

    def batch_finish_episode(self, batch_size, num_rounds=1, max_norm=0.75):
        '''
        MAPPO批处理版本，处理多个智能体的学习
        '''
        batch_policy_loss = [[] for i in range(batch_size)]
        batch_total_loss = []
        
        # 零填充提前终止的剧集
        batch_returns = torch.zeros(batch_size, num_rounds).to(self.device)
        
        # 1. 计算每个剧集的总回报
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            batch_returns[i_b][:r_size] = self.batch_r(i_b)          

        # 2. 计算基于时间的基线值
        batch_baselines = torch.mean(batch_returns, dim=0)

        # 3. 计算每个转换的优势
        batch_advs = batch_returns - batch_baselines
        
        # 4. 在批处理内归一化优势
        eps = np.finfo(np.float32).eps.item()
        adv_mean = batch_advs.mean()
        adv_std = batch_advs.std()
        batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)
        
        # 5. 计算每个批处理中每个剧集的损失
        for i_b in range(batch_size):
            for round_count in range(num_rounds):
                # 检查提前终止前的转换
                if round_count < len(self.batch_saved_t_log_probs[i_b]):
                    # 多智能体PPO分别计算每个智能体的策略损失
                    
                    # 任务选择器（智能体1）
                    t_log_prob = self.batch_saved_t_log_probs[i_b][round_count]
                    t_entropy = self.batch_saved_t_entropy[i_b][round_count]
                    t_adv_n = batch_advs_norm[i_b][round_count]
                    t_old_log_prob = t_log_prob.detach()
                    t_ratio = torch.exp(t_log_prob - t_old_log_prob)
                    t_surr1 = t_ratio * t_adv_n
                    t_surr2 = torch.clamp(t_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * t_adv_n
                    t_policy_loss = -torch.min(t_surr1, t_surr2) - self.entropy_coefficient * t_entropy
                    
                    # 工作者分配器（智能体2）
                    w_log_prob = self.batch_saved_w_log_probs[i_b][round_count]
                    w_entropy = self.batch_saved_w_entropy[i_b][round_count]
                    w_adv_n = batch_advs_norm[i_b][round_count]
                    w_old_log_prob = w_log_prob.detach()
                    w_ratio = torch.exp(w_log_prob - w_old_log_prob)
                    w_surr1 = w_ratio * w_adv_n
                    w_surr2 = torch.clamp(w_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * w_adv_n
                    w_policy_loss = -torch.min(w_surr1, w_surr2) - self.entropy_coefficient * w_entropy
                    
                    # 总策略损失是两个智能体策略损失的和
                    policy_loss = t_policy_loss + w_policy_loss
                    batch_policy_loss[i_b].append(policy_loss)

            if len(batch_policy_loss[i_b]) > 0:
                batch_total_loss.append(torch.stack(batch_policy_loss[i_b]).sum())

        # 重置梯度
        self.optimizer.zero_grad()
        
        # 对所有批次求和
        total_loss = torch.stack(batch_total_loss).sum()
        loss_np = total_loss.data.cpu().numpy()
        
        # 执行反向传播
        total_loss.backward()
        
        # 执行梯度裁剪
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
        
        # 执行优化步骤
        self.optimizer.step()
        
        # 重置奖励和动作缓冲区
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_saved_t_log_probs[i_b][:]
            del self.batch_saved_t_entropy[i_b][:]
            del self.batch_saved_w_log_probs[i_b][:]
            del self.batch_saved_w_entropy[i_b][:]
        return loss_np
    
    def batch_r(self, i_b):
        '''
        计算每个剧集的总回报
        '''
        R = 0.0
        returns = []

        for rw in self.batch_rewards[i_b][::-1]:
            R = rw + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        return returns

    def adjust_lr(self, metrics=0.0):
        '''
        使用lr_scheduler调整学习率
        '''
        self.lr_scheduler.step()
    
