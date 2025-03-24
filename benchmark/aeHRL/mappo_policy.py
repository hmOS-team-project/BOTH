import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy

from hetnet import HybridScheduleNet
from graph.gru_layer import GRU_CellLayer
from graph.classifier import Classifier
from utils import action_helper_rollout, hetgraph_node_helper, build_hetgraph

class MAPPOAgent(nn.Module):
    """基础智能体类，被任务选择器和工作者分配器继承"""
    def __init__(self, selection_mode='sample', 
                 pad_value=True, 
                 task_filtering=True,
                 verbose='none',
                 device=torch.device("cpu"),
                 agent_type='task'):  # 'task' 或 'worker'
        super(MAPPOAgent, self).__init__()
        self.device = device
        self.selection_mode = selection_mode
        self.task_filtering = task_filtering
        self.pad_value = pad_value
        self.agent_type = agent_type
        
        self.state_dim = 1
        self.hidden_dim = 32
        
        self.worker_embedding_size = (self.state_dim + 1) * self.hidden_dim
        self.task_embedding_size = (self.state_dim + 1 + 1) * self.hidden_dim
        
        self._init_gnn()
        self._init_gru()
        if agent_type == 'task':
            self.classifier = Classifier(self.task_embedding_size, 1)
        else:  # worker
            self.classifier = Classifier(self.worker_embedding_size, 1)
    
    def _init_gnn(self):
        in_dim = {'task': 6, 'worker': 1, 'state': 4}
        hid_dim = {'task': 64, 'worker': 64, 'human': 64, 'state': 64}
        out_dim = {'task': 32, 'worker': 32, 'state': 32}  

        cetypes = [('task', 'temporal', 'task'), ('task', 'assigned_to', 'worker'),
                  ('worker', 'com', 'worker'), ('task', 'tin', 'state'),
                  ('worker', 'win', 'state'), ('state', 'sin', 'state'),
                  ('task', 'take_time', 'worker'), ('worker', 'use_time', 'task')]
        
        self.gnn = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(self.device)
    
    def _init_gru(self):
        self.gru_cell_state = GRU_CellLayer(self.hidden_dim * 2, self.hidden_dim, self.device)
        
        if self.agent_type == 'task':
            self.gru_cell_task = GRU_CellLayer(self.hidden_dim * 2, self.hidden_dim, self.device)
        else:  # worker
            self.gru_cell_worker = GRU_CellLayer(self.hidden_dim * 2, self.hidden_dim, self.device)
    
    def merge_embeddings(self, embedding1, embedding2, num_items=None):
        embedding2_repeated = embedding2.repeat(num_items, 1)
        merged = torch.cat((embedding1, embedding2_repeated), dim=1)
        return merged
    
    def get_variables(self, env):
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
        g = g.to(self.device)
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

    def filter_tasks(self, wait, unscheduled_tasks):
        unfiltered_task_set = set(unscheduled_tasks)
        to_filter = set([])
        for si, fj, dur in wait:
            if fj in unfiltered_task_set and si not in to_filter:
                to_filter.add(si)
        filtered_tasks = list(unfiltered_task_set - to_filter)
        filtered_tasks.sort()
        return filtered_tasks

class TaskSelectorAgent(MAPPOAgent):
    def __init__(self, selection_mode='sample', pad_value=True, task_filtering=True, 
                verbose='none', device=torch.device("cpu")):
        super(TaskSelectorAgent, self).__init__(selection_mode, pad_value, task_filtering, 
                                              verbose, device, agent_type='task')
        
    def select_task(self, task_out, unscheduled_tasks):
        task_probs = F.softmax(task_out, dim=-1)
        m = Categorical(task_probs)
        idx = 0
        if self.selection_mode == 'sample':
            idx = m.sample()
        elif self.selection_mode == 'argmax':
            idx = torch.argmax(m.probs)
        self.classifier.saved_log_probs[-1] += m.log_prob(idx)
        self.classifier.saved_entropy[-1] += m.entropy().mean() / len(unscheduled_tasks)
        task_id = unscheduled_tasks[idx.item()]
        return task_id
        
    def forward(self, task_output, state_output, worker_embedding, unscheduled_tasks, feasible_tasks=None):
        """选择一个任务 - 优化版本，直接接收已计算的嵌入"""
        if feasible_tasks is None:
            feasible_tasks = unscheduled_tasks
            
        # 不再需要通过GNN获取任务嵌入，直接使用传入的task_output
        # task中包含第一个虚拟任务，所以要从第二个开始
        indices = [t + 1 for t in unscheduled_tasks]
        task_output_ = task_output[indices]
        
        # 准备分类器日志
        self.classifier.saved_log_probs.append(0)
        self.classifier.saved_entropy.append(0)
        
        # 如需过滤任务
        if self.task_filtering:
            feasible_task_output_ = task_output_.clone()
            possible_tasks = np.in1d(np.array(unscheduled_tasks), np.array(feasible_tasks)).nonzero()[0]
            indices = torch.tensor(np.array(possible_tasks), device=self.device)
            feasible_task_output = torch.index_select(feasible_task_output_, 0, indices.to(self.device, torch.int64))
        else:
            feasible_task_output = task_output_.clone()
        
        # 合并嵌入 - 先添加状态嵌入
        task_relevant_embedding = self.merge_embeddings(feasible_task_output, state_output, len(feasible_tasks))
        # 再添加工作者嵌入
        task_worker_embedding = self.merge_embeddings(task_relevant_embedding, worker_embedding, len(feasible_tasks))
        
        # 获取任务概率分布
        task_probs = self.classifier(task_worker_embedding)
        
        # 选择任务
        task_id = self.select_task(task_probs, feasible_tasks)
        return task_id

class WorkerAllocatorAgent(MAPPOAgent):
    def __init__(self, selection_mode='sample', pad_value=True, task_filtering=True, 
                verbose='none', device=torch.device("cpu")):
        super(WorkerAllocatorAgent, self).__init__(selection_mode, pad_value, task_filtering, 
                                                 verbose, device, agent_type='worker')
    
    def select_worker(self, worker_out):
        worker_probs = F.softmax(worker_out, dim=-1)
        m = Categorical(worker_probs)
        worker_id = 0
        if self.selection_mode == 'sample':
            worker_id = m.sample()
        elif self.selection_mode == 'argmax':
            worker_id = torch.argmax(m.probs)
        self.classifier.saved_log_probs[-1] += m.log_prob(worker_id)
        self.classifier.saved_entropy[-1] += m.entropy().mean() / worker_probs.size(0)
        return worker_id.item()
        
    def forward(self, worker_output, state_output):
        """为选定的任务分配一个工作者 - 优化版本，直接接收已计算的嵌入"""
        # 不再需要通过GNN获取工作者嵌入，直接使用传入的worker_output
        
        # 准备分类器日志
        self.classifier.saved_log_probs.append(0)
        self.classifier.saved_entropy.append(0)
        
        # 合并嵌入
        worker_relevant_embedding = self.merge_embeddings(worker_output, state_output, worker_output.size(0))
        
        # 获取工作者概率分布
        worker_probs = self.classifier(worker_relevant_embedding)
        
        # 选择工作者
        worker_id = self.select_worker(worker_probs)
        return worker_id


class WorkerAllocatorAgent(MAPPOAgent):
    def __init__(self, selection_mode='sample', pad_value=True, task_filtering=True, 
                verbose='none', device=torch.device("cpu")):
        super(WorkerAllocatorAgent, self).__init__(selection_mode, pad_value, task_filtering, 
                                                 verbose, device, agent_type='worker')
    
    def select_worker(self, worker_out):
        worker_probs = F.softmax(worker_out, dim=-1)
        m = Categorical(worker_probs)
        worker_id = 0
        if self.selection_mode == 'sample':
            worker_id = m.sample()
        elif self.selection_mode == 'argmax':
            worker_id = torch.argmax(m.probs)
        self.classifier.saved_log_probs[-1] += m.log_prob(worker_id)
        self.classifier.saved_entropy[-1] += m.entropy().mean() / worker_probs.size(0)
        return worker_id.item()
        
    def forward(self, worker_output, state_output):
        """为选定的任务分配一个工作者 - 优化版本，直接接收已计算的嵌入"""
        # 不再需要通过GNN获取工作者嵌入，直接使用传入的worker_output
        
        # 准备分类器日志
        self.classifier.saved_log_probs.append(0)
        self.classifier.saved_entropy.append(0)
        
        # 合并嵌入
        worker_relevant_embedding = self.merge_embeddings(worker_output, state_output, worker_output.size(0))
        
        # 获取工作者概率分布
        worker_probs = self.classifier(worker_relevant_embedding)
        
        # 选择工作者
        worker_id = self.select_worker(worker_probs)
        return worker_id

class MAPPOGRUPolicyNet(nn.Module):
    """整合两个智能体的策略网络"""
    def __init__(self, selection_mode='sample', pad_value=True, task_filtering=True, 
                verbose='none', device=torch.device("cpu")):
        super(MAPPOGRUPolicyNet, self).__init__()
        
        self.device = device
        self.selection_mode = selection_mode
        self.task_filtering = task_filtering
        self.pad_value = pad_value
        self.verbose = verbose
        
        # 创建两个智能体
        self.task_selector = TaskSelectorAgent(selection_mode, pad_value, task_filtering, verbose, device)
        self.worker_allocator = WorkerAllocatorAgent(selection_mode, pad_value, task_filtering, verbose, device)
        
        # 为方便访问
        self.task_classifier = self.task_selector.classifier
        self.worker_classifier = self.worker_allocator.classifier
        self.gnn = self.task_selector.gnn  # 共享GNN
        
        # GRU用于更新状态
        self.state_dim = 1
        self.hidden_dim = 32
        self.gru_cell_state = GRU_CellLayer(self.hidden_dim * 2, self.hidden_dim, self.device)
        
    def get_variables(self, env):
        return self.task_selector.get_variables(env)
        
    def filter_tasks(self, wait, unscheduled_tasks):
        return self.task_selector.filter_tasks(wait, unscheduled_tasks)
    
    def forward(self, env):
        """生成完整的调度计划"""
        schedule = []
    
        # 重置环境并获取初始状态
        env.reset()
        g, feat_dict, unscheduled_tasks = self.get_variables(env)
    
        # 只计算一次GNN输出
        outputs = self.gnn(g, feat_dict)
        task_output = outputs['task']
        state_output = outputs['state']
        worker_output = outputs['worker']
    
        # 初始化状态
        state_hidden = self.gru_cell_state.init_hidden(self.state_dim)
    
        # 保存任务和工作者嵌入
        # task_output_ = task_output.clone()  # 存储完整副本
    
        # 创建新的日志概率和熵
        self.task_classifier.saved_log_probs.append(0)
        self.task_classifier.saved_entropy.append(0)
        self.worker_classifier.saved_log_probs.append(0)
        self.worker_classifier.saved_entropy.append(0)
    
        while len(unscheduled_tasks) > 0:
            # 过滤可行任务
            feasible_tasks = unscheduled_tasks.copy()
            if self.task_filtering:
                feasible_tasks = self.filter_tasks(env.problem.wait, unscheduled_tasks)
        
            # 使用工作者分配器选择工作者 - 传递已计算的嵌入
            worker_id = self.worker_allocator(worker_output, state_output)
            chosen_worker_embedding = worker_output[worker_id, :].unsqueeze(dim=0)
        
            # 处理只有一个可行任务的情况
            if len(feasible_tasks) == 1:
                task_id = feasible_tasks[0]
                action = (task_id, worker_id, 1.0)
                schedule.append(action)
                index = np.where(unscheduled_tasks == task_id)
                index = index[0][0]
                unscheduled_tasks = np.delete(unscheduled_tasks, index)
                continue
        
            # 使用任务选择器选择任务 - 传递已计算的嵌入
            task_id = self.task_selector(task_output, state_output, chosen_worker_embedding, unscheduled_tasks, feasible_tasks)
        
            # 更新调度计划
            action = [task_id, worker_id, 1.0]
            schedule.append(action)
        
            # 从未调度任务中移除已选任务
            index = np.where(unscheduled_tasks == task_id)
            index = index[0][0]
            unscheduled_tasks = np.delete(unscheduled_tasks, index)
        
            # 获取选中任务的嵌入（需要考虑+1，因为task_output包括虚拟任务）
            chosen_task_embedding = task_output[task_id + 1].unsqueeze(dim=0)
        
            # 更新状态嵌入
            task_worker_embedding = torch.cat((chosen_task_embedding, chosen_worker_embedding), dim=1)
            state_output, state_hidden = self.gru_cell_state(task_worker_embedding, state_hidden)
        
            # 更新工作者嵌入
            chosen_worker_hidden = worker_output[worker_id].unsqueeze(dim=0)
            worker_out_replacement, _ = self.worker_allocator.gru_cell_worker(task_worker_embedding, chosen_worker_hidden)
            worker_output_tmp = torch.cat((worker_output[:worker_id], worker_out_replacement, worker_output[worker_id+1:]), dim=0)
            worker_output = worker_output_tmp
        
            # # 更新图和特征 - 仅在必要时更新
            # if len(unscheduled_tasks) > 0 and env.has_changed():
            #     g, feat_dict, _ = self.get_variables(env)
            #     # 如果环境发生变化，重新计算GNN嵌入
            #     outputs = self.gnn(g, feat_dict)
            #     task_output = outputs['task']
            #     worker_output = outputs['worker']
            #     # state_output通过GRU更新，不需要重新计算
    
        return schedule