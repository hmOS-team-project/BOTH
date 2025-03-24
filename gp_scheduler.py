from env.scheduling_env import SchedulingEnv
from context_encoder import HetGATNetwork
import numpy as np
import random
import torch

class GPScheduler():
    """
    Gaussian Process based Scheduler
    """
    def __init__(self, env: SchedulingEnv, device = torch.device("cuda")):
        self.device = device
        self.model = HetGATNetwork(device=device).to(self.device)
        self.env = env

    # 用于检查排序是否满足wait序列的函数
    def is_valid_sort(self, sorting, wait):
        for pair in wait:
            after_task, before_task, _ = pair
            before_task_idx = np.where(sorting == before_task)[0]
            after_task_idx = np.where(sorting == after_task)[0]
            if before_task_idx > after_task_idx:
                return False
        return True

    # 定义初始的排序函数
    def custom_sort(self, arr, priority_tasks, order_constraints):
        # 对具有最高优先级的任务进行排序
        sorted_arr = sorted(arr, key=lambda x: priority_tasks.index(x) if x in priority_tasks else float('inf'))

        # 处理先后顺序约束
        for constraint in order_constraints:
            task_after, task_before, _ = constraint
            if task_before in sorted_arr and task_after in sorted_arr:
                index_before = sorted_arr.index(task_before)
                index_after = sorted_arr.index(task_after)
                if index_before > index_after:
                    # 交换任务的位置，确保满足约束条件
                    sorted_arr[index_before], sorted_arr[index_after] = sorted_arr[index_after], sorted_arr[
                        index_before]

        return np.array(sorted_arr)

    def find_valid_task_orders(self, unscheduled_tasks, ddl_list, wait):
        # 生成200组排序数组
        num_samples = 100

        # 使用sorted函数按照deadline进行排序
        sorted_deadline_tasks = sorted(ddl_list, key=lambda x: x[1])
        # 提取排序后的任务列表
        sorted_task_ids = [task[0] for task in sorted_deadline_tasks]

        sorted_arrays = []
        for _ in range(num_samples):
            np.random.shuffle(unscheduled_tasks)
            sorted_array = self.custom_sort(unscheduled_tasks, sorted_task_ids, wait)
            sorted_arrays.append(sorted_array)

        return sorted_arrays

        # # 生成100种不同的排序结果
        # valid_sortings = []
        # while len(valid_sortings) < 100:
        #     random.shuffle(unscheduled_tasks)
        #     if self.is_valid_sort(unscheduled_tasks, wait):
        #         valid_sortings.append(unscheduled_tasks.copy())
        # return valid_sortings

    def generate_worker_order(self, num_workers, num_tasks, num_samples=100):
        worker_order = np.empty((num_samples, num_tasks), dtype=int)

        for i in range(num_samples):
            # 从 worker_id 中随机选择 num_tasks 个元素，构成一个长度为 num_tasks 的数组
            order = random.choices(range(num_workers), k=num_tasks)
            # 将该数组放入 worker_order 中的第 i 行
            worker_order[i, :] = order
        return worker_order

    def generate_action(self):
        # env.reset()  # Reset Single Round Scheduler
        _, _, unscheduled_tasks = self.model.get_variables(self.env)

        num_workers = self.env.team.num_robots + self.env.team.num_humans
        num_tasks = self.env.problem.num_tasks

        valid_task_orders = self.find_valid_task_orders(unscheduled_tasks, self.env.problem.ddl, self.env.problem.wait)
        worker_order = self.generate_worker_order(num_workers, num_tasks, num_samples=100)

        # 仅返回前100个排序（如果有的话）
        if len(valid_task_orders) >= 100:
            return np.array(valid_task_orders[:100]), worker_order
        else:
            return np.array(valid_task_orders), worker_order

    # 定义 Mean Pooling 函数
    def mean_pooling(self, x):
        return torch.mean(x, dim=0)

    def generate_context(self, checkpoint_folder):
        # Generate a schedule
        g, feat_dict, _ = self.model.get_variables(self.env)
        
        # Get the output embeddings from the GNN
        self.model.load_checkpoint(checkpoint_folder)
        raw_outputs = self.model.gnn(g, feat_dict)

        # 对每个字典的输出应用 Mean Pooling
        task_output = self.mean_pooling(raw_outputs['task']).detach().numpy() + np.random.normal(0, 0.2, 64)
        worker_output = self.mean_pooling(raw_outputs['worker']).detach().numpy() + np.random.normal(0, 0.2, 64)
        state_output = self.mean_pooling(raw_outputs['state']).detach().numpy() + np.random.normal(0, 0.3, 64)

        # # 将每个key对应的张量转换为一维的NumPy数组
        # task_array = raw_outputs['task'].view(-1).detach().numpy()
        # worker_array = raw_outputs['worker'].view(-1).detach().numpy()
        # state_array = raw_outputs['state'].view(-1).detach().numpy()
        # 创建新的output字典
        # new_output = {'task': task_array, 'worker': worker_array, 'state': state_array}
        new_output = {'task': task_output, 'worker': worker_output, 'state': state_output}

        return new_output

