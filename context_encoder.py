import torch
import copy
import numpy as np
import torch.nn as nn
from env.scheduling_env import SchedulingEnv
from hybridnet.utils import ReplayMemory, Transition, action_helper_rollout
from hybridnet.utils import hetgraph_node_helper, build_hetgraph
from hybridnet.hetnet import HybridScheduleNet

# 定义图注意力网络HetGAT
class HetGATNetwork(nn.Module):
    def __init__(self):
        super(HetGATNetwork, self).__init__()
        self._init_gat()

    def _init_gat(self):
        """
        Initialize Graph Neural Network
        """
        in_dim = {'task': 6, 'worker': 1, 'state': 4}
        hid_dim = {'task': 64, 'worker': 64, 'human': 64, 'state': 64}
        out_dim = {'task': 64, 'worker': 64, 'state': 64}

        cetypes = [('task', 'temporal', 'task'), ('task', 'assigned_to', 'worker'),
                   ('worker', 'com', 'worker'), ('task', 'tin', 'state'),
                   ('worker', 'win', 'state'), ('state', 'sin', 'state'),
                   ('task', 'take_time', 'worker'), ('worker', 'use_time', 'task')]

        self.gnn = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes)

    def get_variables(self, env: SchedulingEnv):
        # Unscheduled Tasks
        num_tasks = env.problem.num_tasks
        num_robots = env.team.num_robots
        num_humans = env.team.num_humans
        curr_g = copy.deepcopy(env.halfDG)
        curr_partials = copy.deepcopy(env.partials)
        curr_partialw = copy.deepcopy(env.partialw)
        durs = copy.deepcopy(env.dur)
        # Act Robot is not used for this model of Scheduler
        act_robot = 0
        unsch_tasks = np.array(action_helper_rollout(num_tasks, curr_partialw), dtype=np.int64)
        # Graph Neural Network
        g = build_hetgraph(curr_g, num_tasks, num_robots, num_humans, durs, curr_partials, unsch_tasks)
        # Feature Dictionary
        num_actions = len(unsch_tasks)
        feat_dict = hetgraph_node_helper(curr_g.number_of_nodes(),
                                         curr_partialw,
                                         curr_partials,
                                         durs,
                                         num_robots + num_humans, num_actions)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key])

        return g, feat_dict_tensor, unsch_tasks
