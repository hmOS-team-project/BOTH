# -*- coding: utf-8 -*-
"""
Created on Thu July 4, 2024

@author: Hui

HGA encoder
"""
import os
import torch
import random
import copy
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam
from env.scheduling_env import SchedulingEnv
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv
from benchmark.hybridnet.utils import ReplayMemory, Transition, action_helper_rollout
from benchmark.hybridnet.utils import hetgraph_node_helper, build_hetgraph
from benchmark.hybridnet.hetnet import HybridScheduleNet
from benchmark.hybridnet.hetnet import GCNScheduleNet

# Define Heterogeneous Graph Attention network (HetGAT)
class HetGATNetwork(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super(HetGATNetwork, self).__init__()
        self.device = device
        self._init_gat()
        # self._load_pretrained_weights()


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

        self.gnn = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(self.device)
        # self.gnn = GCNScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(self.device)

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
    

    # def _load_pretrained_weights(self):
    #     """
    #     Load pretrained weights into the GNN model.
    #     """
    #     folder_trained_checkpoint = 'data/final_trained_models/both/checkpoint_hetgat_small_both.tar'
    #     # folder_trained_checkpoint = 'data/final_trained_models/both/checkpoint_no-attention_both.tar'
    #     checkpoint = torch.load(folder_trained_checkpoint, map_location=self.device)
    #     self.gnn.load_state_dict(checkpoint['hetgat_state_dict'])

    def load_checkpoint(self, trained_checkpoint):
        cp = torch.load(trained_checkpoint)
        self.gnn.load_state_dict(cp['hetgat_state_dict'])
        return cp['i_batch'] + 1


