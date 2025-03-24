# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:53:49 2020

@author: pheno

Heterogeneous Graph Attention Layer

Differences from RSS2020 Version
    1. Input feature of task nodes extended to include statistics of completion times
    2. New edge types for task durations added between task and worker
    3. Location edges encode proximity constraints

Version: 2020-10-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn import GraphConv

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
# cetypes: reutrn of G.canonical_etypes

class HeteroGATLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, cetypes, l_alpha=0.2, use_relu=True):
        super(HeteroGATLayer, self).__init__()
        
        '''
        STN part
        '''
        # equation (1) for all relation types
        # W_r for each relation, used on the src node
        self.fc = nn.ModuleDict({name[1]: nn.Linear(in_dim[name[0]], out_dim[name[2]]) for name in cetypes})
        
        # equation (2)
        # incorporating weighted edge (包含带权边)
        self.edge_fc = nn.Linear(1, out_dim['task'])
        # use default initialization for weights and bias (定义线性层)
        self.attn_fc = nn.Linear(3 * out_dim['task'], 1, bias=False)
            
        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()
        
        '''
        The rest part / attention
        '''
        # # location
        # self.attn_fc_located_in = nn.Linear(2 * out_dim['loc'], 1, bias=False)
        # self.attn_fc_near = nn.Linear(2 * out_dim['loc'], 1, bias=False)
        # worker
        self.attn_fc_assigned_to = nn.Linear(2 * out_dim['worker'], 1, bias=False)
        self.attn_fc_com = nn.Linear(2 * out_dim['worker'], 1, bias=False)
        # graph
        self.attn_fc_tin = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.attn_fc_lin = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.attn_fc_rin = nn.Linear(2 * out_dim['state'], 1, bias=False)

        '''
        --Attention--
        Since for each etype, src is transformed into the out_dim of dst,
        input dim for nn.Linear should be
            2 * out_dim[dst_type], instead of out_dim[src_type] + out_dim[dst_type]
        for example,
            self.attn_fc_tin = nn.Linear(2 * out_dim['graph'], 1, bias=False)
        '''

        '''
        New updates
        '''
        # [task] - [take_time] - [worker]
        self.edge_fc_ttr = nn.Linear(1, out_dim['worker'])
        self.attn_fc_ttr = nn.Linear(3 * out_dim['worker'], 1, bias=False)
        # [worker] - [use_time] - [task]
        self.edge_fc_rut = nn.Linear(1, out_dim['task'])
        self.attn_fc_rut = nn.Linear(3 * out_dim['task'], 1, bias=False)

    # STN
    def edge_attention(self, edges):
        # calculate edge embedding zij
        zij = self.edge_fc(edges.data['weight'])
        # edge UDF for equation (2)
        z3 = torch.cat([edges.src['Wh_temporal'], edges.dst['Wh_temporal'], zij], dim=1)
        a = self.attn_fc(z3)
        return {'e': self.leaky_relu(a), 'embeddings': zij}

    # STN
    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        # add edge weight embedding into 'z'
        z_new = edges.src['Wh_temporal'] + edges.data['embeddings']
        return {'z': z_new, 'e': edges.data['e']}

    # STN
    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        # if self.use_relu:
        #     h = self.relu(h)
        return {'h': h}
    
    # task-loc
    def attention_located_in(self, edges):
        z2 = torch.cat([edges.src['Wh_located_in'], edges.dst['Wh_near']], dim=1)
        a = self.attn_fc_located_in(z2)
        return {'e_located_in': self.leaky_relu(a)}
    
    # task-loc
    def message_located_in(self, edges):
        return {'z_located_in': edges.src['Wh_located_in'], 
                'e_located_in': edges.data['e_located_in']}
    
    # task-loc
    def reduce_located_in(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_located_in'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_located_in'], dim=1)
        # if self.use_relu:
        #     h = self.relu(h)
        return {'h': h}
    
    # # loc-loc
    # def attention_near(self, edges):
    #     z2 = torch.cat([edges.src['Wh_near'], edges.dst['Wh_near']], dim=1)
    #     a = self.attn_fc_near(z2)
    #     return {'e_near': self.leaky_relu(a)}
    
    # # loc-loc
    # def message_near(self, edges):
    #     return {'z_near': edges.src['Wh_near'],
    #             'e_near': edges.data['e_near']}
    
    # # loc-loc
    # def reduce_near(self, nodes):
    #     alpha = F.softmax(nodes.mailbox['e_near'], dim=1)
    #     h = torch.sum(alpha * nodes.mailbox['z_near'], dim=1)
    #     # if self.use_relu:
    #     #     h = self.relu(h)
    #     return {'h': h}
    
    # task-worker
    def attention_assigned_to(self, edges):
        z2 = torch.cat([edges.src['Wh_assigned_to'], edges.dst['Wh_com']], dim=1)
        a = self.attn_fc_assigned_to(z2)
        return {'e_assigned_to': self.leaky_relu(a)}
    
    # task-worker
    def message_assigned_to(self, edges):
        return {'z_assigned_to': edges.src['Wh_assigned_to'],
                'e_assigned_to': edges.data['e_assigned_to']}

    # task-worker
    def reduce_assigned_to(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_assigned_to'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_assigned_to'], dim=1)
        # if self.use_relu:
        #     h = self.relu(h)
        return {'h': h}
    
    # worker-worker
    def attention_com(self, edges):
        z2 = torch.cat([edges.src['Wh_com'], edges.dst['Wh_com']], dim=1)
        # print(edges.src['Wh_com'], edges.dst['Wh_com'].shape, z2)
        a = self.attn_fc_com(z2)
        return {'e_com': self.leaky_relu(a)}
        
    # worker-worker
    def message_com(self, edges):
        return {'z_com': edges.src['Wh_com'],
                'e_com': edges.data['e_com']}

    # worker-worker
    def reduce_com(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_com'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_com'], dim=1)
        # if self.use_relu:
        #     h = self.relu(h)
        return {'h': h}
    
    # task-state
    def attention_tin(self, edges):
        z2 = torch.cat([edges.src['Wh_tin'], edges.dst['Wh_sin']], dim=1)
        a = self.attn_fc_tin(z2)
        return {'e_tin': self.leaky_relu(a)}
    
    # task-state
    def message_tin(self, edges):
        return {'z_tin': edges.src['Wh_tin'],
                'e_tin': edges.data['e_tin']}
    
    # task-state
    def reduce_tin(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_tin'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_tin'], dim=1)
        # if self.use_relu:
        #     h = self.relu(h)
        return {'h': h}

    # # loc-state
    # def attention_lin(self, edges):
    #     z2 = torch.cat([edges.src['Wh_lin'], edges.dst['Wh_sin']], dim=1)
    #     a = self.attn_fc_lin(z2)
    #     return {'e_lin': self.leaky_relu(a)}

    # # loc-state
    # def message_lin(self, edges):
    #     return {'z_lin': edges.src['Wh_lin'],
    #             'e_lin': edges.data['e_lin']}

    # # loc-state
    # def reduce_lin(self, nodes):
    #     alpha = F.softmax(nodes.mailbox['e_lin'], dim=1)
    #     h = torch.sum(alpha * nodes.mailbox['z_lin'], dim=1)
    #     # if self.use_relu:
    #     #     h = self.relu(h)
    #     return {'h': h}
    
    # worker-state
    def attention_win(self, edges):
        z2 = torch.cat([edges.src['Wh_win'], edges.dst['Wh_sin']], dim=1)
        a = self.attn_fc_rin(z2)
        return {'e_win': self.leaky_relu(a)}

    # worker-state
    def message_win(self, edges):
        return {'z_rin': edges.src['Wh_win'],
                'e_win': edges.data['e_win']}
        
    # worker-state
    def reduce_rin(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_win'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_rin'], dim=1)
        # if self.use_relu:
        #     h = self.relu(h)
        return {'h': h}
    
    '''
    New updates
    '''
    # [task] - [take_time] - [worker]
    def attention_ttr(self, edges):
        zij = self.edge_fc_ttr(edges.data['t'])
        z3 = torch.cat([edges.src['Wh_take_time'], edges.dst['Wh_com'], zij], dim=1)
        a = self.attn_fc_ttr(z3)
        return {'e_take_time': self.leaky_relu(a), 'embeddings': zij}
    
    # [task] - [take_time] - [worker]
    def message_ttr(self, edges):
        z_new = edges.src['Wh_take_time'] + edges.data['embeddings']
        return {'z_take_time': z_new, 'e_take_time': edges.data['e_take_time']}
    
    # [task] - [take_time] - [worker]
    def reduce_ttr(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_take_time'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_take_time'], dim=1)
        return {'h': h}
    
    # [worker] - [use_time] - [task]
    def attention_rut(self, edges):
        zij = self.edge_fc_rut(edges.data['t'])
        z3 = torch.cat([edges.src['Wh_use_time'], edges.dst['Wh_temporal'], zij], dim=1)
        a = self.attn_fc_rut(z3)
        return {'e_use_time': self.leaky_relu(a), 'embeddings': zij}
    
    # [worker] - [use_time] - [task]
    def message_rut(self, edges):
        z_new = edges.src['Wh_use_time'] + edges.data['embeddings']
        return {'z_use_time': z_new, 'e_use_time': edges.data['e_use_time']}
    
    # [worker] - [use_time] - [task]
    def reduce_rut(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_use_time'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_use_time'], dim=1)
        return {'h': h}
    
    '''
    Main forward pass
    '''
    def forward(self, g, feat_dict):
        '''
        Equation (1) for each relation type
        '''
        for srctype, etype, dsttype in g.canonical_etypes:
            Wh = self.fc[etype](feat_dict[srctype])
            g.nodes[srctype].data['Wh_%s' % etype] = Wh
        '''
        Equation (2) for each relation type
        '''
        # task
        g['temporal'].apply_edges(self.edge_attention)
        # # location
        # g['located_in'].apply_edges(self.attention_located_in)
        # g['near'].apply_edges(self.attention_near)
        # worker
        g['assigned_to'].apply_edges(self.attention_assigned_to)
        g['com'].apply_edges(self.attention_com)
        # graph
        g['tin'].apply_edges(self.attention_tin)
        # g['lin'].apply_edges(self.attention_lin)
        g['win'].apply_edges(self.attention_win)
        # [task] - [take_time] - [worker]
        g['take_time'].apply_edges(self.attention_ttr)
        # [worker] - [use_time] - [task]
        g['use_time'].apply_edges(self.attention_rut)
        
        '''
        Equation (3) & (4)
        '''
        #g['temporal'].update_all(self.message_func, self.reduce_func)
        #g['located_in'].update_all(self.message_located_in, self.reduce_located_in)
        #g['near'].update_all(self.message_near, self.reduce_near)
        
        funcs = {}
        funcs['temporal'] = (self.message_func, self.reduce_func)
        # funcs['located_in'] = (self.message_located_in, self.reduce_located_in)
        # funcs['near'] = (self.message_near, self.reduce_near)
        funcs['assigned_to'] = (self.message_assigned_to, self.reduce_assigned_to)
        funcs['com'] = (self.message_com, self.reduce_com)
        funcs['tin'] = (self.message_tin, self.reduce_tin)
        # funcs['lin'] = (self.message_lin, self.reduce_lin)
        funcs['win'] = (self.message_win, self.reduce_rin)
        funcs['sin'] = (fn.copy_src('Wh_sin', 'z_sin'), fn.sum('z_sin', 'h'))
        # [task] - [take_time] - [worker]
        funcs['take_time'] = (self.message_ttr, self.reduce_ttr)
        # [worker] - [use_time] - [task]
        funcs['use_time'] = (self.message_rut, self.reduce_rut)
        g.multi_update_all(funcs, 'sum')
        
        # deal with relu activation
        if self.use_relu:
            return {ntype: self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

# input similar to HeteroGATLayer
# merge = 'cat' or 'avg'
class MultiHeteroGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, cetypes, num_heads, merge='cat'):
        super(MultiHeteroGATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.merge = merge
        
        self.heads = nn.ModuleList()
        
        if self.merge == 'cat':        
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayer(in_dim, out_dim, cetypes))
        else:
            #self.relu = nn.ReLU()
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayer(in_dim, out_dim, cetypes, use_relu=False))

    def forward(self, g, feat_dict):
        tmp = {}
        for ntype in feat_dict:
            tmp[ntype] = []
            
        for i in range(self.num_heads):
            head_out = self.heads[i](g, feat_dict)
            
            for ntype in feat_dict:
                tmp[ntype].append(head_out[ntype])
        
        results = {}
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)  
            for ntype in feat_dict:
                results[ntype] = torch.cat(tmp[ntype], dim=1)
        else:
            # merge using average
            for ntype in feat_dict:
                # dont use relu as the predicted q scores are negative values
                #results[ntype] = self.relu(torch.mean(torch.stack(tmp[ntype]), dim=0))
                results[ntype] = torch.mean(torch.stack(tmp[ntype]), dim=0)
        
        return results
    

class HeteroGCNLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, cetypes, use_relu=True):
        super(HeteroGCNLayer, self).__init__()
        
        # GCN weights for each relation type (W_r)
        self.fc = nn.ModuleDict({name[1]: nn.Linear(in_dim[name[0]], out_dim[name[2]]) for name in cetypes})

        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()
    
    # Standard message function for GCN
    def message_func(self, edges):
        # Messages are just the transformed source node features
        return {'z': edges.src['Wh']}

    # Standard reduce function for GCN (sum over neighbors)
    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}
    
    def forward(self, g, feat_dict):
        # Linear transformation for each node type using the corresponding relation type
        for srctype, etype, dsttype in g.canonical_etypes:
            Wh = self.fc[etype](feat_dict[srctype])
            g.nodes[srctype].data['Wh'] = Wh
        
        # Define message passing functions for each edge type
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            funcs[etype] = (self.message_func, self.reduce_func)
        
        # Perform message passing and aggregation for each edge type
        g.multi_update_all(funcs, 'sum')
        
        # Apply activation if specified
        if self.use_relu:
            return {ntype: self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
