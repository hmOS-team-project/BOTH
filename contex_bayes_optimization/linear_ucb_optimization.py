import numpy as np
import warnings
from .action_space import ActionSpace

class LinUCBOptimizer():
    def __init__(self, all_actions_dict, contexts, contexts_dim, actions_dim, embedding_dim, alpha=1.0, init_random=5):
        self._space = ActionSpace(all_actions_dict, contexts, contexts_dim, actions_dim, embedding_dim)
        self.alpha = alpha
        self.init_random = init_random
        
        self.contexts_dim = contexts_dim
        self.actions_dim = actions_dim
        self.A = np.identity(contexts_dim + actions_dim)
        self.b = np.zeros((contexts_dim + actions_dim, 1))

    @property
    def space(self):
        return self._space

    @property
    def res(self):
        return self._space.res()

    def register(self, context, action, reward):
        """Expect observation with known reward"""
        context_array = self.context_to_array(context).reshape(1, -1)
        action_array = self.action_to_array(action).reshape(1, -1)
        x = np.concatenate((context_array, action_array), axis=1).reshape(-1, 1)

        
        if x.shape[0] != self.A.shape[0]:
            raise ValueError(f"Shape mismatch: x has shape {x.shape} but A has shape {self.A.shape}")

        self.A += x @ x.T
        self.b += reward * x

        # 添加context, action, reward到self._space中
        self._space.register(context, action, reward)

    def array_to_context(self, context):
        return self._space.array_to_context(context)
    
    def action_to_array(self, action):
        return self._space.action_to_array(action)

    def context_to_array(self, context):
        return self._space.context_to_array(context)

    def suggest(self, context):
        """Most promising point to probe next"""
        if len(self._space) < self.init_random:
            return self._space.array_to_action(self._space.random_sample())

        context_array = self.context_to_array(context).reshape(-1, 1)

        # print(f'context_array: {context_array.shape}')
        theta = np.linalg.inv(self.A) @ self.b

        best_action = None
        best_ucb = -np.inf

        for action in self._space.actions:
            # print(action)
            action_array = np.array(action).reshape(-1, 1)  # 将 action 转换为列向量

            x = np.concatenate((context_array, action_array), axis=0) # 在行方向上拼接
            
            mean = theta.T @ x
            ucb = mean + self.alpha * np.sqrt(x.T @ np.linalg.inv(self.A) @ x)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
        
        return self._space.array_to_action(best_action)
