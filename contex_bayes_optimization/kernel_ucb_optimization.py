import numpy as np
import warnings
from .action_space import ActionSpace

def rbf_kernel(x, y, gamma=1.0):
    """示例 RBF 核函数"""
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))

class KernelUCBOptimizer():
    def __init__(self, all_actions_dict, contexts, contexts_dim, actions_dim, 
                 embedding_dim, alpha=1.0, init_random=5, gamma=1.0, lambda_reg=1e-5):
        self._space = ActionSpace(all_actions_dict, contexts, contexts_dim, 
                                  actions_dim, embedding_dim)
        self.alpha = alpha
        self.init_random = init_random
        
        self.contexts_dim = contexts_dim
        self.actions_dim = actions_dim
        self.gamma = gamma
        self.lambda_reg = lambda_reg

        # 保存（context, action）拼接后的向量及对应 reward
        self.observations = []  # 每个元素为一维数组
        self.rewards = []       # 每个元素为 float
        self.K_inv = None       # 核矩阵的逆矩阵

    @property
    def space(self):
        return self._space

    @property
    def res(self):
        return self._space.res()
    
    def register(self, context, action, reward):
        """保存观察样本并更新核矩阵的逆"""
        # 将新的 observation 添加到 action space
        self._space.register(context, action, reward)
        context_array = self.context_to_array(context).reshape(-1)
        action_array = self.action_to_array(action).reshape(-1)
        new_point = np.concatenate((context_array, action_array))
        
        self.observations.append(new_point)
        self.rewards.append(reward)
        
        # 更新核矩阵及其逆
        X = np.array(self.observations)  # shape (n, d)
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = rbf_kernel(X[i], X[j], self.gamma)
        # 加入正则项，保证数值稳定
        self.K_inv = np.linalg.inv(K + self.lambda_reg * np.eye(n))

    def array_to_context(self, context):
        return self._space.array_to_context(context)
    
    def action_to_array(self, action):
        return self._space.action_to_array(action)

    def context_to_array(self, context):
        return self._space.context_to_array(context)

    def suggest(self, context):
        """通过计算核上置信上界，选择下一个最有希望的 action"""
        # 若接收到观察数还少，则随机采样
        if len(self._space) < self.init_random or len(self.observations) < self.init_random:
            return self._space.array_to_action(self._space.random_sample())
        
        context_array = self.context_to_array(context).reshape(-1)
        best_action = None
        best_ucb = -np.inf

        X_obs = np.array(self.observations)  # 历史 observation，shape (n, d)
        rewards = np.array(self.rewards).reshape(-1, 1)  # shape (n,1)
        
        # 对于每个候选 action 计算 UCB
        for action in self._space.actions:
            action_array = np.array(action).reshape(-1)
            new_point = np.concatenate((context_array, action_array))
            
            # 构造核向量 k_x，与每个 observation 之间的核值
            k_x = np.array([rbf_kernel(new_point, obs, self.gamma) for obs in X_obs]).reshape(-1, 1)
            mean = k_x.T @ self.K_inv @ rewards
            variance = rbf_kernel(new_point, new_point, self.gamma) - k_x.T @ self.K_inv @ k_x
            
            # 注意：variance 可能出现非常小的负数，由于数值误差，可取 max(variance, 0)
            ucb = mean + self.alpha * np.sqrt(max(variance, 0))
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return self._space.array_to_action(best_action)

