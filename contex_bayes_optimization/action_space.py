import numpy as np
from itertools import product

class ActionSpace(object):

    def __init__(self, discvars, contexts, contexts_dim, actions_dim, embedding_dim):

        # Get the name of the parameters
        self._action_keys = discvars.keys()
        self._context_keys = contexts.keys()
        # 设置 bounds，确保其长度与 length_scale 一致
        self._bounds = [(1e-2, 1e2)] * (contexts_dim + actions_dim)
        self.actions_dim = actions_dim
        self.contexts_dim = contexts_dim
        self.embedding_dim = embedding_dim
        # 提取所有 [task, worker] 组合列表
        # combinations = [[{'task': i, 'worker': j} for j in bound['domain']] for i, bound in enumerate(discvars)]
        # 使用列表推导式将其转化为所需的数组形式
        # allList = [[[entry['task'], entry['worker']] for entry in sublist] for sublist in combinations]
        # allActions = np.array(allList)
        allList = [discvars[k] for k in discvars.keys()]
        allActions = np.array(list(product(*allList)))

        # 将 allActions 转化为形状 (10000, 18) 的数组
        reshaped_actions = allActions.reshape(10000, -1)
        self._allActions = reshaped_actions

        # preallocated memory for X and Y points
        self._context = np.empty(shape=(0, self.contexts_dim))
        self._action = np.empty(shape=(0, self.actions_dim))
        self._context_action = np.empty(shape=(0, self.actions_dim + self.contexts_dim))
        self._reward = np.empty(shape=(0,))

    def __len__(self):
        assert len(self._action) == len(self._reward)
        assert len(self._action) == len(self._context)
        return len(self._reward)

    # @property是Python中用于创建属性的装饰器（decorator）->允许将一个方法转换为一个类的属性，从而可以像访问属性一样访问此方法，而不需调用
    @property
    def actions(self):
        return self._allActions
    
    @property
    def empty(self):
        return len(self) == 0
    
    @property
    def context(self):
        return self._context

    @property
    def action(self):
        return self._action
    
    @property
    def context_action(self):
        return self._context_action

    @property
    def reward(self):
        return self._reward
    
    @property
    def context_dim(self):
        return len(self._context_keys)
        # return self._context_dim

    @property
    def action_dim(self):
        return len(self._action_keys)

    @property
    def context_keys(self):
        return self._context_keys
    
    @property
    def action_keys(self):
        return self._action_keys

    @property
    def bounds(self):
        return self._bounds

    def action_to_array(self, action):
        try:
            assert set(action) == set(self._action_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(action)) +
                "not match the expected set of keys ({}).".format(self._action_keys)
            )
        return np.asarray([action[key] for key in self._action_keys])
    
    def context_to_array(self, context):
        try:
            assert set(context) == set(self._context_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(context)) +
                "not match the expected set of keys ({}).".format(self._context_keys)
            )
        # return np.asarray([context[key] for key in self._context_keys])
        return np.concatenate([v for v in context.values()])

    def array_to_action(self, x):
        try:
            # assert len(x) == len(self._action_keys)
            assert len(x) == self.actions_dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._action_keys))
            )
        x = x.reshape(self.action_dim, len(x) // self.action_dim)
        return dict(zip(self._action_keys, x))

    def array_to_context(self, x):
        try:
            assert len(x) == len(self._context_keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._context_keys))
            )
        # 遍历x的键值对，并将值赋给context中的相应键
        return x
        # return dict(zip(self._context_keys, x))


    def register(self, context, action, reward):

        c = self.context_to_array(context)
        a = self.action_to_array(action)
        ca = np.concatenate([c.reshape(1, -1), a.reshape(1, -1)], axis=1)
        
        self._context = np.concatenate([self._context, c.reshape(1, -1)])
        self._action = np.concatenate([self._action, a.reshape(1, -1)])
        self._reward = np.concatenate([self._reward, [reward]])
        self._context_action = np.concatenate([self._context_action, ca.reshape(1, -1)])


    def random_sample(self):
        rand_idx = np.random.randint(len(self._allActions))
        action_sample = self._allActions[rand_idx, :]
        # action_sample = action_sample.reshape(self.action_dim, len(action_sample) // self.action_dim)
        # return self._allActions[rand_idx, :]
        return action_sample


    def res(self):
        """Get all reward values found and corresponding parameters."""
        # context = [dict(zip(self._context_keys, p)) for p in self.context]
        # action = [dict(zip(self._action_keys, p)) for p in self.action]
        # 定义各个键对应的长度
        context_key_lengths = {'task': self.embedding_dim, 'worker': self.embedding_dim, 'state': self.embedding_dim}
        # context = {key: np.empty((self.context.shape[0], length)) for key, length in context_key_lengths.items()}
        context = {key: self.context[:, i * context_key_lengths[key]: (i + 1) * context_key_lengths[key]]
                   for i, key in enumerate(self._context_keys)}
        action = {key: self.action[:, i * (self.actions_dim // self.action_dim):(i + 1) * (self.actions_dim // self.action_dim)]
                  for i, key in enumerate(self._action_keys)}
        # return [
        #     {"reward": r, "action": a, "context": c}
        #     for r, a, c in (self.reward, action, context)
        # ]
        r = self.reward
        a = np.column_stack((action['tasks'], action['workers']))
        c = np.column_stack((context['task'], context['worker'], context['state']))

        # 构造 res 字典
        res_dict = {'reward': r, 'action': a, 'context': c}
        return res_dict

