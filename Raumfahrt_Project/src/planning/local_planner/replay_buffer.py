# -*- coding: utf-8 -*-
"""
经验回放缓冲区模块
实现普通经验回放和优先经验回放
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """
    普通经验回放缓冲区
    """

    def __init__(self, capacity):
        """
        初始化经验回放缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, additional_features=None, next_additional_features=None):
        """
        存储经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
            additional_features: 额外特征
            next_additional_features: 下一状态的额外特征
        """
        self.buffer.append((state, action, reward, next_state, done, additional_features, next_additional_features))

    def sample(self, batch_size):
        """
        采样经验

        Args:
            batch_size: 批次大小

        Returns:
            采样的经验批次
        """
        idxes = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, additional_features, next_additional_features = zip(*[self.buffer[i] for i in idxes])
        return states, actions, rewards, next_states, dones, additional_features, next_additional_features

    def __len__(self):
        """
        返回缓冲区大小

        Returns:
            缓冲区大小
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    基于TD误差的优先级采样
    """

    def __init__(self, capacity, alpha=0.6):
        """
        初始化优先经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数，控制优先级的影响程度
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, additional_features=None, next_additional_features=None):
        """
        存储经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
            additional_features: 额外特征
            next_additional_features: 下一状态的额外特征
        """
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, additional_features, next_additional_features))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, additional_features, next_additional_features)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        基于优先级采样经验

        Args:
            batch_size: 批次大小
            beta: 重要性采样权重的指数

        Returns:
            采样的经验批次和权重
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # 计算概率
        probs = prios ** self.alpha
        probs /= probs.sum()

        # 采样
        idxes = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        states, actions, rewards, next_states, dones, additional_features, next_additional_features = zip(*[self.buffer[i] for i in idxes])

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[idxes]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return states, actions, rewards, next_states, dones, additional_features, next_additional_features, weights, idxes

    def update_priorities(self, idxes, priorities):
        """
        更新优先级

        Args:
            idxes: 索引
            priorities: 新的优先级
        """
        for idx, prio in zip(idxes, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """
        返回缓冲区大小

        Returns:
            缓冲区大小
        """
        return len(self.buffer)


# 测试代码
if __name__ == "__main__":
    # 测试普通经验回放
    print("测试普通经验回放缓冲区")
    buffer = ReplayBuffer(capacity=1000)

    # 添加经验
    for i in range(100):
        state = np.random.rand(80, 64, 4)
        action = np.random.randint(0, 9)
        reward = np.random.randn()
        next_state = np.random.rand(80, 64, 4)
        done = False
        features = np.random.rand(6)
        next_features = np.random.rand(6)
        buffer.push(state, action, reward, next_state, done, features, next_features)

    print(f"缓冲区大小: {len(buffer)}")

    # 采样
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones, additional_features, next_additional_features = batch
    print(f"采样批次大小: {len(states)}")
    print(f"状态形状: {states[0].shape}")
    print(f"额外特征形状: {additional_features[0].shape}")

    # 测试优先经验回放
    print("\n测试优先经验回放缓冲区")
    per_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)

    # 添加经验
    for i in range(100):
        state = np.random.rand(80, 64, 4)
        action = np.random.randint(0, 9)
        reward = np.random.randn()
        next_state = np.random.rand(80, 64, 4)
        done = False
        features = np.random.rand(6)
        next_features = np.random.rand(6)
        per_buffer.push(state, action, reward, next_state, done, features, next_features)

    print(f"缓冲区大小: {len(per_buffer)}")

    # 采样
    batch = per_buffer.sample(32, beta=0.4)
    states, actions, rewards, next_states, dones, additional_features, next_additional_features, weights, idxes = batch
    print(f"采样批次大小: {len(states)}")
    print(f"权重形状: {weights.shape}")
    print(f"索引形状: {idxes.shape}")

    # 更新优先级
    new_priorities = np.random.rand(32)
    per_buffer.update_priorities(idxes, new_priorities)
    print("优先级更新完成")

    print("\n回放缓冲区测试完成!")
