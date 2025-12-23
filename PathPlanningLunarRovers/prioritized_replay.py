# -*- coding: utf-8 -*-
"""
优先经验回放模块 (Prioritized Experience Replay, PER)
基于TD误差的优先级采样机制，提高关键经验的利用效率
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import namedtuple
from config import config


# 定义经验元组结构（包含additional_features用于A*路径信息）
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done',
                                        'additional_features', 'next_additional_features'])


class SumTree:
    """
    求和树数据结构
    用于高效实现基于优先级的采样
    叶节点存储优先级，内部节点存储子节点优先级之和
    """

    def __init__(self, capacity: int):
        """
        初始化求和树

        Args:
            capacity: 树的容量（叶节点数量）
        """
        self.capacity = capacity
        self.write_index = 0  # 下一个写入位置
        self.n_entries = 0    # 当前存储的经验数量

        # 树的总节点数 = 2 * capacity - 1
        # 其中内部节点数 = capacity - 1，叶节点数 = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # 存储实际数据的数组
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, index: int, change: float):
        """
        向上传播优先级变化

        Args:
            index: 发生变化的节点索引
            change: 优先级变化量
        """
        parent = (index - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index: int, value: float) -> int:
        """
        根据优先级值检索叶节点

        Args:
            index: 当前节点索引
            value: 目标优先级值

        Returns:
            叶节点索引
        """
        left = 2 * index + 1
        right = left + 1

        # 如果是叶节点，返回当前索引
        if left >= len(self.tree):
            return index

        # 根据左子节点的值决定搜索方向
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """返回所有优先级之和（根节点的值）"""
        return self.tree[0]

    def add(self, priority: float, data: Experience):
        """
        添加新的经验数据

        Args:
            priority: 经验的优先级
            data: 经验数据
        """
        # 计算叶节点在树中的索引
        tree_index = self.write_index + self.capacity - 1

        # 存储数据
        self.data[self.write_index] = data

        # 更新优先级
        self.update(tree_index, priority)

        # 更新写入位置（循环覆盖）
        self.write_index = (self.write_index + 1) % self.capacity

        # 更新存储数量
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index: int, priority: float):
        """
        更新指定节点的优先级

        Args:
            tree_index: 树节点索引
            priority: 新的优先级值
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def get(self, value: float) -> Tuple[int, float, Experience]:
        """
        根据优先级值获取对应的经验

        Args:
            value: 采样用的优先级值

        Returns:
            (树索引, 优先级, 经验数据)
        """
        tree_index = self._retrieve(0, value)
        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], self.data[data_index]


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    基于TD误差为每个经验分配优先级，优先采样重要经验
    """

    def __init__(self, capacity: int = None, alpha: float = None,
                 beta_start: float = None, beta_end: float = None,
                 beta_frames: int = 100000):
        """
        初始化优先经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=均匀采样, 1=完全按优先级)
            beta_start: 重要性采样权重初始值
            beta_end: 重要性采样权重最终值
            beta_frames: beta线性增长的总帧数
        """
        self.capacity = capacity or config.REPLAY_BUFFER_SIZE
        self.alpha = alpha or config.PER_ALPHA
        self.beta_start = beta_start or config.PER_BETA_START
        self.beta_end = beta_end or config.PER_BETA_END
        self.beta_frames = beta_frames

        # 创建求和树
        self.tree = SumTree(self.capacity)

        # 最大优先级（用于新经验的初始优先级）
        self.max_priority = 1.0

        # 小常数，避免优先级为0
        self.epsilon = config.PER_EPSILON

        # 当前帧数（用于计算beta）
        self.frame = 0

    def _get_beta(self) -> float:
        """
        计算当前的beta值（线性增长）

        Returns:
            当前beta值
        """
        beta = self.beta_start + (self.beta_end - self.beta_start) * \
               min(1.0, self.frame / self.beta_frames)
        return beta

    def _get_priority(self, td_error: float) -> float:
        """
        根据TD误差计算优先级

        优先级公式: ψ_i = (|δ_i| + ε)^α

        重要改进：裁剪TD误差，防止极端优先级导致采样偏差
        - 当模型学会后，碰撞产生的TD误差可能高达1000+
        - 不裁剪会导致碰撞经验主导采样，模型"遗忘"成功策略

        Args:
            td_error: TD误差

        Returns:
            优先级值
        """
        # 裁剪TD误差，防止极端优先级（关键修复！）
        # 将TD误差限制在合理范围，避免单一失败经验主导采样
        clipped_error = min(abs(td_error), 10.0)
        return (clipped_error + self.epsilon) ** self.alpha

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool,
             additional_features: np.ndarray = None,
             next_additional_features: np.ndarray = None):
        """
        将新经验添加到缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            additional_features: 当前状态的额外特征（A*方向、目标方向等）
            next_additional_features: 下一状态的额外特征
        """
        # 创建经验元组
        experience = Experience(state, action, reward, next_state, done,
                               additional_features, next_additional_features)

        # 新经验使用最大优先级，确保至少被采样一次
        priority = self.max_priority ** self.alpha

        # 添加到求和树
        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, List[int]]:
        """
        从缓冲区中按优先级采样一批经验

        采样概率公式: P(i) = ψ_i^α / Σ_k ψ_k^α

        Args:
            batch_size: 批量大小

        Returns:
            (states, actions, rewards, next_states, dones,
             additional_features, next_additional_features, weights, indices)
            weights: 重要性采样权重
            indices: 采样经验在树中的索引（用于更新优先级）
        """
        batch_size = min(batch_size, self.tree.n_entries)

        # 存储采样结果
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        additional_features_list = []
        next_additional_features_list = []
        indices = []
        priorities = []

        # 计算采样区间
        total_priority = self.tree.total()
        segment = total_priority / batch_size

        # 分层采样：将优先级范围均匀分成batch_size个区间
        for i in range(batch_size):
            # 在每个区间内随机采样
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)

            # 从树中获取对应经验
            tree_idx, priority, experience = self.tree.get(value)

            if experience is None or experience == 0:
                continue

            # 收集数据
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            next_states.append(experience.next_state)
            dones.append(experience.done)
            additional_features_list.append(experience.additional_features)
            next_additional_features_list.append(experience.next_additional_features)
            indices.append(tree_idx)
            priorities.append(priority)

        # 计算重要性采样权重
        # w_i = (N * P(i))^(-β) / max_j(w_j)
        beta = self._get_beta()
        sampling_probabilities = np.array(priorities) / total_priority
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        # 归一化权重
        weights = weights / weights.max()

        # 更新帧计数
        self.frame += 1

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones),
                np.array(additional_features_list), np.array(next_additional_features_list),
                weights, indices)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        根据新的TD误差更新经验优先级

        Args:
            indices: 经验在树中的索引列表
            td_errors: 对应的TD误差数组
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)

            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """返回当前缓冲区中的经验数量"""
        return self.tree.n_entries


class ReplayBuffer:
    """
    标准经验回放缓冲区（无优先级）
    作为基准对比使用
    """

    def __init__(self, capacity: int = None):
        """
        初始化缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity or config.REPLAY_BUFFER_SIZE
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool,
             additional_features: np.ndarray = None,
             next_additional_features: np.ndarray = None):
        """添加新经验"""
        experience = Experience(state, action, reward, next_state, done,
                               additional_features, next_additional_features)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """随机采样一批经验"""
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states = np.array([self.buffer[i].state for i in indices])
        actions = np.array([self.buffer[i].action for i in indices])
        rewards = np.array([self.buffer[i].reward for i in indices])
        next_states = np.array([self.buffer[i].next_state for i in indices])
        dones = np.array([self.buffer[i].done for i in indices])
        additional_features = np.array([self.buffer[i].additional_features for i in indices])
        next_additional_features = np.array([self.buffer[i].next_additional_features for i in indices])

        return states, actions, rewards, next_states, dones, additional_features, next_additional_features

    def __len__(self) -> int:
        return len(self.buffer)


# 测试代码
if __name__ == "__main__":
    # 测试优先经验回放
    per_buffer = PrioritizedReplayBuffer(capacity=1000)

    # 添加一些测试经验
    for i in range(100):
        state = np.random.rand(80, 64, 4)
        action = np.random.randint(0, 7)
        reward = np.random.randn()
        next_state = np.random.rand(80, 64, 4)
        done = i % 10 == 0

        per_buffer.push(state, action, reward, next_state, done)

    print(f"缓冲区大小: {len(per_buffer)}")

    # 测试采样
    states, actions, rewards, next_states, dones, weights, indices = per_buffer.sample(32)
    print(f"采样批次大小: {len(states)}")
    print(f"权重范围: [{weights.min():.4f}, {weights.max():.4f}]")

    # 测试优先级更新
    td_errors = np.random.rand(32)
    per_buffer.update_priorities(indices, td_errors)
    print("优先级更新完成")
