# -*- coding: utf-8 -*-
"""
D3QN智能体模块 (D3QN Agent)
整合D3QN网络、优先经验回放和A*全局规划的完整智能体
"""

import torch
import torch.optim as optim
import numpy as np
import os
from typing import Tuple, Dict, Optional, List
from collections import deque

from ..global_planner.astar import AStarPlanner, create_grid_map
from .network import DuelingDQN, D3QNLoss, create_networks, hard_update, soft_update
from .replay_buffer import PrioritizedReplayBuffer


class D3QNAgent:
    """
    D3QN智能体
    实现完整的训练和推理流程，包括：
    - Dueling Double DQN网络
    - 优先经验回放
    - ε-greedy探索策略
    - A*全局路径规划辅助
    """

    def __init__(self, use_per: bool = True, use_astar: bool = True):
        """
        初始化D3QN智能体

        Args:
            use_per: 是否使用优先经验回放
            use_astar: 是否使用A*全局规划辅助
        """
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"D3QN智能体使用设备: {self.device}")

        # 创建网络
        self.online_net, self.target_net = create_networks(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=0.0001  # LEARNING_RATE
        )

        # 学习率调度器（指数衰减）
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.99  # LR_DECAY_RATE
        )
        self.current_lr = 0.0001

        # 损失函数
        self.loss_fn = D3QNLoss(gamma=0.99)  # GAMMA

        # 经验回放
        self.use_per = use_per
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=100000,  # REPLAY_BUFFER_SIZE
                alpha=0.6  # PER_ALPHA
            )
        else:
            from .replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer(capacity=100000)

        # 探索参数
        self.epsilon = 1.0  # EPSILON_START
        self.epsilon_min = 0.01  # EPSILON_MIN
        self.epsilon_decay = 0.995  # EPSILON_DECAY

        # A*规划器
        self.use_astar = use_astar
        self.astar_planner = None
        self.global_path = None
        self.current_waypoint_idx = 0

        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.q_values = []

        # 动作空间
        self.num_actions = 9  # ACTION_SPACE_SIZE

    def init_astar_planner(self, grid_map: np.ndarray, resolution: float = 0.1):
        """
        初始化A*规划器

        Args:
            grid_map: 栅格地图
            resolution: 地图分辨率
        """
        self.astar_planner = AStarPlanner(grid_map, resolution)

    def plan_global_path(self, start: Tuple[float, float],
                         goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        使用A*算法规划全局路径

        Args:
            start: 起点坐标
            goal: 目标坐标

        Returns:
            路径点列表，如果规划失败返回None
        """
        if self.astar_planner is None:
            return None

        self.global_path = self.astar_planner.find_path(start, goal)
        self.current_waypoint_idx = 0

        if self.global_path:
            # 平滑路径
            self.global_path = self.astar_planner.smooth_path(self.global_path)

        return self.global_path

    def get_next_waypoint(self, current_pos: Tuple[float, float],
                          waypoint_threshold: float = 0.5) -> Optional[Tuple[float, float]]:
        """
        获取下一个路径点

        Args:
            current_pos: 当前位置
            waypoint_threshold: 到达路径点的距离阈值

        Returns:
            下一个路径点坐标
        """
        if self.global_path is None or len(self.global_path) == 0:
            return None

        # 检查是否到达当前路径点
        while self.current_waypoint_idx < len(self.global_path):
            waypoint = self.global_path[self.current_waypoint_idx]
            distance = np.sqrt(
                (current_pos[0] - waypoint[0]) ** 2 +
                (current_pos[1] - waypoint[1]) ** 2
            )

            if distance < waypoint_threshold:
                self.current_waypoint_idx += 1
            else:
                return waypoint

        return None

    def select_action(self, state: np.ndarray,
                      additional_features: np.ndarray = None,
                      current_pos: Tuple[float, float] = None,
                      current_theta: float = None,
                      training: bool = True) -> int:
        """
        选择动作

        核心思想：A*路径信息仅通过additional_features传入网络作为状态信息，
        网络通过学习自主决策如何利用A*信息，实现端到端学习

        Args:
            state: 当前状态（深度图）
            additional_features: 额外特征 [目标方向, 目标距离, A*方向, A*距离, 障碍物方向, 障碍物距离]
            current_pos: 当前位置 (x, y)（保留参数但不用于引导）
            current_theta: 当前航向角（保留参数但不用于引导）
            training: 是否处于训练模式

        Returns:
            选择的动作索引
        """
        # ε-greedy探索：纯随机探索，让网络学习利用additional_features中的A*信息
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        # 贪婪选择：完全由网络决策
        # 网络通过additional_features获得A*路径信息，自主学习如何利用
        network_action = self.online_net.get_action(
            state, additional_features,
            epsilon=0.0, device=self.device
        )

        return network_action

    def _get_astar_guided_action(self, current_pos: Tuple[float, float],
                                  current_theta: float) -> Optional[int]:
        """
        根据A*路径获取引导动作

        Args:
            current_pos: 当前位置
            current_theta: 当前航向角

        Returns:
            建议的动作索引
        """
        if self.global_path is None or len(self.global_path) == 0:
            return None

        # 获取下一个路径点
        waypoint = self.get_next_waypoint(current_pos, waypoint_threshold=0.3)
        if waypoint is None:
            return None

        # 计算到路径点的方向
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        target_angle = np.arctan2(dy, dx)

        # 计算需要转向的角度
        angle_diff = target_angle - current_theta
        # 归一化到 [-π, π]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # 根据角度差选择动作
        # 新动作空间: 0=低速左转30°, 1=低速左转15°, 2=低速直行, 3=低速右转15°, 4=低速右转30°
        #            5=高速左转15°, 6=高速直行, 7=高速右转15°, 8=停止
        if abs(angle_diff) < np.pi / 24:  # <7.5°，方向基本正确
            return 6  # 高速直行
        elif abs(angle_diff) < np.pi / 12:  # <15°，需要小幅调整
            return 5 if angle_diff > 0 else 7  # 高速小转弯
        elif abs(angle_diff) < np.pi / 6:  # <30°，需要中等调整
            return 1 if angle_diff > 0 else 3  # 低速小转弯
        else:  # >=30°，需要大幅调整
            return 0 if angle_diff > 0 else 4  # 低速大转弯

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool,
                         additional_features: np.ndarray = None,
                         next_additional_features: np.ndarray = None):
        """
        存储经验到回放缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
            additional_features: 当前状态的额外特征（A*方向、目标方向等）
            next_additional_features: 下一状态的额外特征
        """
        self.replay_buffer.push(state, action, reward, next_state, done,
                               additional_features, next_additional_features)

    def train_step(self) -> Optional[float]:
        """
        执行一步训练

        Returns:
            损失值，如果缓冲区样本不足返回None
        """
        # 检查是否有足够的样本（使用预填充阈值）
        min_size = 1000  # MIN_REPLAY_SIZE
        if len(self.replay_buffer) < min_size:
            return None

        # 采样经验（现在包含additional_features）
        if self.use_per:
            (states, actions, rewards, next_states, dones,
             additional_features, next_additional_features,
             weights, indices) = self.replay_buffer.sample(32)  # 使用更小的批量大小，加快训练
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            (states, actions, rewards, next_states, dones,
             additional_features, next_additional_features) = \
                self.replay_buffer.sample(32)  # 使用更小的批量大小，加快训练
            weights = None
            indices = None

        # 转换为张量（优化方式）
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 转换additional_features为张量
        if additional_features is not None and additional_features[0] is not None:
            additional_features = torch.FloatTensor(additional_features).to(self.device)
            next_additional_features = torch.FloatTensor(next_additional_features).to(self.device)
        else:
            additional_features = None
            next_additional_features = None

        # 调整维度: (batch, height, width, channels) -> (batch, channels, height, width)
        if states.dim() == 4 and states.size(-1) == 4:  # DEPTH_FRAME_STACK
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)

        # 计算损失（现在传入additional_features）
        loss, td_errors = self.loss_fn(
            self.online_net, self.target_net,
            states, actions, rewards, next_states, dones,
            additional_features=additional_features,
            next_additional_features=next_additional_features,
            weights=weights
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（使用配置参数）
        grad_clip = 10.0  # GRAD_CLIP_NORM
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=grad_clip)
        self.optimizer.step()

        # 更新优先级（如果使用PER）
        if self.use_per and indices is not None:
            self.replay_buffer.update_priorities(
                indices,
                td_errors.abs().detach().cpu().numpy()
            )

        # 记录统计
        self.losses.append(loss.item())

        # 记录Q值（传入additional_features）
        with torch.no_grad():
            q_values = self.online_net(states, additional_features)
            mean_q = q_values.max(dim=1)[0].mean().item()
            self.q_values.append(mean_q)

        self.training_step += 1

        # 目标网络更新
        use_soft_update = True
        if use_soft_update:
            # 软更新：每步进行小幅度更新，更平滑稳定
            # θ_target = τ * θ_online + (1 - τ) * θ_target
            if self.training_step % 4 == 0:  # TARGET_UPDATE_FREQ
                self._soft_update(self.online_net, self.target_net, 0.001)  # TAU
        else:
            # 硬更新：每隔一定步数完全复制（原始方式，可能导致Q值震荡）
            if self.training_step % 1000 == 0:
                self._hard_update(self.online_net, self.target_net)

        return loss.item()

    def _soft_update(self, online_net, target_net, tau):
        """
        软更新目标网络参数
        """
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def _hard_update(self, online_net, target_net):
        """
        硬更新目标网络参数
        """
        target_net.load_state_dict(online_net.state_dict())

    def update_epsilon(self):
        """更新探索率（指数衰减）"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def update_learning_rate(self):
        """
        更新学习率（每回合调用一次）

        使用指数衰减，但不低于最小学习率
        注意：只有在实际训练发生后才更新学习率
        """
        # 只有训练步数>0时才更新学习率，避免在预填充阶段衰减
        if self.training_step == 0:
            return

        lr_min = 1e-6
        if self.current_lr > lr_min:
            self.lr_scheduler.step()
            # 获取当前学习率
            self.current_lr = self.optimizer.param_groups[0]['lr']
            # 确保不低于最小值
            if self.current_lr < lr_min:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_min
                self.current_lr = lr_min

    def save_model(self, path: str, episode: int = None):
        """
        保存模型

        Args:
            path: 保存路径
            episode: 当前回合数
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'current_lr': self.current_lr,
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }

        torch.save(checkpoint, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path: str) -> Dict:
        """
        加载模型

        Args:
            path: 模型路径

        Returns:
            检查点信息
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载学习率调度器状态
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if 'current_lr' in checkpoint:
            self.current_lr = checkpoint['current_lr']

        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.training_step = checkpoint.get('training_step', 0)

        if 'episode_rewards' in checkpoint:
            self.episode_rewards = checkpoint['episode_rewards']
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = checkpoint['episode_lengths']

        print(f"模型已从 {path} 加载")
        return checkpoint

    def get_training_stats(self) -> Dict:
        """
        获取训练统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'learning_rate': self.current_lr,
            'buffer_size': len(self.replay_buffer),
            'num_episodes': len(self.episode_rewards),
        }

        if self.losses:
            stats['avg_loss'] = np.mean(self.losses[-100:])

        if self.q_values:
            stats['avg_q_value'] = np.mean(self.q_values[-100:])

        if self.episode_rewards:
            stats['avg_reward'] = np.mean(self.episode_rewards[-100:])
            stats['max_reward'] = max(self.episode_rewards)

        if self.episode_lengths:
            stats['avg_length'] = np.mean(self.episode_lengths[-100:])

        return stats


class HierarchicalAgent:
    """
    分层智能体
    结合A*静态规划和D3QN动态调整的完整路径规划系统
    """

    def __init__(self):
        """初始化分层智能体"""
        # D3QN智能体
        self.d3qn_agent = D3QNAgent(use_per=True, use_astar=True)

        # A*规划器
        self.astar_planner = None
        self.global_path = None
        self.local_goal = None

        # 状态缓冲
        self.state_buffer = deque(maxlen=4)  # DEPTH_FRAME_STACK

        # 规划参数
        self.waypoint_threshold = 1.0  # 到达路径点的阈值
        self.replan_threshold = 2.0   # 偏离路径触发重规划的阈值
        self.look_ahead = 3           # 前瞻路径点数量

    def init_planner(self, obstacles: List[Tuple[float, float, float]] = None):
        """
        初始化路径规划器

        Args:
            obstacles: 静态障碍物列表 [(x, y, radius), ...]
        """
        # 创建栅格地图（考虑月球车体积进行障碍物膨胀）
        grid_map = create_grid_map(
            10.0,  # ENV_WIDTH
            10.0,  # ENV_HEIGHT
            0.1,   # GRID_RESOLUTION
            obstacles,
            robot_radius=0.2  # ROVER_SAFE_RADIUS
        )

        # 初始化A*规划器
        self.astar_planner = AStarPlanner(grid_map, 0.1)
        self.d3qn_agent.astar_planner = self.astar_planner

    def plan_path(self, start: Tuple[float, float],
                  goal: Tuple[float, float]) -> bool:
        """
        规划从起点到目标的全局路径

        Args:
            start: 起点坐标
            goal: 目标坐标

        Returns:
            是否成功规划路径
        """
        if self.astar_planner is None:
            print("警告: A*规划器未初始化")
            return False

        self.global_path = self.astar_planner.find_path(start, goal)

        if self.global_path:
            self.global_path = self.astar_planner.smooth_path(self.global_path)
            print(f"全局路径规划成功，共 {len(self.global_path)} 个路径点")
            return True
        else:
            print("全局路径规划失败")
            return False

    def get_local_goal(self, current_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        获取局部目标点

        Args:
            current_pos: 当前位置

        Returns:
            局部目标点坐标
        """
        if self.global_path is None:
            return None

        # 找到最近的路径点
        min_dist = float('inf')
        nearest_idx = 0
        for i, waypoint in enumerate(self.global_path):
            dist = np.sqrt(
                (current_pos[0] - waypoint[0]) ** 2 +
                (current_pos[1] - waypoint[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 选择前瞻路径点作为局部目标
        target_idx = min(nearest_idx + self.look_ahead, len(self.global_path) - 1)
        self.local_goal = self.global_path[target_idx]

        return self.local_goal

    def should_replan(self, current_pos: Tuple[float, float]) -> bool:
        """
        检查是否需要重新规划路径

        Args:
            current_pos: 当前位置

        Returns:
            是否需要重新规划
        """
        if self.global_path is None:
            return True

        # 计算到最近路径点的距离
        min_dist = float('inf')
        for waypoint in self.global_path:
            dist = np.sqrt(
                (current_pos[0] - waypoint[0]) ** 2 +
                (current_pos[1] - waypoint[1]) ** 2
            )
            min_dist = min(min_dist, dist)

        return min_dist > self.replan_threshold

    def select_action(self, state: np.ndarray,
                      additional_features: np.ndarray,
                      current_pos: Tuple[float, float],
                      training: bool = True) -> int:
        """
        选择动作

        结合全局规划和D3QN策略

        Args:
            state: 当前状态
            additional_features: 额外特征
            current_pos: 当前位置
            training: 是否训练模式

        Returns:
            选择的动作
        """
        # 获取局部目标
        local_goal = self.get_local_goal(current_pos)

        # 如果有局部目标，调整额外特征中的方向信息
        if local_goal is not None:
            dx = local_goal[0] - current_pos[0]
            dy = local_goal[1] - current_pos[1]
            local_distance = np.sqrt(dx ** 2 + dy ** 2)
            # 可以将局部目标信息融合到决策中

        # 使用D3QN选择动作
        return self.d3qn_agent.select_action(
            state, additional_features, training
        )

    def train_step(self) -> Optional[float]:
        """执行训练步骤"""
        return self.d3qn_agent.train_step()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """存储经验"""
        self.d3qn_agent.store_experience(state, action, reward, next_state, done)

    def update_epsilon(self):
        """更新探索率"""
        self.d3qn_agent.update_epsilon()

    def save_model(self, path: str, episode: int = None):
        """保存模型"""
        self.d3qn_agent.save_model(path, episode)

    def load_model(self, path: str):
        """加载模型"""
        return self.d3qn_agent.load_model(path)


# 测试代码
if __name__ == "__main__":
    print("=== 测试D3QN智能体 ===")

    # 创建D3QN智能体
    agent = D3QNAgent(use_per=True)
    print(f"智能体创建成功")
    print(f"设备: {agent.device}")
    print(f"初始ε: {agent.epsilon}")

    # 测试动作选择
    state = np.random.rand(80, 64, 4).astype(np.float32)
    features = np.array([0.5, 2.0], dtype=np.float32)
    action = agent.select_action(state, features, training=True)
    print(f"选择动作: {action}")

    # 添加测试经验
    print("\n添加测试经验...")
    for i in range(100):
        next_state = np.random.rand(80, 64, 4).astype(np.float32)
        reward = np.random.randn()
        done = i % 20 == 0
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state

    print(f"缓冲区大小: {len(agent.replay_buffer)}")

    # 测试训练
    print("\n测试训练步骤...")
    for i in range(10):
        loss = agent.train_step()
        if loss is not None:
            print(f"  Step {i + 1}: Loss = {loss:.4f}")

    # 获取统计信息
    stats = agent.get_training_stats()
    print(f"\n训练统计: {stats}")

    # 测试分层智能体
    print("\n=== 测试分层智能体 ===")
    hierarchical_agent = HierarchicalAgent()

    # 初始化规划器
    obstacles = [(3.0, 3.0, 0.5), (7.0, 7.0, 0.5)]
    hierarchical_agent.init_planner(obstacles)

    # 规划路径
    start = (0.5, 0.5)
    goal = (9.5, 9.5)
    success = hierarchical_agent.plan_path(start, goal)
    print(f"路径规划成功: {success}")

    if success:
        # 获取局部目标
        local_goal = hierarchical_agent.get_local_goal(start)
        print(f"局部目标: {local_goal}")

    print("\n智能体测试完成!")
