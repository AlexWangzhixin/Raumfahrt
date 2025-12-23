# -*- coding: utf-8 -*-
"""
D3QN神经网络模块 (Dueling Double Deep Q-Network)
结合Dueling网络架构和Double Q-Learning机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config import config


class DuelingDQN(nn.Module):
    """
    Dueling DQN网络架构

    将Q值分解为状态价值V(s)和优势函数A(s,a):
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    网络结构:
    - 卷积层处理深度图像
    - 全连接层融合特征
    - 分离的价值流和优势流
    """

    def __init__(self, input_shape: Tuple[int, int, int] = None,
                 num_actions: int = None,
                 additional_features: int = 16):  # 16个特征: 基础8维 + 8方向障碍物距离
        """
        初始化Dueling DQN网络

        Args:
            input_shape: 输入深度图像形状 (height, width, channels)
            num_actions: 动作空间大小
            additional_features: 额外特征数量（目标方向角和距离）
        """
        super(DuelingDQN, self).__init__()

        # 使用配置参数或默认值
        if input_shape is None:
            height, width = config.DEPTH_IMAGE_SIZE
            channels = config.DEPTH_FRAME_STACK
            input_shape = (height, width, channels)

        self.num_actions = num_actions or config.ACTION_SPACE_SIZE
        self.additional_features = additional_features

        # 输入形状: (batch, channels, height, width)
        in_channels = input_shape[2]

        # ==================== 卷积层 ====================
        # Layer 1: 32个8x8卷积核, stride=4, ReLU激活
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)

        # Layer 2: 64个4x4卷积核, stride=2, ReLU激活
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # Layer 3: 64个3x3卷积核, stride=1, ReLU激活
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 计算卷积层输出尺寸
        conv_out_size = self._get_conv_output_size(input_shape)

        # ==================== 全连接层 ====================
        # 特征融合层（卷积特征 + 额外特征）
        self.fc_input_dim = conv_out_size + additional_features

        # 共享全连接层
        self.fc_shared = nn.Linear(self.fc_input_dim, config.FC_HIDDEN_DIM)

        # ==================== Dueling分支 ====================
        # 价值流 (Value Stream): 评估状态的价值
        self.value_stream = nn.Sequential(
            nn.Linear(config.FC_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出单个状态价值
        )

        # 优势流 (Advantage Stream): 评估各动作的相对优势
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.FC_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)  # 输出每个动作的优势
        )

        # 初始化权重
        self._init_weights()

    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """
        计算卷积层输出的展平尺寸

        Args:
            input_shape: 输入形状 (height, width, channels)

        Returns:
            展平后的特征维度
        """
        # 创建虚拟输入计算输出尺寸
        height, width, channels = input_shape
        dummy_input = torch.zeros(1, channels, height, width)

        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return int(np.prod(x.size()[1:]))

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化（适用于ReLU激活）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, depth_images: torch.Tensor,
                additional_features: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            depth_images: 深度图像张量 (batch, channels, height, width)
            additional_features: 额外特征 (batch, 2) [目标方向角, 目标距离]

        Returns:
            Q值张量 (batch, num_actions)
        """
        # ==================== 卷积特征提取 ====================
        x = F.relu(self.conv1(depth_images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平卷积输出
        x = x.reshape(x.size(0), -1)

        # ==================== 特征融合 ====================
        if additional_features is not None:
            # 拼接卷积特征和额外特征
            x = torch.cat([x, additional_features], dim=1)
        else:
            # 如果没有额外特征，填充零
            batch_size = x.size(0)
            zeros = torch.zeros(batch_size, self.additional_features, device=x.device)
            x = torch.cat([x, zeros], dim=1)

        # 共享全连接层
        x = F.relu(self.fc_shared(x))

        # ==================== Dueling架构 ====================
        # 计算状态价值
        value = self.value_stream(x)

        # 计算动作优势
        advantage = self.advantage_stream(x)

        # Q值 = V(s) + (A(s,a) - mean(A(s,a')))
        # 减去均值以增加训练稳定性
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(self, state: np.ndarray, additional_features: np.ndarray = None,
                   epsilon: float = 0.0, device: torch.device = None) -> int:
        """
        根据ε-greedy策略选择动作

        Args:
            state: 深度图像状态
            additional_features: 额外特征 [方向角, 距离]
            epsilon: 探索率
            device: 计算设备

        Returns:
            选择的动作索引
        """
        if device is None:
            device = next(self.parameters()).device

        # ε-greedy探索
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)

        # 贪婪选择
        with torch.no_grad():
            # 转换为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # 调整维度顺序: (batch, height, width, channels) -> (batch, channels, height, width)
            if state_tensor.dim() == 4 and state_tensor.size(-1) == config.DEPTH_FRAME_STACK:
                state_tensor = state_tensor.permute(0, 3, 1, 2)

            if additional_features is not None:
                additional_tensor = torch.FloatTensor(additional_features).unsqueeze(0).to(device)
            else:
                additional_tensor = None

            q_values = self.forward(state_tensor, additional_tensor)
            action = q_values.argmax(dim=1).item()

        return action


class D3QNLoss(nn.Module):
    """
    D3QN损失函数

    结合Double DQN的目标值计算和Dueling架构:
    - 使用在线网络选择动作
    - 使用目标网络评估Q值
    - 支持优先经验回放的重要性采样权重
    """

    def __init__(self, gamma: float = None):
        """
        初始化损失函数

        Args:
            gamma: 折扣因子
        """
        super(D3QNLoss, self).__init__()
        self.gamma = gamma or config.GAMMA

    def forward(self, online_net: DuelingDQN, target_net: DuelingDQN,
                states: torch.Tensor, actions: torch.Tensor,
                rewards: torch.Tensor, next_states: torch.Tensor,
                dones: torch.Tensor, additional_features: torch.Tensor = None,
                next_additional_features: torch.Tensor = None,
                weights: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算D3QN损失

        TD误差公式: δ = r + γ * Q_target(s', argmax_a Q_online(s', a)) - Q_online(s, a)

        Args:
            online_net: 在线网络
            target_net: 目标网络
            states: 当前状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一状态批次
            dones: 终止标志批次
            additional_features: 当前状态的额外特征
            next_additional_features: 下一状态的额外特征
            weights: 重要性采样权重（用于PER）

        Returns:
            (loss, td_errors): 损失值和TD误差
        """
        batch_size = states.size(0)

        # ==================== 计算当前Q值 ====================
        # Q_online(s, a)
        current_q_values = online_net(states, additional_features)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ==================== Double DQN目标值计算 ====================
        with torch.no_grad():
            # 使用在线网络选择下一状态的最佳动作
            # a' = argmax_a Q_online(s', a)
            next_q_values_online = online_net(next_states, next_additional_features)
            next_actions = next_q_values_online.argmax(dim=1)

            # 使用目标网络评估选定动作的Q值
            # Q_target(s', a')
            next_q_values_target = target_net(next_states, next_additional_features)
            next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # 计算目标Q值
            # Q_target = r + γ * Q_target(s', a') * (1 - done)
            target_q = rewards + self.gamma * next_q * (1 - dones.float())

        # ==================== 计算TD误差和损失 ====================
        td_errors = target_q - current_q

        if weights is not None:
            # 带重要性采样权重的损失（用于PER）
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        else:
            # 标准Huber损失
            loss = F.smooth_l1_loss(current_q, target_q)

        return loss, td_errors.detach()


def create_networks(device: torch.device = None) -> Tuple[DuelingDQN, DuelingDQN]:
    """
    创建在线网络和目标网络

    Args:
        device: 计算设备

    Returns:
        (online_net, target_net)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建网络
    online_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)

    # 复制权重到目标网络
    target_net.load_state_dict(online_net.state_dict())

    # 目标网络不需要梯度
    for param in target_net.parameters():
        param.requires_grad = False

    return online_net, target_net


def soft_update(online_net: DuelingDQN, target_net: DuelingDQN, tau: float = 0.005):
    """
    软更新目标网络参数

    θ_target = τ * θ_online + (1 - τ) * θ_target

    Args:
        online_net: 在线网络
        target_net: 目标网络
        tau: 软更新系数
    """
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)


def hard_update(online_net: DuelingDQN, target_net: DuelingDQN):
    """
    硬更新目标网络参数（完全复制）

    Args:
        online_net: 在线网络
        target_net: 目标网络
    """
    target_net.load_state_dict(online_net.state_dict())


# 测试代码
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建网络
    online_net, target_net = create_networks(device)
    print(f"网络创建成功")

    # 打印网络结构
    print("\n网络结构:")
    print(online_net)

    # 计算参数量
    total_params = sum(p.numel() for p in online_net.parameters())
    trainable_params = sum(p.numel() for p in online_net.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    batch_size = 32
    height, width = config.DEPTH_IMAGE_SIZE
    channels = config.DEPTH_FRAME_STACK

    # 创建测试输入
    test_depth = torch.randn(batch_size, channels, height, width).to(device)
    test_features = torch.randn(batch_size, 2).to(device)

    # 前向传播
    q_values = online_net(test_depth, test_features)
    print(f"\n输入形状: 深度图 {test_depth.shape}, 额外特征 {test_features.shape}")
    print(f"输出Q值形状: {q_values.shape}")

    # 测试动作选择
    state = np.random.rand(height, width, channels).astype(np.float32)
    features = np.array([0.5, 2.0], dtype=np.float32)
    action = online_net.get_action(state, features, epsilon=0.1, device=device)
    print(f"\n选择动作: {action}")

    # 测试损失计算
    loss_fn = D3QNLoss()
    states = torch.randn(batch_size, channels, height, width).to(device)
    next_states = torch.randn(batch_size, channels, height, width).to(device)
    actions = torch.randint(0, config.ACTION_SPACE_SIZE, (batch_size,)).to(device)
    rewards = torch.randn(batch_size).to(device)
    dones = torch.zeros(batch_size).to(device)
    weights = torch.ones(batch_size).to(device)

    loss, td_errors = loss_fn(online_net, target_net, states, actions, rewards,
                              next_states, dones, test_features, test_features, weights)
    print(f"\n损失值: {loss.item():.4f}")
    print(f"TD误差均值: {td_errors.abs().mean().item():.4f}")
