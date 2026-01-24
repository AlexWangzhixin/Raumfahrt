# -*- coding: utf-8 -*-
"""
D3QN网络结构模块
实现Dueling Double DQN网络和损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingDQN(nn.Module):
    """
    Dueling Double DQN网络
    分离状态价值和优势函数，提高价值估计精度
    """

    def __init__(self, action_space_size=9):
        """
        初始化网络结构

        Args:
            action_space_size: 动作空间大小
        """
        super(DuelingDQN, self).__init__()
        
        # 深度图处理网络（卷积部分）
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 输入：(4, 80, 64)
            nn.ReLU(),
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 第三层卷积
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 计算卷积层输出尺寸
        conv_out_size = self._calculate_conv_output_size()

        # 额外特征处理（目标方向、距离等）
        self.feature_fc = nn.Linear(6, 32)  # 6个额外特征

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size + 32, 512),  # 融合卷积特征和额外特征
            nn.ReLU(),
        )

        # Dueling Architecture
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 单个状态价值
        )

        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size)  # 每个动作的优势值
        )

    def _calculate_conv_output_size(self):
        """
        计算卷积层输出尺寸

        Returns:
            卷积层输出的特征数量
        """
        # 创建一个虚拟输入
        dummy_input = torch.zeros(1, 4, 80, 64)
        output = self.conv_layers(dummy_input)
        return output.view(1, -1).size(1)

    def forward(self, state, additional_features=None):
        """
        前向传播

        Args:
            state: 深度图状态 (batch, height, width, channels)
            additional_features: 额外特征 [目标方向, 目标距离, A*方向, A*距离, 障碍物方向, 障碍物距离]

        Returns:
            Q值 (batch, action_space_size)
        """
        # 调整维度: (batch, height, width, channels) -> (batch, channels, height, width)
        if state.dim() == 4 and state.size(-1) == 4:
            state = state.permute(0, 3, 1, 2)

        # 处理深度图
        conv_output = self.conv_layers(state)
        conv_flat = conv_output.view(conv_output.size(0), -1)

        # 处理额外特征
        if additional_features is not None:
            feature_output = F.relu(self.feature_fc(additional_features))
            # 融合特征
            combined = torch.cat([conv_flat, feature_output], dim=1)
        else:
            combined = conv_flat

        # 全连接层
        fc_output = self.fc_layers(combined)

        # Dueling Architecture
        value = self.value_stream(fc_output)
        advantage = self.advantage_stream(fc_output)

        # 计算Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(self, state, additional_features=None, epsilon=0.0, device='cpu'):
        """
        获取动作

        Args:
            state: 当前状态
            additional_features: 额外特征
            epsilon: 探索率
            device: 设备

        Returns:
            选择的动作索引
        """
        if np.random.random() < epsilon:
            # 随机探索
            return np.random.randint(0, self.advantage_stream[-1].out_features)
        else:
            # 贪婪选择
            with torch.no_grad():
                # 转换为张量
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                if additional_features is not None:
                    features_tensor = torch.FloatTensor(additional_features).unsqueeze(0).to(device)
                else:
                    features_tensor = None
                
                # 前向传播
                q_values = self.forward(state_tensor, features_tensor)
                # 选择Q值最大的动作
                return q_values.argmax().item()


class D3QNLoss:
    """
    D3QN损失函数
    实现Double DQN的目标Q值计算
    """

    def __init__(self, gamma=0.99):
        """
        初始化损失函数

        Args:
            gamma: 折扣因子
        """
        self.gamma = gamma

    def __call__(self, online_net, target_net, states, actions, rewards, next_states, dones,
                  additional_features=None, next_additional_features=None, weights=None):
        """
        计算损失

        Args:
            online_net: 在线网络
            target_net: 目标网络
            states: 当前状态
            actions: 执行的动作
            rewards: 获得的奖励
            next_states: 下一状态
            dones: 是否终止
            additional_features: 当前状态的额外特征
            next_additional_features: 下一状态的额外特征
            weights: 优先经验回放的权重

        Returns:
            损失值和TD误差
        """
        # 计算当前Q值
        current_q = online_net(states, additional_features)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: 使用在线网络选择动作，目标网络计算价值
        with torch.no_grad():
            # 在线网络选择动作
            next_q_online = online_net(next_states, next_additional_features)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            # 目标网络计算价值
            next_q_target = target_net(next_states, next_additional_features)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            
            # 计算目标Q值
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算TD误差
        td_errors = current_q - target_q

        # 计算损失
        if weights is not None:
            # 加权损失（用于优先经验回放）
            loss = (td_errors.pow(2) * weights).mean()
        else:
            # 均方误差损失
            loss = td_errors.pow(2).mean()

        return loss, td_errors


def create_networks(device):
    """
    创建在线网络和目标网络

    Args:
        device: 设备

    Returns:
        online_net, target_net: 在线网络和目标网络
    """
    online_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    
    # 初始时目标网络参数与在线网络相同
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()  # 目标网络设置为评估模式
    
    return online_net, target_net


def hard_update(source_net, target_net):
    """
    硬更新目标网络参数

    Args:
        source_net: 源网络（在线网络）
        target_net: 目标网络
    """
    target_net.load_state_dict(source_net.state_dict())


def soft_update(source_net, target_net, tau=0.001):
    """
    软更新目标网络参数

    Args:
        source_net: 源网络（在线网络）
        target_net: 目标网络
        tau: 更新比例
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# 测试代码
if __name__ == "__main__":
    # 测试网络创建
    print("测试Dueling DQN网络")
    net = DuelingDQN(action_space_size=9)
    print(f"网络创建成功")
    print(f"网络参数数量: {sum(p.numel() for p in net.parameters())}")

    # 测试前向传播
    # 创建虚拟输入
    state = torch.randn(1, 4, 80, 64)  # (batch, channels, height, width)
    features = torch.randn(1, 6)  # 额外特征

    # 前向传播
    q_values = net(state, features)
    print(f"Q值输出形状: {q_values.shape}")
    print(f"Q值: {q_values.detach().numpy()}")

    # 测试动作选择
    action = net.get_action(state.squeeze(0), features.squeeze(0), epsilon=0.0)
    print(f"选择的动作: {action}")

    # 测试损失函数
    print("\n测试损失函数")
    loss_fn = D3QNLoss(gamma=0.99)

    # 创建批量输入
    batch_size = 32
    states = torch.randn(batch_size, 4, 80, 64)
    actions = torch.randint(0, 9, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 80, 64)
    dones = torch.zeros(batch_size)
    batch_features = torch.randn(batch_size, 6)
    next_batch_features = torch.randn(batch_size, 6)

    # 创建在线网络和目标网络
    online_net, target_net = create_networks('cpu')

    # 计算损失
    loss, td_errors = loss_fn(
        online_net, target_net,
        states, actions, rewards, next_states, dones,
        additional_features=batch_features,
        next_additional_features=next_batch_features
    )

    print(f"损失值: {loss.item()}")
    print(f"TD误差形状: {td_errors.shape}")

    # 测试网络更新
    print("\n测试网络更新")
    # 硬更新
    hard_update(online_net, target_net)
    print("硬更新完成")

    # 软更新
    soft_update(online_net, target_net, tau=0.01)
    print("软更新完成")

    print("\n网络测试完成!")
