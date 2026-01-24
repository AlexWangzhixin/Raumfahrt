#!/usr/bin/env python3
"""
启动RL训练脚本
基于D3QN算法的路径规划训练
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
# 直接导入 utils.py 文件
import sys
import os
import importlib.util

# 导入 utils.py 文件
spec = importlib.util.spec_from_file_location("utils", os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'utils.py'))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

from src.planning.local_planner.agent import D3QNAgent
from src.config.config import TRAINING_PARAMS, PLANNING_PARAMS

class RLTraining:
    """
    RL训练类
    负责D3QN智能体的训练过程
    """
    
    def __init__(self):
        """
        初始化训练器
        """
        self.agent = D3QNAgent(
            use_per=True, 
            use_astar=True
        )
        self.training_params = TRAINING_PARAMS
        self.planning_params = PLANNING_PARAMS
        self.timestamp = utils.get_timestamp()
        
        # 创建训练日志目录
        self.log_dir = os.path.join(
            'outputs', 'logs', f'training_{self.timestamp}'
        )
        utils.ensure_directory(self.log_dir)
        
        # 创建检查点目录
        self.checkpoint_dir = os.path.join(
            'outputs', 'checkpoints', f'training_{self.timestamp}'
        )
        utils.ensure_directory(self.checkpoint_dir)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_steps': [],
            'q_values': [],
        }
        
        # 环境参数
        self.start_position = np.array([0.0, 0.0])
        self.goal_position = np.array([10.0, 10.0])
        self.current_position = self.start_position.copy()
        self.episode_steps = 0
    
    def train(self):
        """
        执行训练过程
        """
        print(f"开始训练 D3QN 智能体...")
        print(f"训练参数: {self.training_params}")
        print(f"日志目录: {self.log_dir}")
        
        for episode in range(self.training_params['NUM_EPISODES']):
            # 初始化 episode
            state = self._reset_environment()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = self._step_environment(action)
                
                # 存储经验
                self.agent.store_experience(
                    state, action, reward, next_state, done
                )
                
                # 学习
                loss = self.agent.train_step()
                q_value = None
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # 检查是否达到最大步数
                if episode_length >= self.training_params.get('MAX_EPISODE_STEPS', 1000):
                    done = True
            
            # 记录统计信息
            self.stats['episode_rewards'].append(episode_reward)
            self.stats['episode_lengths'].append(episode_length)
            self.stats['episode_steps'].append(episode_length)
            if q_value is not None:
                self.stats['q_values'].append(q_value)
            
            # 打印训练信息
            if episode % 100 == 0:
                avg_reward = np.mean(self.stats['episode_rewards'][-100:])
                avg_length = np.mean(self.stats['episode_lengths'][-100:])
                print(f"Episode {episode}: 平均奖励 = {avg_reward:.2f}, 平均长度 = {avg_length:.2f}")
            
            # 保存检查点
            if episode % self.training_params['EVAL_FREQUENCY'] == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f'checkpoint_{episode}.pt'
                )
                self.agent.save_model(checkpoint_path, episode)
                print(f"保存检查点到: {checkpoint_path}")
            
            # 评估
            if episode % self.training_params['EVAL_FREQUENCY'] == 0:
                self.evaluate(episode)
        
        # 训练完成
        print("训练完成！")
        self._save_stats()
    
    def _reset_environment(self):
        """
        重置环境
        
        Returns:
            初始状态
        """
        # 重置位置
        self.current_position = self.start_position.copy()
        self.episode_steps = 0
        
        # 计算初始距离
        initial_distance = np.linalg.norm(self.current_position - self.goal_position)
        
        # 这里应该返回环境的初始状态
        # 暂时返回一个模拟的深度图状态
        # 形状: (height, width, channels) = (80, 64, 4)
        return np.random.rand(80, 64, 4)
    
    def _step_environment(self, action):
        """
        执行环境步骤
        
        Args:
            action: 动作
        
        Returns:
            next_state, reward, done, info
        """
        # 记录前一步的距离
        prev_distance = np.linalg.norm(self.current_position - self.goal_position)
        
        # 模拟动作执行（根据动作更新位置）
        # 简单的动作映射：0-8 对应不同的移动方向
        action_mapping = [
            (-0.1, -0.1),  # 0: 向左下移动
            (-0.1, 0.0),   # 1: 向左移动
            (-0.1, 0.1),   # 2: 向左上移动
            (0.0, -0.1),   # 3: 向下移动
            (0.0, 0.0),    # 4: 不动
            (0.0, 0.1),    # 5: 向上移动
            (0.1, -0.1),   # 6: 向右下移动
            (0.1, 0.0),    # 7: 向右移动
            (0.1, 0.1)     # 8: 向右上移动
        ]
        
        # 更新位置
        delta = action_mapping[action]
        self.current_position[0] += delta[0]
        self.current_position[1] += delta[1]
        
        # 计算新的距离
        new_distance = np.linalg.norm(self.current_position - self.goal_position)
        
        # 计算奖励
        # 奖励 = 距离减少量 - 步数惩罚
        distance_reward = prev_distance - new_distance
        step_penalty = 0.01
        reward = distance_reward - step_penalty
        
        # 检查是否到达目标
        done = new_distance < 0.5
        
        # 检查是否达到最大步数
        self.episode_steps += 1
        if self.episode_steps >= self.training_params.get('MAX_EPISODE_STEPS', 1000):
            done = True
        
        # 生成下一个状态
        next_state = np.random.rand(80, 64, 4)
        
        # 生成信息
        info = {
            'distance_to_goal': new_distance,
            'current_position': self.current_position.tolist(),
            'steps_taken': self.episode_steps
        }
        
        return next_state, reward, done, info
    
    def evaluate(self, episode):
        """
        评估智能体性能
        
        Args:
            episode: 当前 episode
        """
        print(f"评估智能体在 episode {episode}...")
        # 这里应该实现评估逻辑
    
    def _save_stats(self):
        """
        保存训练统计信息
        """
        stats_path = os.path.join(self.log_dir, 'training_stats.npz')
        np.savez(
            stats_path,
            episode_rewards=self.stats['episode_rewards'],
            episode_lengths=self.stats['episode_lengths'],
            episode_steps=self.stats['episode_steps'],
            q_values=self.stats['q_values']
        )
        print(f"保存训练统计到: {stats_path}")

def main():
    """
    主函数
    """
    trainer = RLTraining()
    trainer.train()

if __name__ == '__main__':
    main()
