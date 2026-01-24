#!/usr/bin/env python3
"""
启动RL训练脚本
基于D3QN算法的路径规划训练
"""

import sys
import os
import numpy as np
import torch
from src.core.utils import get_timestamp, ensure_directory
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
        self.timestamp = get_timestamp()
        
        # 创建训练日志目录
        self.log_dir = os.path.join(
            'outputs', 'logs', f'training_{self.timestamp}'
        )
        ensure_directory(self.log_dir)
        
        # 创建检查点目录
        self.checkpoint_dir = os.path.join(
            'outputs', 'checkpoints', f'training_{self.timestamp}'
        )
        ensure_directory(self.checkpoint_dir)
        
        # 训练统计
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_steps': [],
            'q_values': [],
        }
    
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
                self.agent.store_transition(
                    state, action, reward, next_state, done
                )
                
                # 学习
                loss, q_value = self.agent.learn()
                
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
                self.agent.save(checkpoint_path)
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
        # 这里应该返回环境的初始状态
        # 暂时返回一个模拟状态
        return np.random.rand(8)  # 假设状态是8维的
    
    def _step_environment(self, action):
        """
        执行环境步骤
        
        Args:
            action: 动作
        
        Returns:
            next_state, reward, done, info
        """
        # 这里应该执行实际的环境步骤
        # 暂时返回模拟数据
        next_state = np.random.rand(8)
        reward = np.random.randn()
        done = np.random.random() < 0.1  # 10% 的概率结束
        info = {}
        
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
