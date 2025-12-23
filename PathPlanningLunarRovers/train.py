# -*- coding: utf-8 -*-
"""
训练脚本 - A*-D3QN-Opt月球车路径规划训练
实现完整的训练流程，支持三种复杂度场景
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List

from config import config
from lunar_environment import LunarEnvironment
from d3qn_agent import D3QNAgent, HierarchicalAgent
from astar import create_grid_map
from utils import set_random_seed, create_directories, Logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A*-D3QN-Opt 月球车路径规划训练')

    parser.add_argument('--method', type=str, default='a_star_d3qn_opt',
                        choices=['d3qn', 'd3qn_per', 'a_star_d3qn_opt'],
                        help='训练方法: d3qn, d3qn_per, a_star_d3qn_opt')

    parser.add_argument('--stage', type=int, default=1,
                        choices=[1, 2, 3],
                        help='环境复杂度: 1=无障碍, 2=静态障碍, 3=动态障碍')

    parser.add_argument('--episodes', type=int, default=2000,
                        help='训练回合数')

    parser.add_argument('--render', action='store_true',
                        help='是否渲染环境')

    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')

    parser.add_argument('--load_model', type=str, default=None,
                        help='加载预训练模型路径')

    return parser.parse_args()


def create_agent(method: str) -> D3QNAgent:
    """
    根据方法创建智能体

    Args:
        method: 训练方法名称

    Returns:
        智能体实例
    """
    if method == 'd3qn':
        # 基础D3QN，不使用PER
        return D3QNAgent(use_per=False, use_astar=False)
    elif method == 'd3qn_per':
        # D3QN + 优先经验回放
        return D3QNAgent(use_per=True, use_astar=False)
    else:
        # A*-D3QN-Opt: 完整方法
        return D3QNAgent(use_per=True, use_astar=True)


def train_episode(env: LunarEnvironment, agent: D3QNAgent,
                  render: bool = False) -> Dict:
    """
    训练一个回合

    Args:
        env: 月球环境
        agent: D3QN智能体
        render: 是否渲染

    Returns:
        回合统计信息
    """
    # 重置环境
    state = env.reset()
    additional_features = env.get_additional_features()

    # 初始化A*规划器（如果使用）
    if agent.use_astar and agent.astar_planner is None:
        # 首次初始化：只用静态障碍物创建基础地图
        static_obstacles = []
        for obs in env.obstacles:
            if not obs.is_dynamic:
                static_obstacles.append((obs.x, obs.y, obs.radius))

        grid_map = create_grid_map(
            config.ENV_WIDTH, config.ENV_HEIGHT,
            config.GRID_RESOLUTION, static_obstacles,
            robot_radius=config.ROVER_SAFE_RADIUS
        )
        agent.init_astar_planner(grid_map, config.GRID_RESOLUTION)

    # 规划全局路径（包含动态障碍物当前位置）
    if agent.use_astar:
        # 获取所有障碍物（静态+动态）的当前位置
        all_obstacles = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
        # 更新栅格地图
        grid_map = create_grid_map(
            config.ENV_WIDTH, config.ENV_HEIGHT,
            config.GRID_RESOLUTION, all_obstacles,
            robot_radius=config.ROVER_SAFE_RADIUS
        )
        agent.astar_planner.update_map(grid_map)
        # 规划路径
        start = (env.rover.x, env.rover.y)
        goal = (env.target_x, env.target_y)
        path = agent.plan_global_path(start, goal)
        env.set_astar_path(path)

    # 回合统计
    total_reward = 0.0
    step_count = 0
    losses = []

    # 回合主循环
    done = False
    while not done:
        # 选择动作（传入当前位置和航向角，用于A*引导）
        current_pos = (env.rover.x, env.rover.y)
        current_theta = env.rover.theta
        action = agent.select_action(
            state, additional_features,
            current_pos=current_pos,
            current_theta=current_theta,
            training=True
        )

        # 执行动作
        next_state, reward, done, info = env.step(action)
        next_additional_features = env.get_additional_features()

        # 存储经验（包含additional_features，让网络学习利用A*路径信息）
        agent.store_experience(state, action, reward, next_state, done,
                              additional_features, next_additional_features)

        # 训练网络
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        # 更新状态
        state = next_state
        additional_features = next_additional_features
        total_reward += reward
        step_count += 1

        # 渲染
        if render:
            env.render()

        # 动态重规划A*路径（每10步更新，考虑动态障碍物移动）
        if agent.use_astar and step_count % 10 == 0:
            # 获取所有障碍物当前位置（含动态障碍物新位置）
            all_obstacles = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
            # 更新栅格地图
            grid_map = create_grid_map(
                config.ENV_WIDTH, config.ENV_HEIGHT,
                config.GRID_RESOLUTION, all_obstacles,
                robot_radius=config.ROVER_SAFE_RADIUS
            )
            agent.astar_planner.update_map(grid_map)
            # 重新规划路径
            current_pos = (env.rover.x, env.rover.y)
            goal = (env.target_x, env.target_y)
            path = agent.plan_global_path(current_pos, goal)
            env.set_astar_path(path)

    # 更新探索率
    agent.update_epsilon()

    # 更新学习率（每回合衰减）
    agent.update_learning_rate()

    # 记录回合统计
    agent.episode_rewards.append(total_reward)
    agent.episode_lengths.append(step_count)

    return {
        'reward': total_reward,
        'steps': step_count,
        'collision': info.get('collision', False),
        'reached_target': info.get('reached_target', False),
        'avg_loss': np.mean(losses) if losses else 0.0,
        'epsilon': agent.epsilon,
        'learning_rate': agent.current_lr,
    }


def train(args):
    """
    主训练函数

    Args:
        args: 命令行参数
    """
    # 设置随机种子
    set_random_seed(args.seed)

    # 创建目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.method}_stage{args.stage}_{timestamp}"
    save_dir = os.path.join(args.save_dir, exp_name)
    create_directories(save_dir)

    # 创建日志记录器
    logger = Logger(os.path.join(save_dir, 'training.log'))
    logger.log(f"实验配置:")
    logger.log(f"  方法: {args.method}")
    logger.log(f"  阶段: {args.stage}")
    logger.log(f"  回合数: {args.episodes}")
    logger.log(f"  随机种子: {args.seed}")

    # 创建环境
    env = LunarEnvironment(stage=args.stage, render=args.render)
    logger.log(f"环境创建成功 - Stage {args.stage}")

    # 创建智能体
    agent = create_agent(args.method)
    logger.log(f"智能体创建成功 - {args.method}")

    # 加载预训练模型
    if args.load_model:
        agent.load_model(args.load_model)
        logger.log(f"加载预训练模型: {args.load_model}")

    # 训练统计
    all_rewards = []
    all_lengths = []
    all_losses = []
    success_history = []  # 每个回合是否成功的记录
    collisions = 0
    successes = 0
    best_avg_reward = float('-inf')  # 用于保存最佳模型

    # 开始时间
    start_time = time.time()

    # 训练主循环
    logger.log("\n开始训练...")
    for episode in range(1, args.episodes + 1):
        # 训练一个回合
        episode_stats = train_episode(env, agent, args.render)

        # 记录统计
        all_rewards.append(episode_stats['reward'])
        all_lengths.append(episode_stats['steps'])
        all_losses.append(episode_stats['avg_loss'])

        if episode_stats['collision']:
            collisions += 1
        if episode_stats['reached_target']:
            successes += 1
            success_history.append(True)
        else:
            success_history.append(False)

        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_length = np.mean(all_lengths[-100:])
            avg_loss = np.mean(all_losses[-100:]) if all_losses else 0
            current_lr = episode_stats.get('learning_rate', config.LEARNING_RATE)

            elapsed = time.time() - start_time
            logger.log(
                f"Episode {episode}/{args.episodes} | "
                f"Reward: {episode_stats['reward']:.2f} (Avg: {avg_reward:.2f}) | "
                f"Steps: {episode_stats['steps']} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {episode_stats['epsilon']:.3f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        # 保存模型
        if episode % config.SAVE_MODEL_FREQ == 0:
            model_path = os.path.join(save_dir, f'model_episode_{episode}.pth')
            agent.save_model(model_path, episode)

        # 保存最佳模型（基于最近100回合的平均奖励）
        if episode >= 100:
            current_avg_reward = np.mean(all_rewards[-100:])
            if current_avg_reward > best_avg_reward:
                best_avg_reward = current_avg_reward
                best_model_path = os.path.join(save_dir, 'model_best.pth')
                agent.save_model(best_model_path, episode)
                logger.log(f"  >> 保存最佳模型! Avg Reward: {best_avg_reward:.2f}")

    # 训练结束
    total_time = time.time() - start_time
    logger.log(f"\n训练完成!")
    logger.log(f"  总时间: {total_time / 3600:.2f} 小时")
    logger.log(f"  碰撞次数: {collisions}")
    logger.log(f"  成功次数: {successes}")
    logger.log(f"  成功率: {successes / args.episodes * 100:.2f}%")
    logger.log(f"  平均奖励: {np.mean(all_rewards):.2f}")
    logger.log(f"  平均步数: {np.mean(all_lengths):.2f}")

    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'model_final.pth')
    agent.save_model(final_model_path, args.episodes)

    # 绘制训练曲线（三个独立的收敛图）
    plot_training_curves(all_rewards, all_lengths, all_losses, save_dir, success_history)

    # 关闭环境
    env.close()

    return {
        'rewards': all_rewards,
        'lengths': all_lengths,
        'losses': all_losses,
        'collisions': collisions,
        'successes': successes,
        'total_time': total_time,
    }


def plot_training_curves(rewards: List[float], lengths: List[int],
                         losses: List[float], save_dir: str,
                         success_history: List[bool] = None):
    """
    绘制三个独立的训练收敛曲线

    Args:
        rewards: 回合奖励列表
        lengths: 回合长度列表
        losses: 损失列表
        save_dir: 保存目录
        success_history: 每个回合是否成功的列表
    """
    window = 50  # 移动平均窗口

    # ==================== 1. 奖励收敛图 ====================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(rewards)), moving_avg,
                 color='blue', linewidth=2, label=f'{window}-Episode Moving Average')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Reward Convergence', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_convergence.png'), dpi=150)
    plt.close(fig1)

    # ==================== 2. 步长收敛图 ====================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(lengths, alpha=0.3, color='green', label='Raw Episode Length')
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(lengths)), moving_avg,
                 color='green', linewidth=2, label=f'{window}-Episode Moving Average')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Length (Steps)', fontsize=12)
    ax2.set_title('Episode Length Convergence', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'episode_length_convergence.png'), dpi=150)
    plt.close(fig2)

    # ==================== 3. 成功率收敛图 ====================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if success_history is not None and len(success_history) > 0:
        # 计算滑动窗口成功率
        success_rate = []
        for i in range(len(success_history)):
            start_idx = max(0, i - window + 1)
            window_successes = success_history[start_idx:i + 1]
            rate = sum(window_successes) / len(window_successes) * 100
            success_rate.append(rate)

        ax3.plot(success_rate, color='red', linewidth=2,
                 label=f'{window}-Episode Success Rate')
        ax3.fill_between(range(len(success_rate)), success_rate, alpha=0.3, color='red')
    else:
        # 如果没有成功率历史，显示提示
        ax3.text(0.5, 0.5, 'No success data available',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=14)

    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Success Rate (%)', fontsize=12)
    ax3.set_title('Success Rate Convergence', fontsize=14)
    ax3.set_ylim(0, 100)
    if success_history is not None and len(success_history) > 0:
        ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'success_rate_convergence.png'), dpi=150)
    plt.close(fig3)

    print(f"三个收敛图已保存到: {save_dir}")


def compare_methods():
    """比较三种方法在不同场景下的性能"""
    methods = ['d3qn', 'd3qn_per', 'a_star_d3qn_opt']
    stages = [1, 2, 3]

    results = {}

    for stage in stages:
        results[stage] = {}
        for method in methods:
            print(f"\n{'=' * 50}")
            print(f"训练: {method} on Stage {stage}")
            print('=' * 50)

            args = argparse.Namespace(
                method=method,
                stage=stage,
                episodes=100,  # 减少回合数用于快速测试
                render=False,
                seed=42,
                save_dir='./comparison_results',
                load_model=None
            )

            result = train(args)
            results[stage][method] = result

    # 绘制比较图表
    plot_comparison(results)

    return results


def plot_comparison(results: Dict):
    """
    绘制方法比较图表

    Args:
        results: 比较结果字典
    """
    methods = ['d3qn', 'd3qn_per', 'a_star_d3qn_opt']
    stages = [1, 2, 3]
    stage_names = ['Low Complexity', 'Medium Complexity', 'High Complexity']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        ax = axes[i]

        for method in methods:
            rewards = results[stage][method]['rewards']
            # 计算移动平均
            window = 20
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax.plot(range(window - 1, len(rewards)), moving_avg, label=method)
            else:
                ax.plot(rewards, label=method)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title(f'{stage_name} (Stage {stage})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./comparison_results/method_comparison.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 确保utils模块存在
    try:
        from utils import set_random_seed, create_directories, Logger
    except ImportError:
        # 如果utils未创建，使用内联实现
        import random
        import torch

        def set_random_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        def create_directories(path):
            os.makedirs(path, exist_ok=True)

        class Logger:
            def __init__(self, log_file):
                self.log_file = log_file
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

            def log(self, message):
                print(message)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')

    # 开始训练
    train(args)
