# -*- coding: utf-8 -*-
"""
测试脚本 - A*-D3QN-Opt月球车路径规划测试与评估
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from config import config
from lunar_environment import LunarEnvironment
from d3qn_agent import D3QNAgent
from astar import create_grid_map


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A*-D3QN-Opt 月球车路径规划测试')

    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')

    parser.add_argument('--stage', type=int, default=1,
                        choices=[1, 2, 3],
                        help='测试环境阶段')

    parser.add_argument('--episodes', type=int, default=100,
                        help='测试回合数')

    parser.add_argument('--render', action='store_true',
                        help='是否渲染')

    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='结果保存目录')

    parser.add_argument('--record_path', action='store_true',
                        help='是否记录路径轨迹')

    return parser.parse_args()


def test_episode(env: LunarEnvironment, agent: D3QNAgent,
                 render: bool = False,
                 record_path: bool = False) -> Dict:
    """
    测试一个回合

    Args:
        env: 月球环境
        agent: D3QN智能体
        render: 是否渲染
        record_path: 是否记录路径

    Returns:
        测试统计信息
    """
    # 重置环境
    state = env.reset()
    additional_features = env.get_additional_features()

    # 记录轨迹
    trajectory = [(env.rover.x, env.rover.y)] if record_path else None

    # 回合统计
    total_reward = 0.0
    step_count = 0
    start_time = time.time()

    done = False
    while not done:
        # 选择动作（评估模式，不探索）
        action = agent.select_action(state, additional_features, training=False)

        # 执行动作
        next_state, reward, done, info = env.step(action)
        next_additional_features = env.get_additional_features()

        # 更新状态
        state = next_state
        additional_features = next_additional_features
        total_reward += reward
        step_count += 1

        # 记录轨迹
        if record_path:
            trajectory.append((env.rover.x, env.rover.y))

        # 渲染
        if render:
            env.render()

    # 计算执行时间
    execution_time = time.time() - start_time

    # 计算路径长度
    path_length = 0.0
    if trajectory:
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            path_length += np.sqrt(dx ** 2 + dy ** 2)

    return {
        'reward': total_reward,
        'steps': step_count,
        'collision': info.get('collision', False),
        'reached_target': info.get('reached_target', False),
        'out_of_bounds': info.get('out_of_bounds', False),
        'execution_time': execution_time,
        'path_length': path_length,
        'trajectory': trajectory,
        'energy_consumed': info.get('energy_consumed', 0.0),
    }


def run_tests(args) -> Dict:
    """
    运行测试

    Args:
        args: 命令行参数

    Returns:
        测试结果
    """
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建环境
    env = LunarEnvironment(stage=args.stage, render=args.render)
    print(f"环境创建成功 - Stage {args.stage}")

    # 创建并加载智能体
    agent = D3QNAgent(use_per=True, use_astar=True)
    agent.load_model(args.model)
    agent.epsilon = 0.0  # 测试时不探索
    print(f"模型加载成功: {args.model}")

    # 初始化A*规划器
    obstacles = []
    for obs in env.obstacles:
        if not obs.is_dynamic:
            obstacles.append((obs.x, obs.y, obs.radius))
    grid_map = create_grid_map(config.ENV_WIDTH, config.ENV_HEIGHT,
                                config.GRID_RESOLUTION, obstacles)
    agent.init_astar_planner(grid_map, config.GRID_RESOLUTION)

    # 测试统计
    all_results = []
    trajectories = []

    print(f"\n开始测试 ({args.episodes} 回合)...")
    for episode in range(1, args.episodes + 1):
        result = test_episode(env, agent, args.render, args.record_path)
        all_results.append(result)

        if args.record_path and result['trajectory']:
            trajectories.append({
                'trajectory': result['trajectory'],
                'success': result['reached_target'],
                'collision': result['collision'],
            })

        if episode % 10 == 0:
            print(f"Episode {episode}/{args.episodes} - "
                  f"Reward: {result['reward']:.2f}, "
                  f"Steps: {result['steps']}, "
                  f"Success: {result['reached_target']}")

    # 计算统计指标
    stats = calculate_statistics(all_results)

    # 打印结果
    print_results(stats, args.stage)

    # 保存结果
    save_results(stats, trajectories, args.save_dir, args.stage)

    # 绘制结果图表
    if trajectories:
        plot_trajectories(trajectories, env, args.save_dir, args.stage)

    # 关闭环境
    env.close()

    return stats


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    计算测试统计指标

    Args:
        results: 测试结果列表

    Returns:
        统计指标字典
    """
    n = len(results)

    # 基本统计
    rewards = [r['reward'] for r in results]
    steps = [r['steps'] for r in results]
    path_lengths = [r['path_length'] for r in results if r['path_length'] > 0]
    execution_times = [r['execution_time'] for r in results]
    energy_consumed = [r['energy_consumed'] for r in results]

    # 成功/失败统计
    successes = sum(1 for r in results if r['reached_target'])
    collisions = sum(1 for r in results if r['collision'])
    timeouts = sum(1 for r in results if not r['reached_target'] and not r['collision'])

    stats = {
        # 成功率
        'success_rate': successes / n * 100,
        'collision_rate': collisions / n * 100,
        'timeout_rate': timeouts / n * 100,

        # 奖励统计
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'reward_min': np.min(rewards),
        'reward_max': np.max(rewards),

        # 步数统计
        'steps_mean': np.mean(steps),
        'steps_std': np.std(steps),
        'steps_min': np.min(steps),
        'steps_max': np.max(steps),

        # 路径长度统计
        'path_length_mean': np.mean(path_lengths) if path_lengths else 0,
        'path_length_std': np.std(path_lengths) if path_lengths else 0,

        # 执行时间统计
        'time_mean': np.mean(execution_times),
        'time_std': np.std(execution_times),

        # 能耗统计
        'energy_mean': np.mean(energy_consumed),
        'energy_std': np.std(energy_consumed),

        # 总数
        'total_episodes': n,
        'total_successes': successes,
        'total_collisions': collisions,
    }

    return stats


def print_results(stats: Dict, stage: int):
    """
    打印测试结果

    Args:
        stats: 统计指标
        stage: 测试阶段
    """
    print("\n" + "=" * 60)
    print(f"测试结果 - Stage {stage}")
    print("=" * 60)

    print(f"\n成功率统计:")
    print(f"  成功率: {stats['success_rate']:.2f}%")
    print(f"  碰撞率: {stats['collision_rate']:.2f}%")
    print(f"  超时率: {stats['timeout_rate']:.2f}%")

    print(f"\n奖励统计:")
    print(f"  平均奖励: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}")
    print(f"  最小/最大奖励: {stats['reward_min']:.2f} / {stats['reward_max']:.2f}")

    print(f"\n步数统计:")
    print(f"  平均步数: {stats['steps_mean']:.2f} ± {stats['steps_std']:.2f}")
    print(f"  最小/最大步数: {stats['steps_min']} / {stats['steps_max']}")

    print(f"\n路径长度统计:")
    print(f"  平均路径长度: {stats['path_length_mean']:.2f}m ± {stats['path_length_std']:.2f}m")

    print(f"\n效率统计:")
    print(f"  平均执行时间: {stats['time_mean']:.4f}s ± {stats['time_std']:.4f}s")
    print(f"  平均能耗: {stats['energy_mean']:.2f}J ± {stats['energy_std']:.2f}J")

    print("=" * 60)


def save_results(stats: Dict, trajectories: List[Dict],
                 save_dir: str, stage: int):
    """
    保存测试结果

    Args:
        stats: 统计指标
        trajectories: 轨迹列表
        save_dir: 保存目录
        stage: 测试阶段
    """
    import json

    # 保存统计结果
    stats_file = os.path.join(save_dir, f'stats_stage{stage}.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存到: {stats_file}")

    # 保存轨迹数据
    if trajectories:
        trajectories_file = os.path.join(save_dir, f'trajectories_stage{stage}.json')
        # 转换为可序列化格式
        serializable_trajectories = []
        for t in trajectories:
            serializable_trajectories.append({
                'trajectory': [(float(x), float(y)) for x, y in t['trajectory']],
                'success': t['success'],
                'collision': t['collision'],
            })
        with open(trajectories_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_trajectories, f)
        print(f"轨迹数据已保存到: {trajectories_file}")


def plot_trajectories(trajectories: List[Dict], env: LunarEnvironment,
                      save_dir: str, stage: int):
    """
    绘制轨迹图

    Args:
        trajectories: 轨迹列表
        env: 环境（用于获取障碍物位置）
        save_dir: 保存目录
        stage: 测试阶段
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 所有轨迹
    ax = axes[0]
    ax.set_xlim(0, config.ENV_WIDTH)
    ax.set_ylim(0, config.ENV_HEIGHT)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'All Trajectories - Stage {stage}')
    ax.grid(True, alpha=0.3)

    # 绘制障碍物
    for obs in env.obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.radius, color='gray', alpha=0.5)
        ax.add_patch(circle)

    # 绘制轨迹
    for t in trajectories:
        traj = np.array(t['trajectory'])
        if t['success']:
            ax.plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.3, linewidth=0.5)
        elif t['collision']:
            ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=0.5)
        else:
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=0.5)

    # 绘制起点和终点
    ax.plot(0.5, 0.5, 'ko', markersize=10, label='Start')
    ax.plot(9.5, 9.5, 'g*', markersize=15, label='Goal')
    ax.legend()

    # 2. 成功轨迹热力图
    ax = axes[1]

    # 创建热力图网格
    grid_size = 50
    heat_map = np.zeros((grid_size, grid_size))

    for t in trajectories:
        if t['success']:
            for x, y in t['trajectory']:
                grid_x = int(x / config.ENV_WIDTH * (grid_size - 1))
                grid_y = int(y / config.ENV_HEIGHT * (grid_size - 1))
                grid_x = np.clip(grid_x, 0, grid_size - 1)
                grid_y = np.clip(grid_y, 0, grid_size - 1)
                heat_map[grid_y, grid_x] += 1

    im = ax.imshow(heat_map, origin='lower', cmap='hot',
                   extent=[0, config.ENV_WIDTH, 0, config.ENV_HEIGHT])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Successful Trajectory Heatmap - Stage {stage}')
    plt.colorbar(im, ax=ax, label='Frequency')

    # 绘制障碍物轮廓
    for obs in env.obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.radius,
                             fill=False, color='white', linewidth=2)
        ax.add_patch(circle)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectories_stage{stage}.png'), dpi=150)
    plt.close()
    print(f"轨迹图已保存到: {save_dir}/trajectories_stage{stage}.png")


def compare_methods_test():
    """比较不同方法的测试性能"""
    methods = ['d3qn', 'd3qn_per', 'a_star_d3qn_opt']
    stages = [1, 2, 3]

    # 假设模型已训练并保存
    results = {}

    for stage in stages:
        results[stage] = {}
        for method in methods:
            model_path = f'./results/{method}_stage{stage}/model_final.pth'
            if os.path.exists(model_path):
                args = argparse.Namespace(
                    model=model_path,
                    stage=stage,
                    episodes=100,
                    render=False,
                    save_dir=f'./test_results/{method}_stage{stage}',
                    record_path=True
                )
                results[stage][method] = run_tests(args)
            else:
                print(f"模型不存在: {model_path}")

    return results


def demo():
    """演示模式：展示单次路径规划"""
    print("演示模式")

    # 创建环境
    env = LunarEnvironment(stage=2, render=True)

    # 创建智能体（使用随机策略演示）
    agent = D3QNAgent(use_per=True, use_astar=True)

    # 初始化A*规划器
    obstacles = [(obs.x, obs.y, obs.radius) for obs in env.obstacles if not obs.is_dynamic]
    grid_map = create_grid_map(config.ENV_WIDTH, config.ENV_HEIGHT,
                                config.GRID_RESOLUTION, obstacles)
    agent.init_astar_planner(grid_map, config.GRID_RESOLUTION)

    # 重置环境
    state = env.reset()

    # 规划全局路径
    start = (env.rover.x, env.rover.y)
    goal = (env.target_x, env.target_y)
    global_path = agent.plan_global_path(start, goal)

    if global_path:
        print(f"全局路径规划成功: {len(global_path)} 个路径点")

    # 运行演示
    done = False
    step = 0
    while not done and step < 200:
        additional_features = env.get_additional_features()
        action = agent.select_action(state, additional_features, training=False)
        state, reward, done, info = env.step(action)
        env.render()
        step += 1
        time.sleep(0.05)

    print(f"演示结束 - 步数: {step}, 到达目标: {info['reached_target']}")
    env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.model == 'demo':
        demo()
    else:
        run_tests(args)
