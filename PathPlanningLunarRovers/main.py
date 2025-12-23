# -*- coding: utf-8 -*-
"""
A*-D3QN-Opt: 月球车路径规划系统主程序入口
Path Planning for Lunar Rovers in Dynamic Environments:
An Autonomous Navigation Framework Enhanced by Digital Twin-Based A*-D3QN

论文复现代码主入口
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='A*-D3QN-Opt: 月球车路径规划系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  训练模型:
    python main.py train --method a_star_d3qn_opt --stage 1 --episodes 1000

  测试模型:
    python main.py test --model ./models/model.pth --stage 2

  演示模式:
    python main.py demo --stage 2

  比较方法:
    python main.py compare
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--method', type=str, default='a_star_d3qn_opt',
                              choices=['d3qn', 'd3qn_per', 'a_star_d3qn_opt'],
                              help='训练方法')
    train_parser.add_argument('--stage', type=int, default=1,
                              choices=[1, 2, 3],
                              help='环境复杂度阶段')
    train_parser.add_argument('--episodes', type=int, default=1000,
                              help='训练回合数')
    train_parser.add_argument('--render', action='store_true',
                              help='是否渲染环境')
    train_parser.add_argument('--seed', type=int, default=42,
                              help='随机种子')
    train_parser.add_argument('--save_dir', type=str, default='./results',
                              help='结果保存目录')
    train_parser.add_argument('--load_model', type=str, default=None,
                              help='加载预训练模型路径')

    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('--model', type=str, required=True,
                             help='模型文件路径')
    test_parser.add_argument('--stage', type=int, default=1,
                             choices=[1, 2, 3],
                             help='测试环境阶段')
    test_parser.add_argument('--episodes', type=int, default=100,
                             help='测试回合数')
    test_parser.add_argument('--render', action='store_true',
                             help='是否渲染')
    test_parser.add_argument('--save_dir', type=str, default='./test_results',
                             help='结果保存目录')
    test_parser.add_argument('--record_path', action='store_true',
                             help='是否记录路径轨迹')

    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='演示模式')
    demo_parser.add_argument('--stage', type=int, default=2,
                             choices=[1, 2, 3],
                             help='演示环境阶段')
    demo_parser.add_argument('--model', type=str, default=None,
                             help='可选：加载训练好的模型')

    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较不同方法')
    compare_parser.add_argument('--episodes', type=int, default=100,
                                help='每种方法的训练回合数')

    args = parser.parse_args()

    if args.command == 'train':
        from train import train
        train(args)

    elif args.command == 'test':
        from test import run_tests
        run_tests(args)

    elif args.command == 'demo':
        run_demo(args)

    elif args.command == 'compare':
        run_comparison(args)

    else:
        parser.print_help()


def run_demo(args):
    """
    运行演示模式

    Args:
        args: 命令行参数
    """
    import time
    import numpy as np
    from config import config
    from lunar_environment import LunarEnvironment
    from d3qn_agent import D3QNAgent
    from astar import create_grid_map

    print("=" * 60)
    print("A*-D3QN-Opt 月球车路径规划演示")
    print("=" * 60)

    # 创建环境
    env = LunarEnvironment(stage=args.stage, render=True)
    print(f"\n环境创建成功 - Stage {args.stage}")
    print(f"  - 无障碍物" if args.stage == 1 else
          f"  - 静态障碍物" if args.stage == 2 else
          f"  - 动态障碍物")

    # 创建智能体
    agent = D3QNAgent(use_per=True, use_astar=True)

    # 加载模型（如果提供）
    if args.model:
        try:
            agent.load_model(args.model)
            print(f"模型加载成功: {args.model}")
            agent.epsilon = 0.0
        except:
            print(f"警告: 无法加载模型 {args.model}，使用随机策略")

    # 初始化A*规划器
    obstacles = [(obs.x, obs.y, obs.radius) for obs in env.obstacles if not obs.is_dynamic]
    grid_map = create_grid_map(config.ENV_WIDTH, config.ENV_HEIGHT,
                                config.GRID_RESOLUTION, obstacles)
    agent.init_astar_planner(grid_map, config.GRID_RESOLUTION)

    # 运行演示
    num_demos = 5
    for demo_idx in range(num_demos):
        print(f"\n--- 演示 {demo_idx + 1}/{num_demos} ---")

        # 重置环境
        state = env.reset()
        additional_features = env.get_additional_features()

        # 规划全局路径
        start = (env.rover.x, env.rover.y)
        goal = (env.target_x, env.target_y)
        global_path = agent.plan_global_path(start, goal)

        if global_path:
            print(f"全局路径规划成功: {len(global_path)} 个路径点")

        # 执行路径
        done = False
        step = 0
        total_reward = 0

        while not done and step < config.MAX_STEPS_PER_EPISODE:
            # 选择动作
            action = agent.select_action(state, additional_features, training=False)

            # 执行动作
            state, reward, done, info = env.step(action)
            additional_features = env.get_additional_features()

            total_reward += reward
            step += 1

            # 渲染
            env.render()
            time.sleep(0.05)

        # 打印结果
        status = "到达目标!" if info['reached_target'] else \
                 "发生碰撞!" if info['collision'] else "超时"
        print(f"结果: {status}")
        print(f"  步数: {step}")
        print(f"  累计奖励: {total_reward:.2f}")
        print(f"  能量消耗: {info['energy_consumed']:.2f}J")

        time.sleep(1)

    env.close()
    print("\n演示结束!")


def run_comparison(args):
    """
    运行方法比较实验

    Args:
        args: 命令行参数
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from train import train
    import argparse

    print("=" * 60)
    print("A*-D3QN-Opt 方法比较实验")
    print("=" * 60)

    methods = ['d3qn', 'd3qn_per', 'a_star_d3qn_opt']
    method_names = ['D3QN', 'D3QN-PER', 'A*-D3QN-Opt (Ours)']
    stages = [1, 2, 3]
    stage_names = ['低复杂度', '中复杂度', '高复杂度']

    results = {}

    # 运行每种方法在每个阶段的训练
    for stage in stages:
        results[stage] = {}
        for method, method_name in zip(methods, method_names):
            print(f"\n{'=' * 40}")
            print(f"训练: {method_name} @ Stage {stage}")
            print('=' * 40)

            train_args = argparse.Namespace(
                method=method,
                stage=stage,
                episodes=args.episodes,
                render=False,
                seed=42,
                save_dir=f'./comparison/{method}_stage{stage}',
                load_model=None
            )

            try:
                result = train(train_args)
                results[stage][method] = {
                    'rewards': result['rewards'],
                    'collisions': result['collisions'],
                    'successes': result['successes'],
                    'time': result['total_time'],
                }
            except Exception as e:
                print(f"训练失败: {e}")
                results[stage][method] = None

    # 绘制比较结果
    plot_comparison_results(results, methods, method_names, stages, stage_names)

    print("\n比较实验完成!")


def plot_comparison_results(results, methods, method_names, stages, stage_names):
    """绘制比较结果图表"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 绘制每个阶段的学习曲线
    for i, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        ax = axes[0, i]

        for j, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            if results[stage].get(method) is not None:
                rewards = results[stage][method]['rewards']
                # 移动平均
                window = min(50, len(rewards) // 5)
                if len(rewards) >= window:
                    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                    ax.plot(range(window - 1, len(rewards)), moving_avg,
                            color=color, linewidth=2, label=method_name)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title(f'{stage_name} (Stage {stage})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 绘制性能指标对比柱状图
    metrics = ['成功率', '碰撞率', '平均奖励']
    x = np.arange(len(stages))
    width = 0.25

    for k, (metric_name, metric_key) in enumerate([
        ('成功率', 'success_rate'),
        ('碰撞率', 'collision_rate'),
        ('平均奖励', 'avg_reward')
    ]):
        ax = axes[1, k]

        for j, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            values = []
            for stage in stages:
                if results[stage].get(method) is not None:
                    if metric_key == 'success_rate':
                        r = results[stage][method]
                        val = r['successes'] / len(r['rewards']) * 100
                    elif metric_key == 'collision_rate':
                        r = results[stage][method]
                        val = r['collisions'] / len(r['rewards']) * 100
                    else:
                        val = np.mean(results[stage][method]['rewards'])
                    values.append(val)
                else:
                    values.append(0)

            ax.bar(x + j * width, values, width, label=method_name, color=color)

        ax.set_xlabel('Stage')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'Stage {s}' for s in stages])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./comparison/comparison_results.png', dpi=150)
    plt.close()
    print("\n比较结果图表已保存到: ./comparison/comparison_results.png")


if __name__ == "__main__":
    main()
