#!/usr/bin/env python3
"""
运行缓动轨迹仿真脚本
使用缓动轨迹生成器生成平滑的垂直直线轨迹，并运行月球车仿真
"""

import sys
import os
import argparse
import urllib.request

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.dynamics.rover_dynamics import LunarRoverDynamics
from src.environment.modeling import EnvironmentModeling
from src.core.planning.trajectory_generator import TrajectoryGenerator
from src.core.experiment import load_config


# 创建轨迹生成器实例
generator = TrajectoryGenerator()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _resolve_path(path, base_dir):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _download_if_needed(remote_url, local_path):
    if os.path.isfile(local_path) or not remote_url:
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"下载 DEM: {remote_url}")
    urllib.request.urlretrieve(remote_url, local_path)
    print(f"已下载到: {local_path}")


def _select_dem_source(config):
    dem_selection = config.get("dem_selection")
    dem_sources = config.get("dem_sources", {})
    dem_cfg = config.get("dem", {})
    allow_download = bool(config.get("dem_allow_download", True))

    selected = dem_sources.get(dem_selection) if dem_selection else dem_cfg.get("source")
    if isinstance(selected, dict):
        local_path = selected.get("local")
        remote_url = selected.get("remote")
    else:
        local_path = selected
        remote_url = None

    local_path = _resolve_path(local_path, PROJECT_ROOT)
    if local_path and allow_download and remote_url and not os.path.isfile(local_path):
        _download_if_needed(remote_url, local_path)

    return local_path


def _align_environment_to_path(env_model, path_data, align_cfg):
    if not align_cfg:
        return
    mode = align_cfg.get("mode", "none")
    if mode in ("none", None):
        return

    margin = float(align_cfg.get("margin", 0.0))
    target_resolution = align_cfg.get("target_resolution")

    path_xy = np.array(path_data)[:, :2]
    min_x, min_y = np.min(path_xy, axis=0)
    max_x, max_y = np.max(path_xy, axis=0)
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    env_model.crop_to_bounds(min_x, min_y, max_x, max_y, target_resolution=target_resolution)


def load_path_data(config):
    """
    加载路径数据，使用缓动轨迹生成器生成标准测试轨迹
    """
    print("=== 生成缓动测试轨迹 ===")
    
    # 配置参数
    sim_cfg = config.get("simulation", {})
    traj_cfg = config.get("trajectory", {})
    ctrl_cfg = config.get("control", {})

    START = tuple(sim_cfg.get("start_pos", [100, 100]))
    END = tuple(sim_cfg.get("end_pos", [100, 500]))
    DURATION = float(sim_cfg.get("duration", 100.0))  # 100秒走完，模拟月球车真实速度
    FPS = int(sim_cfg.get("fps", 50))          # 50Hz采样
    MAX_VELOCITY = float(ctrl_cfg.get("max_velocity", 0.5))  # 月球车最大速度
    include_yaw = bool(traj_cfg.get("include_yaw", True))
    
    # 生成轨迹（使用新的轨迹生成器）
    smooth_path = generator.generate_smooth_straight_line(
        START,
        END,
        DURATION,
        FPS,
        max_velocity=MAX_VELOCITY,
        include_yaw=include_yaw
    )
    
    print(f"生成的轨迹点数: {len(smooth_path)}")
    print(f"起点: {smooth_path[0]}")
    print(f"终点: {smooth_path[-1]}")
    print(f"中间点: {smooth_path[len(smooth_path)//2]}")
    
    # 计算轨迹长度
    path_length = np.sqrt((END[0] - START[0])**2 + (END[1] - START[1])**2)
    print(f"轨迹长度: {path_length:.2f} 米")
    print(f"平均速度: {path_length / DURATION:.2f} 米/秒")
    
    return smooth_path, DURATION, FPS


def run_simulation(config, path_data, duration, fps):
    """
    运行月球车仿真
    """
    print("\n=== 运行月球车仿真 ===")
    
    # 初始化环境模型（支持 environment_artifact / 多源DEM）
    env_model = EnvironmentModeling()
    inputs_cfg = config.get("inputs", {})
    env_artifact = _resolve_path(inputs_cfg.get("environment_artifact"), PROJECT_ROOT)
    if env_artifact and os.path.isfile(env_artifact):
        env_model.load_map(env_artifact)
        print(f"使用 environment_artifact: {env_artifact}")
    else:
        dem_cfg = config.get("dem", {})
        dem_path = _select_dem_source(config)
        if dem_path and os.path.isfile(dem_path):
            map_resolution = float(dem_cfg.get("source_resolution", 1.0))
            normalize = bool(dem_cfg.get("normalize", False))
            env_model.load_elevation_from_tiff(dem_path, map_resolution=map_resolution, normalize=normalize)
            print(f"使用 DEM: {dem_path}")
        else:
            print("未找到DEM或environment_artifact，使用默认高程图")

    # 对齐 DEM 与轨迹坐标系
    _align_environment_to_path(env_model, path_data, config.get("dem", {}).get("align"))

    print("环境模型初始化完成")
    
    # 初始化月球车动力学模型
    rover = LunarRoverDynamics(env_model=env_model)
    print("动力学模型初始化完成")
    
    # 仿真参数
    dt = 1.0 / fps  # 时间步长
    steps = len(path_data)
    
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    # 存储数据
    simulation_data = {
        'time': [],
        'position': [],
        'velocity': [],
        'energy': [],
        'slip_ratio': [],
        'sinkage': [],
        'target_position': [],
        'target_position_3d': [],
        'target_yaw': []
    }
    
    # 重置状态
    initial_x = float(path_data[0][0])
    initial_y = float(path_data[0][1])
    initial_z = env_model.get_elevation(initial_x, initial_y) if env_model else 0.0
    initial_position = [initial_x, initial_y, initial_z]
    rover.reset(initial_position)
    print(f"初始位置: {initial_position}")
    
    print("开始仿真...")
    
    # 运行仿真
    for step in range(steps):
        t = step * dt
        
        # 获取目标位置（可包含 yaw）
        target_pos = path_data[step]
        target_x = float(target_pos[0])
        target_y = float(target_pos[1])
        target_yaw = float(target_pos[2]) if len(target_pos) > 2 else None

        # 通过地形高程更新目标 Z
        target_z = env_model.get_elevation(target_x, target_y) if env_model else 0.0
        target_position_3d = [target_x, target_y, target_z]
        
        # 获取当前状态
        current_state = rover.get_state()
        current_pos = current_state['position'][:2]
        current_yaw = current_state['orientation'][2]
        
        # 计算控制命令
        # 简单的比例控制
        error = np.array([target_x, target_y]) - np.array(current_pos)
        error_norm = np.linalg.norm(error)
        
        # 计算期望朝向
        if target_yaw is None:
            desired_yaw = np.arctan2(error[1], error[0]) if error_norm > 1e-6 else current_yaw
        else:
            desired_yaw = target_yaw
        yaw_error = _wrap_angle(desired_yaw - current_yaw)

        # 简单比例控制（线速度 + 角速度）
        ctrl_cfg = config.get("control", {})
        max_velocity = float(ctrl_cfg.get("max_velocity", 0.5))
        kp_pos = float(ctrl_cfg.get("kp_pos", 0.5))
        kp_yaw = float(ctrl_cfg.get("kp_yaw", 1.0))
        linear_cmd = min(max_velocity, kp_pos * error_norm)
        linear_cmd *= max(0.0, np.cos(yaw_error))
        angular_cmd = kp_yaw * yaw_error

        # 差速驱动：由 (v, w) 转换为左右轮角速度
        track_width = rover.params['track_width']
        wheel_radius = rover.params['wheel_radius']
        v_left = linear_cmd - angular_cmd * track_width / 2.0
        v_right = linear_cmd + angular_cmd * track_width / 2.0
        omega_left = v_left / wheel_radius
        omega_right = v_right / wheel_radius
        wheel_commands = np.array([omega_left, omega_right, omega_left, omega_right, omega_left, omega_right])
        
        # 更新状态
        state = rover.step(wheel_commands, dt)
        
        # 存储数据
        simulation_data['time'].append(t)
        simulation_data['position'].append(state['position'][:2])
        simulation_data['velocity'].append(np.linalg.norm(state['velocity'][:2]))
        simulation_data['energy'].append(state['energy_consumed'])
        simulation_data['slip_ratio'].append(np.mean(state['slip_ratios']))
        simulation_data['sinkage'].append(np.mean(state['sinkages']))
        simulation_data['target_position'].append([target_x, target_y])
        simulation_data['target_position_3d'].append(target_position_3d)
        simulation_data['target_yaw'].append(desired_yaw)
        
        # 打印进度
        if step % 500 == 0:
            print(f"进度: {step}/{steps} ({t:.1f}s)")
            print(f"  当前位置: {state['position'][:2]}")
            print(f"  目标位置: {[target_x, target_y]}")
            print(f"  速度: {np.linalg.norm(state['velocity'][:2]):.2f} m/s")
    
    print("仿真完成！")
    return simulation_data


def save_results(simulation_data, path_data):
    """
    保存仿真结果
    """
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存仿真数据
    simulation_file = os.path.join(results_dir, 'easing_simulation_results.npz')
    np.savez(
        simulation_file,
        time=np.array(simulation_data['time']),
        position=np.array(simulation_data['position']),
        velocity=np.array(simulation_data['velocity']),
        energy=np.array(simulation_data['energy']),
        slip_ratio=np.array(simulation_data['slip_ratio']),
        sinkage=np.array(simulation_data['sinkage']),
        target_position=np.array(simulation_data['target_position']),
        target_position_3d=np.array(simulation_data['target_position_3d']),
        target_yaw=np.array(simulation_data['target_yaw']),
        reference_path=np.array(path_data)
    )
    print(f"仿真结果已保存到: {simulation_file}")


def visualize_results(simulation_data, path_data):
    """
    可视化仿真结果
    """
    import matplotlib.pyplot as plt
    
    # 创建可视化目录
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 1. 轨迹对比
    plt.figure(figsize=(12, 8))
    
    # 绘制参考轨迹
    reference_path = np.array(path_data)
    plt.plot(reference_path[:, 0], reference_path[:, 1], 'r--', linewidth=2, label='参考轨迹')
    
    # 绘制实际轨迹
    actual_position = np.array(simulation_data['position'])
    plt.plot(actual_position[:, 0], actual_position[:, 1], 'b-', linewidth=2, label='实际轨迹')
    
    # 绘制起点和终点
    plt.scatter(reference_path[0, 0], reference_path[0, 1], color='green', marker='o', s=100, label='起点')
    plt.scatter(reference_path[-1, 0], reference_path[-1, 1], color='red', marker='*', s=100, label='终点')
    
    plt.xlabel('X 位置 (m)')
    plt.ylabel('Y 位置 (m)')
    plt.title('月球车轨迹对比 (缓动测试)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    trajectory_file = os.path.join(viz_dir, 'easing_trajectory_comparison.png')
    plt.savefig(trajectory_file, dpi=300)
    plt.close()
    print(f"轨迹对比图已保存到: {trajectory_file}")
    
    # 2. 速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_data['time'], simulation_data['velocity'], 'r-', linewidth=2, label='实际速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.title('月球车速度曲线 (缓动测试)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    velocity_file = os.path.join(viz_dir, 'easing_velocity_profile.png')
    plt.savefig(velocity_file, dpi=300)
    plt.close()
    print(f"速度曲线图已保存到: {velocity_file}")
    
    # 3. 位置误差
    plt.figure(figsize=(12, 6))
    target_positions = np.array(simulation_data['target_position'])
    actual_positions = np.array(simulation_data['position'])
    position_error = np.linalg.norm(target_positions - actual_positions, axis=1)
    
    plt.plot(simulation_data['time'], position_error, 'b-', linewidth=2, label='位置误差')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置误差 (m)')
    plt.title('月球车位置误差 (缓动测试)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    error_file = os.path.join(viz_dir, 'easing_position_error.png')
    plt.savefig(error_file, dpi=300)
    plt.close()
    print(f"位置误差图已保存到: {error_file}")


def main():
    """
    主函数
    """
    try:
        parser = argparse.ArgumentParser(description="Run easing trajectory simulation.")
        parser.add_argument(
            "--config",
            default=os.path.join("configs", "easing_simulation.yaml"),
            help="Path to YAML/JSON config file.",
        )
        args = parser.parse_args()

        config_path = args.config
        if not os.path.isabs(config_path):
            candidate = os.path.join(PROJECT_ROOT, config_path)
            if os.path.isfile(candidate):
                config_path = candidate

        config = load_config(config_path)

        # 加载路径数据
        path_data, duration, fps = load_path_data(config)
        
        # 运行仿真
        simulation_data = run_simulation(config, path_data, duration, fps)
        
        # 保存结果
        save_results(simulation_data, path_data)
        
        # 可视化结果
        visualize_results(simulation_data, path_data)

        print("\n=== 仿真完成 ===")
        print("所有任务已完成！")
        print("提示：实际轨迹相对参考轨迹的滞后属于系统惯性导致的跟踪误差，可作为论文中的系统延迟说明。")
        
    except Exception as e:
        print(f"仿真过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
