#!/usr/bin/env python3
"""
嫦娥4号/玉兔二号 数字孪生重演脚本 (Digital Twin Replay)

功能：
1. 加载第3章生成的高精度月面环境模型 (npy/tiff)。
2. 加载"真实"的巡视器路径数据 (csv)。
3. 在数字孪生环境中"重演"行走过程。
4. 计算并输出沿途的物理力学参数（沉陷、打滑），用于验证环境模型的准确性。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.modeling import EnvironmentModeling
from src.dynamics.rover_dynamics import LunarRoverDynamics
import importlib.util

# 动态导入 Visualization
spec = importlib.util.spec_from_file_location("Visualization", os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'visualization.py'))
visualization_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualization_module)
Visualization = visualization_module.Visualization
from src.dynamics.rover_model import RoverModel

# 导入路径处理模块
from src.core.path_processing import load_real_path_data

# ------------------------------------------------------------------------
# 1. 数据加载模块 (强制使用真实数据)
# ------------------------------------------------------------------------


def load_chapter3_map_strict(map_path):
    """
    严格加载第3章地图，不使用随机生成作为后备。
    """
    print(f"正在加载环境地图 (Chapter 3 Output): {map_path}")
    
    if not os.path.exists(map_path):
        print(f"\n[严重错误] 找不到地图文件: {map_path}")
        print("请确保先运行了第3章的建模脚本，并生成了 .npy 或 .tiff 文件。")
        print("本脚本拒绝使用随机生成的地图，以保证博士论文数据的真实性。")
        sys.exit(1) # 直接退出

    try:
        if map_path.endswith('.npy'):
            data = np.load(map_path, allow_pickle=True)
            # 处理保存为字典的情况
            if data.shape == () and isinstance(data.item(), dict):
                return data.item()
            # 处理保存为纯数组的情况
            else:
                return wrap_array_to_map_dict(data)
        else:
            # 这里简化处理，假设是TIFF，实际项目中需用 rasterio
            import rasterio
            with rasterio.open(map_path) as src:
                elevation = src.read(1)
                return wrap_array_to_map_dict(elevation, res=src.res[0])
    except Exception as e:
        print(f"地图加载异常: {e}")
        sys.exit(1)

def wrap_array_to_map_dict(elevation_data, res=0.1):
    """将纯高程数组封装为标准字典格式"""
    h, w = elevation_data.shape
    map_data = {
        'elevation_map': elevation_data,
        'obstacle_map': np.zeros_like(elevation_data), # 需从Ch3算法生成
        'map_resolution': res,
        'map_size': (w * res, h * res),
        # 初始化物理属性层 (Bekker参数)
        'physics_map': np.zeros((h, w, 5)) 
    }
    # 填充物理参数默认值 (应该从Ch3的语义分割结果映射而来)
    # [kc, kphi, n, c, phi]
    map_data['physics_map'][:, :, 0] = 1370.0  # kc
    map_data['physics_map'][:, :, 1] = 814000.0 # kphi
    map_data['physics_map'][:, :, 2] = 1.0     # n
    map_data['physics_map'][:, :, 3] = 170.0   # c
    map_data['physics_map'][:, :, 4] = 31.0    # phi
    return map_data

# ------------------------------------------------------------------------
# 2. 控制与仿真模块 (路径跟随)
# ------------------------------------------------------------------------

def get_control_command(current_pos, current_yaw, target_pos, k_rho=1.5, k_alpha=4.0):
    """
    简单的路径跟随控制器 (Proportional Control)
    计算让巡视器驶向 target_pos 的车轮指令
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    distance = np.sqrt(dx**2 + dy**2)
    target_angle = np.arctan2(dy, dx)
    
    # 计算航向误差 (-pi to pi)
    alpha = target_angle - current_yaw
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    
    # 如果距离很近，停止
    if distance < 0.2:
        return np.zeros(6), True # Reached
    
    # 简单的差速控制逻辑
    base_velocity = 0.5 # m/s (设定巡视器标准速度)
    
    # 转向调整
    angular_correction = k_alpha * alpha
    
    # 左侧车轮速度
    v_left = base_velocity - angular_correction * 0.3 # 0.3 is half track width approx
    # 右侧车轮速度
    v_right = base_velocity + angular_correction * 0.3
    
    # 限制最大速度
    v_left = np.clip(v_left, -1.0, 1.0)
    v_right = np.clip(v_right, -1.0, 1.0)
    
    # 生成6轮指令 [L, L, L, R, R, R]
    commands = np.array([v_left, v_left, v_left, v_right, v_right, v_right])
    
    return commands, False

def run_replay_simulation(env_model, rover_dynamics, path_points, dt=0.1):
    """
    执行路径重演仿真
    """
    print(">>> 开始执行数字孪生路径重演...")
    
    # 初始化
    start_pos = [path_points[0][0], path_points[0][1], 0.0]
    # 计算初始航向 (指向第二个点)
    init_dx = path_points[1][0] - path_points[0][0]
    init_dy = path_points[1][1] - path_points[0][1]
    start_yaw = np.arctan2(init_dy, init_dx)
    
    rover_dynamics.reset(start_pos)
    # 强制设置初始航向 (Hack: 修改内部状态，需确保 dynamics 类支持)
    try:
        # 假设 state 是字典
        if isinstance(rover_dynamics.state, dict):
            if 'orientation' in rover_dynamics.state:
                rover_dynamics.state['orientation'][2] = start_yaw
            else:
                print("警告: 无法设置初始航向，使用默认值")
        else:
            # 假设 state 是列表或数组
            rover_dynamics.state[3] = start_yaw
    except Exception as e:
        print(f"警告: 设置初始航向失败: {e}")
    
    results = {
        'time': [], 'x': [], 'y': [], 'yaw': [],
        'slip_mean': [], 'sinkage_mean': [],
        'risk_level': [] # 综合风险指标
    }
    
    current_waypoint_idx = 1
    max_time = 300 # 秒，超时保护
    current_time = 0
    
    while current_waypoint_idx < len(path_points) and current_time < max_time:
        # 获取当前状态
        state = rover_dynamics.get_state()
        
        # 处理不同类型的 state
        try:
            if isinstance(state, dict):
                # 如果 state 是字典
                curr_pos = state['position'][:2]
                curr_yaw = state['orientation'][2]
            else:
                # 如果 state 是列表或数组
                curr_pos = state[0:2]
                curr_yaw = state[3]
        except Exception as e:
            print(f"警告: 获取状态失败: {e}")
            # 使用默认值
            curr_pos = [0, 0]
            curr_yaw = 0
        
        # 获取目标点
        target = path_points[current_waypoint_idx]
        
        # 计算控制指令
        cmd, reached = get_control_command(curr_pos, curr_yaw, target)
        
        if reached:
            print(f"  [T={current_time:.1f}s] 到达路点 {current_waypoint_idx}: {target}")
            current_waypoint_idx += 1
            continue
            
        # 物理步进
        info = rover_dynamics.step(cmd, dt)
        
        # 记录数据
        results['time'].append(current_time)
        try:
            results['x'].append(info['position'][0])
            results['y'].append(info['position'][1])
            results['yaw'].append(info['orientation'][2])
        except Exception as e:
            print(f"警告: 记录状态失败: {e}")
            results['x'].append(0)
            results['y'].append(0)
            results['yaw'].append(0)
        
        # 记录关键的物理交互参数
        # 注意：这里的 slip 和 sinkage 是由 Ch4 动力学模型基于 Ch3 地图参数计算出来的
        try:
            avg_slip = np.mean(info.get('slip_ratios', [0]))
            avg_sinkage = np.mean(info.get('sinkages', [0]))
        except Exception as e:
            print(f"警告: 计算物理参数失败: {e}")
            avg_slip = 0
            avg_sinkage = 0
        
        results['slip_mean'].append(avg_slip)
        results['sinkage_mean'].append(avg_sinkage)
        
        # 简单的风险评估逻辑
        risk = 0
        if avg_slip > 0.2: risk += 1 # 打滑风险
        if avg_sinkage > 0.05: risk += 2 # 沉陷风险
        results['risk_level'].append(risk)
        
        current_time += dt
        
    print(f"仿真结束。总耗时: {current_time:.1f}s")
    return results

# ------------------------------------------------------------------------
# 3. 可视化分析模块
# ------------------------------------------------------------------------

def analyze_and_plot(results, env_model, real_path, output_dir):
    """
    生成符合博士论文规范的对比分析图
    """
    print("正在生成分析图表...")
    
    # 1. 轨迹对比图 (在 DEM 地图上)
    plt.figure(figsize=(10, 8))
    
    # 绘制高程底图
    extent = [0, env_model.map_size[0], 0, env_model.map_size[1]]
    plt.imshow(env_model.elevation_map, extent=extent, cmap='gray', origin='lower', alpha=0.8)
    plt.colorbar(label='高程 (m)')
    
    # 绘制真实路径 (参考)
    plt.plot(real_path[:, 0], real_path[:, 1], 'g--', linewidth=2, label='预设/真实路径 (Reference)')
    
    # 绘制仿真轨迹 (孪生推演)
    plt.plot(results['x'], results['y'], 'r-', linewidth=1.5, label='数字孪生推演轨迹 (Digital Twin)')
    
    # 标注高风险点
    risks = np.array(results['risk_level'])
    high_risk_idx = np.where(risks > 0)[0]
    if len(high_risk_idx) > 0:
        plt.scatter(np.array(results['x'])[high_risk_idx], 
                    np.array(results['y'])[high_risk_idx], 
                    c='orange', s=10, marker='x', label='高风险区域 (打滑/沉陷)')

    plt.title('图 6-x: 基于数字孪生的路径跟踪与通过性验证', fontsize=14)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend(loc='upper right')
    plt.grid(linestyle=':', alpha=0.5)
    
    save_path = os.path.join(output_dir, 'Ch6_Trajectory_Validation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存轨迹图: {save_path}")
    
    # 2. 物理参数演化曲线
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('仿真时间 (s)')
    ax1.set_ylabel('平均滑转率 (Slip Ratio)', color='tab:red')
    l1, = ax1.plot(results['time'], results['slip_mean'], color='tab:red', label='滑转率')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0, 1.0) # 滑转率通常在 0-1 之间
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('平均沉陷量 (Sinkage) [m]', color='tab:blue')
    l2, = ax2.plot(results['time'], results['sinkage_mean'], color='tab:blue', linestyle='--', label='沉陷量')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('图 6-y: 行走过程中的轮-土相互作用参数演化', fontsize=14)
    plt.legend([l1, l2], ['滑转率', '沉陷量'], loc='upper left')
    plt.grid(True)
    
    save_path2 = os.path.join(output_dir, 'Ch6_Dynamics_Evolution.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"保存动力学曲线: {save_path2}")


def main():
    print("=== 面向月面巡视器自主行走的数字孪生仿真验证 (Chapter 6) ===")
    
    # 1. 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs', 'paper_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 关键：指定具体文件路径
    # 地图来自第3章输出
    map_file = os.path.join(base_dir, 'outputs', 'visualizations', 'tiff', 'nac_dtm_elevation_matrix.npy')
    # 路径来自外部输入 (你需要准备这个CSV，或者让代码自动生成演示数据)
    path_file = os.path.join(base_dir, 'data', 'trajectories', 'yutu2_real_path_segment.csv')
    
    # 2. 加载数据
    map_data = load_chapter3_map_strict(map_file)
    real_path = load_real_path_data(path_file)
    
    # 3. 初始化模型
    env_model = EnvironmentModeling(
        map_resolution=map_data['map_resolution'], 
        map_size=map_data['map_size']
    )
    # 注入数据
    env_model.elevation_map = map_data['elevation_map']
    env_model.physics_map = map_data['physics_map'] # 包含 Bekker 参数
    
    rover_dynamics = LunarRoverDynamics(
        rover_params={'mass': 140, 'wheel_radius': 0.15, 'track_width': 0.6}, # 玉兔二号参数
        env_model=env_model
    )
    
    # 4. 运行仿真 (闭环控制)
    results = run_replay_simulation(env_model, rover_dynamics, real_path)
    
    # 5. 生成论文图表
    analyze_and_plot(results, env_model, real_path, output_dir)
    
    print("=== 仿真验证完成 ===")

if __name__ == "__main__":
    main()
