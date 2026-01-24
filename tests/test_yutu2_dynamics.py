import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.dynamics.yutu2_rover_dynamics import Yutu2RoverDynamics

def test_forward_motion():
    """
    测试前进运动
    """
    print("=== 测试前进运动 ===")
    
    rover = Yutu2RoverDynamics()
    
    # 前进命令
    wheel_commands = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    
    positions = []
    velocities = []
    energies = []
    
    for step in range(20):
        state = rover.step(wheel_commands, dt=0.1)
        positions.append(state['position'].copy())
        velocities.append(np.linalg.norm(state['velocity']))
        energies.append(state['energy_consumed'])
        
        print(f"Step {step+1}: Pos={state['position'][:2]}, Vel={np.linalg.norm(state['velocity']):.2f} m/s, Energy={state['energy_consumed']:.2f} J")
    
    # 验证结果
    final_position = np.array(positions[-1])
    final_velocity = velocities[-1]
    final_energy = energies[-1]
    
    print(f"\n测试结果:")
    print(f"最终位置: {final_position[:2]}")
    print(f"最终速度: {final_velocity:.2f} m/s")
    print(f"总能量消耗: {final_energy:.2f} J")
    
    # 验证位置是否变化
    initial_position = np.array(positions[0])
    position_change = np.linalg.norm(final_position[:2] - initial_position[:2])
    print(f"位置变化: {position_change:.4f} m")
    
    # 验证速度是否被正确限制
    max_velocity = 0.2  # 玉兔2号最大速度
    if abs(final_velocity - max_velocity) < 0.01:
        print("速度限制: 通过")
    else:
        print("速度限制: 失败")
    
    return positions, velocities, energies

def test_turning_motion():
    """
    测试转向运动
    """
    print("\n=== 测试转向运动 ===")
    
    rover = Yutu2RoverDynamics()
    
    # 左转向命令（左轮速度慢，右轮速度快）
    wheel_commands = np.array([5.0, 10.0, 15.0, 5.0, 10.0, 15.0])
    
    positions = []
    orientations = []
    
    for step in range(20):
        state = rover.step(wheel_commands, dt=0.1)
        positions.append(state['position'].copy())
        orientations.append(state['orientation'].copy())
        
        print(f"Step {step+1}: Pos={state['position'][:2]}, Orient={state['orientation']}")
    
    # 验证结果
    initial_position = np.array(positions[0])
    final_position = np.array(positions[-1])
    position_change = np.linalg.norm(final_position[:2] - initial_position[:2])
    
    print(f"\n测试结果:")
    print(f"位置变化: {position_change:.4f} m")
    print(f"转向运动: {'通过' if position_change > 0 else '失败'}")
    
    return positions, orientations

def test_energy_consumption():
    """
    测试能量消耗
    """
    print("\n=== 测试能量消耗 ===")
    
    rover = Yutu2RoverDynamics()
    
    # 不同速度的能量消耗
    speeds = [0.05, 0.1, 0.15, 0.2]
    energy_consumptions = []
    
    for speed in speeds:
        rover.reset()
        wheel_commands = np.array([speed * 100, speed * 100, speed * 100, speed * 100, speed * 100, speed * 100])
        
        for step in range(10):
            state = rover.step(wheel_commands, dt=0.1)
        
        energy_consumptions.append(rover.energy_consumed)
        print(f"速度 {speed:.2f} m/s: 能量消耗 {rover.energy_consumed:.2f} J")
    
    # 验证能量消耗是否随速度增加
    if energy_consumptions[-1] > energy_consumptions[0]:
        print("能量消耗: 通过")
    else:
        print("能量消耗: 失败")
    
    return speeds, energy_consumptions

def test_terrain_interaction():
    """
    测试地形交互
    """
    print("\n=== 测试地形交互 ===")
    
    rover = Yutu2RoverDynamics()
    
    # 前进命令
    wheel_commands = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    
    contact_states = []
    sinkages = []
    
    for step in range(10):
        state = rover.step(wheel_commands, dt=0.1)
        contact_states.append(rover.contact_states['wheel_terrain_contact'].copy())
        sinkages.append(np.mean(rover.contact_states['sinkage']))
        
        print(f"Step {step+1}: 接触状态={rover.contact_states['wheel_terrain_contact']}, 平均下陷={np.mean(rover.contact_states['sinkage']):.4f} m")
    
    # 验证接触状态
    all_contact = all(np.all(cs) for cs in contact_states)
    print(f"接触状态: {'通过' if all_contact else '失败'}")
    
    return contact_states, sinkages

def test_stability():
    """
    测试稳定性
    """
    print("\n=== 测试稳定性 ===")
    
    rover = Yutu2RoverDynamics()
    
    # 前进命令
    wheel_commands = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    
    stabilities = []
    
    for step in range(10):
        state = rover.step(wheel_commands, dt=0.1)
        nav_features = rover.get_navigation_features()
        stabilities.append(nav_features['stability_index'])
        
        print(f"Step {step+1}: 稳定性={nav_features['stability_index']:.2f}")
    
    # 验证稳定性
    avg_stability = np.mean(stabilities)
    print(f"平均稳定性: {avg_stability:.2f}")
    
    if avg_stability > 0.8:
        print("稳定性: 通过")
    else:
        print("稳定性: 失败")
    
    return stabilities

def visualize_results(positions, velocities, energies):
    """
    可视化测试结果
    """
    print("\n=== 生成可视化结果 ===")
    
    plt.figure(figsize=(15, 10))
    
    # 轨迹图
    plt.subplot(311)
    positions = np.array(positions)
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(positions[0, 0], positions[0, 1], 'bo', markersize=8, label='Start')
    plt.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=8, label='End')
    plt.title('Forward Motion Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    # 速度曲线
    plt.subplot(312)
    times = np.arange(len(velocities)) * 0.1
    plt.plot(times, velocities, 'r-', linewidth=2, label='Velocity')
    plt.axhline(y=0.2, color='g', linestyle='--', label='Max Speed')
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    
    # 能量曲线
    plt.subplot(313)
    plt.plot(times, energies, 'g-', linewidth=2, label='Energy')
    plt.title('Energy Consumption vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('yutu2_dynamics_test_results.png', dpi=300, bbox_inches='tight')
    print("测试结果可视化已保存: yutu2_dynamics_test_results.png")

def main():
    """
    主测试函数
    """
    print("=== 玉兔2号月球车动力学模型全面测试 ===")
    
    # 测试前进运动
    positions, velocities, energies = test_forward_motion()
    
    # 测试转向运动
    turning_positions, turning_orientations = test_turning_motion()
    
    # 测试能量消耗
    speeds, energy_consumptions = test_energy_consumption()
    
    # 测试地形交互
    contact_states, sinkages = test_terrain_interaction()
    
    # 测试稳定性
    stabilities = test_stability()
    
    # 生成可视化结果
    visualize_results(positions, velocities, energies)
    
    print("\n=== 测试完成 ===")
    print("所有测试结果已生成，检查 yutu2_dynamics_test_results.png 查看可视化结果")

if __name__ == "__main__":
    main()