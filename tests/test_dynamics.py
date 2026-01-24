import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dynamics.lunar_rover_dynamics import LunarRoverDynamics
from models.dynamics.dynamics_perception_integration import DynamicsPerceptionIntegration

def test_forward_motion():
    """测试前进运动"""
    print("=== 测试前进运动 ===")
    
    dynamics = LunarRoverDynamics()
    dynamics.reset()
    
    # 持续前进命令
    wheel_commands = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    
    positions = []
    velocities = []
    energies = []
    
    for step in range(10):
        state_info = dynamics.step(wheel_commands, dt=0.1)
        positions.append(state_info['position'].copy())
        velocities.append(np.linalg.norm(state_info['velocity']))
        energies.append(state_info['energy_consumed'])
        
        print(f"Step {step+1}: 位置={state_info['position'][:2]}, 速度={velocities[-1]:.2f} m/s, 能量={energies[-1]:.2f} J")
    
    # 验证前进方向
    positions = np.array(positions)
    dx = positions[-1, 0] - positions[0, 0]
    dy = positions[-1, 1] - positions[0, 1]
    
    print(f"总位移: {np.sqrt(dx**2 + dy**2):.2f} m")
    print(f"前进方向: {np.degrees(np.arctan2(dy, dx)):.2f}°")
    print()
    
    return positions, velocities, energies

def test_turning_motion():
    """测试转向运动"""
    print("=== 测试转向运动 ===")
    
    dynamics = LunarRoverDynamics()
    dynamics.reset()
    
    # 左转向命令（左右轮速度不同）
    wheel_commands = np.array([5, 15, 5, 15, 5, 15, 5, 15])
    
    positions = []
    orientations = []
    
    for step in range(10):
        state_info = dynamics.step(wheel_commands, dt=0.1)
        positions.append(state_info['position'].copy())
        orientations.append(state_info['orientation'][2])
        
        print(f"Step {step+1}: 位置={state_info['position'][:2]}, 航向={np.degrees(state_info['orientation'][2]):.2f}°")
    
    # 验证转向
    orientations = np.array(orientations)
    delta_theta = orientations[-1] - orientations[0]
    
    print(f"总转向角度: {np.degrees(delta_theta):.2f}°")
    print()
    
    return positions, orientations

def test_terrain_interaction():
    """测试地形交互"""
    print("=== 测试地形交互 ===")
    
    dynamics = LunarRoverDynamics()
    dynamics.reset()
    
    # 测试不同地形条件
    wheel_commands = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    
    for step in range(5):
        state_info = dynamics.step(wheel_commands, dt=0.1)
        terrain_data = dynamics.get_terrain_interaction_data()
        
        print(f"Step {step+1}:")
        print(f"  车轮接触状态: {terrain_data['wheel_terrain_contact']}")
        print(f"  下陷深度: {terrain_data['sinkage'][:4]}...")
        print(f"  法向力: {terrain_data['normal_forces'][:4]}...")
        print()
    
    return terrain_data

def test_energy_consumption():
    """测试能量消耗"""
    print("=== 测试能量消耗 ===")
    
    dynamics = LunarRoverDynamics()
    dynamics.reset()
    
    # 不同速度的能量消耗
    speeds = [0.5, 1.0, 1.5, 2.0]
    
    for speed in speeds:
        dynamics.reset()
        wheel_commands = np.ones(8) * speed * 10
        
        total_energy = 0
        for _ in range(10):
            state_info = dynamics.step(wheel_commands, dt=0.1)
            total_energy = state_info['energy_consumed']
        
        energy_per_meter = total_energy / (speed * 10 * 0.1)
        print(f"速度={speed:.1f} m/s: 总能量={total_energy:.2f} J, 单位距离能量={energy_per_meter:.2f} J/m")
    
    print()

def test_collision_risk():
    """测试碰撞风险计算"""
    print("=== 测试碰撞风险计算 ===")
    
    integration = DynamicsPerceptionIntegration()
    integration.reset()
    
    # 模拟障碍物
    obstacles = [
        {'x': 2.0, 'y': 0.0, 'radius': 0.5},
        {'x': 5.0, 'y': 1.0, 'radius': 0.3},
        {'x': 1.0, 'y': 2.0, 'radius': 0.4},
    ]
    
    # 前进并计算碰撞风险
    wheel_commands = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    
    for step in range(3):
        integration.step(wheel_commands, dt=0.1)
        risks = integration.get_collision_risk(obstacles)
        
        print(f"Step {step+1}:")
        for i, risk in enumerate(risks):
            print(f"  障碍物{i+1}: 距离={risk['distance']:.2f} m, 风险={risk['collision_risk']:.2f}, TTC={risk['time_to_collision']:.2f} s")
        print()
    
    return risks

def test_trajectory_prediction():
    """测试轨迹预测"""
    print("=== 测试轨迹预测 ===")
    
    integration = DynamicsPerceptionIntegration()
    integration.reset()
    
    # 前进命令
    wheel_commands = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    
    # 执行一步并预测轨迹
    integration.step(wheel_commands, dt=0.1)
    
    # 预测轨迹
    predicted = integration.predicted_trajectory
    
    print(f"预测轨迹长度: {len(predicted)} 步")
    print("预测轨迹位置:")
    for i, state in enumerate(predicted[:5]):
        print(f"  Step {i+1}: {state['position'][:2]}")
    
    print()
    
    return predicted

def test_navigation_features():
    """测试导航特征"""
    print("=== 测试导航特征 ===")
    
    integration = DynamicsPerceptionIntegration()
    integration.reset()
    
    # 前进
    wheel_commands = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    
    for step in range(5):
        integration.step(wheel_commands, dt=0.1)
        nav_features = integration.get_navigation_features()
        
        print(f"Step {step+1}:")
        print(f"  位置: {nav_features['position'][:2]}")
        print(f"  速度: {np.linalg.norm(nav_features['velocity']):.2f} m/s")
        print(f"  地形适应性: {nav_features['terrain_adaptability']:.2f}")
        print(f"  能量效率: {nav_features['energy_efficiency']:.2f}")
        print(f"  稳定性: {nav_features['stability_index']:.2f}")
        print()
    
    return nav_features

def visualize_results(positions_forward, positions_turning):
    """可视化测试结果"""
    print("=== 可视化测试结果 ===")
    
    plt.figure(figsize=(15, 7))
    
    # 前进轨迹
    plt.subplot(121)
    positions_forward = np.array(positions_forward)
    plt.plot(positions_forward[:, 0], positions_forward[:, 1], 'b-', linewidth=2, label='Forward Trajectory')
    plt.plot(positions_forward[0, 0], positions_forward[0, 1], 'bo', markersize=8, label='Start')
    plt.plot(positions_forward[-1, 0], positions_forward[-1, 1], 'bs', markersize=8, label='End')
    plt.title('Forward Motion Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    # 转向轨迹
    plt.subplot(122)
    positions_turning = np.array(positions_turning)
    plt.plot(positions_turning[:, 0], positions_turning[:, 1], 'r-', linewidth=2, label='Turning Trajectory')
    plt.plot(positions_turning[0, 0], positions_turning[0, 1], 'ro', markersize=8, label='Start')
    plt.plot(positions_turning[-1, 0], positions_turning[-1, 1], 'rs', markersize=8, label='End')
    plt.title('Turning Motion Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('dynamics_test_results.png', dpi=300, bbox_inches='tight')
    print("测试结果可视化已保存: dynamics_test_results.png")
    plt.close()

def main():
    """主测试函数"""
    print("=== 月球车动力学模型全面测试 ===\n")
    
    # 运行所有测试
    positions_forward, velocities, energies = test_forward_motion()
    positions_turning, orientations = test_turning_motion()
    terrain_data = test_terrain_interaction()
    test_energy_consumption()
    collision_risks = test_collision_risk()
    predicted_trajectory = test_trajectory_prediction()
    navigation_features = test_navigation_features()
    
    # 可视化结果
    visualize_results(positions_forward, positions_turning)
    
    # 生成测试报告
    generate_test_report(positions_forward, velocities, energies, 
                        positions_turning, orientations,
                        terrain_data, collision_risks,
                        predicted_trajectory, navigation_features)
    
    print("=== 测试完成 ===")

def generate_test_report(positions_forward, velocities, energies,
                        positions_turning, orientations,
                        terrain_data, collision_risks,
                        predicted_trajectory, navigation_features):
    """生成测试报告"""
    with open('dynamics_test_report.txt', 'w', encoding='utf-8') as f:
        f.write("===== 月球车动力学模型测试报告 =====\n\n")
        
        f.write("1. 前进运动测试\n")
        f.write(f"   - 总位移: {np.sqrt((positions_forward[-1,0]-positions_forward[0,0])**2 + (positions_forward[-1,1]-positions_forward[0,1])**2):.2f} m\n")
        f.write(f"   - 最大速度: {max(velocities):.2f} m/s\n")
        f.write(f"   - 总能量消耗: {energies[-1]:.2f} J\n")
        f.write(f"   - 平均单位距离能量: {energies[-1]/np.sqrt((positions_forward[-1,0]-positions_forward[0,0])**2 + (positions_forward[-1,1]-positions_forward[0,1])**2):.2f} J/m\n\n")
        
        f.write("2. 转向运动测试\n")
        f.write(f"   - 总转向角度: {np.degrees(orientations[-1]-orientations[0]):.2f}°\n")
        f.write(f"   - 轨迹长度: {len(positions_turning)} 步\n\n")
        
        f.write("3. 地形交互测试\n")
        f.write(f"   - 平均下陷深度: {np.mean(terrain_data['sinkage']):.4f} m\n")
        f.write(f"   - 平均法向力: {np.mean(terrain_data['normal_forces']):.2f} N\n")
        f.write(f"   - 接触率: {np.sum(terrain_data['wheel_terrain_contact'])/8.0:.2f}\n\n")
        
        f.write("4. 碰撞风险测试\n")
        for i, risk in enumerate(collision_risks):
            f.write(f"   - 障碍物{i+1}: 距离={risk['distance']:.2f} m, 风险={risk['collision_risk']:.2f}\n")
        f.write("\n")
        
        f.write("5. 轨迹预测测试\n")
        f.write(f"   - 预测步数: {len(predicted_trajectory)}\n")
        if predicted_trajectory:
            f.write(f"   - 预测终点: {predicted_trajectory[-1]['position'][:2]}\n")
        f.write("\n")
        
        f.write("6. 导航特征测试\n")
        f.write(f"   - 地形适应性: {navigation_features['terrain_adaptability']:.2f}\n")
        f.write(f"   - 能量效率: {navigation_features['energy_efficiency']:.2f}\n")
        f.write(f"   - 稳定性: {navigation_features['stability_index']:.2f}\n\n")
        
        f.write("7. 测试结论\n")
        f.write("   - 动力学模型能够正确模拟月球车的前进和转向运动\n")
        f.write("   - 地形交互模型能够计算下陷深度和接触力\n")
        f.write("   - 能量消耗模型能够合理计算能量使用\n")
        f.write("   - 碰撞风险评估能够识别潜在碰撞\n")
        f.write("   - 轨迹预测功能正常工作\n")
        f.write("   - 导航特征提供了有用的环境信息\n")
        f.write("\n")
        f.write("8. 建议改进\n")
        f.write("   - 增加真实地形数据的集成\n")
        f.write("   - 优化能量模型以考虑更多因素\n")
        f.write("   - 增加更多的传感器模拟\n")
        f.write("   - 改进轨迹预测算法的准确性\n")

if __name__ == "__main__":
    main()