import pybullet as p 
import pybullet_data 
import numpy as np 
import time 
from lunar_terrain import LunarTerrainModel

class PyBulletLunarRover:
    """
    【第四章核心模型】高保真月面巡视器动力学仿真
    功能：
    1. 集成第三章的月面地形模型
    2. 提供高保真动力学仿真
    3. 作为物理真值用于强化学习
    4. 验证数字孪生预测模型
    """
    def __init__(self, terrain_mesh_path=None, gui=True):
        # 1. 引擎初始化
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -1.625) # 月球重力
        p.setRealTimeSimulation(0)  # 关闭实时仿真，手动控制
        
        # 2. 加载月面地形 (衔接第三章)
        self.terrain_model = LunarTerrainModel(
            dtm_path=terrain_mesh_path,
            resolution=0.5,
            size=20.0
        )
        
        # 处理地形数据
        self.terrain_model.load_dtm_data()
        self.terrain_model.process_terrain()
        
        # 在PyBullet中创建地形
        self.terrain_id = self.terrain_model.create_pybullet_terrain(self.client)

        # 3. 加载巡视器 (月球车)
        # 这里暂时用R2D2代替，后续应替换为玉兔二号的URDF
        start_pos = [0, 0, 0.5]  # 初始位置，离地面0.5m
        start_orn = p.getQuaternionFromEuler([0, 0, 0])  # 初始姿态
        self.rover_id = p.loadURDF("r2d2.urdf", start_pos, start_orn)
        
        # 获取轮子关节索引 (R2D2的轮子关节ID)
        self.wheel_joints = [2, 3, 6, 7]  # 左右前后轮
        self.num_wheels = len(self.wheel_joints)
        self.max_force = 100  # 电机最大扭矩
        
        # 设置月球车物理参数
        self._setup_rover_dynamics()
        
        # 4. 状态记录
        self.trace_log = []
        self.current_time = 0.0
        self.dt = 0.01  # 仿真步长
    
    def _setup_rover_dynamics(self):
        """
        设置月球车的物理动力学参数
        """
        # 设置车身参数
        p.changeDynamics(self.rover_id, -1, 
                        mass=150.0,  # 月球车质量 (kg)
                        lateralFriction=0.8,
                        spinningFriction=0.2,
                        rollingFriction=0.1
                        )
        
        # 设置轮子参数
        for joint in self.wheel_joints:
            p.changeDynamics(self.rover_id, joint,
                            lateralFriction=0.9,
                            spinningFriction=0.1,
                            rollingFriction=0.05,
                            restitution=0.01
                            )
    
    def reset(self, x=0, y=0, theta=0):
        """
        重置月球车状态
        :param x: 初始x坐标 (m)
        :param y: 初始y坐标 (m)
        :param theta: 初始航向角 (rad)
        :return: 当前状态
        """
        p.resetBasePositionAndOrientation(
            self.rover_id,
            [x, y, 0.5],
            p.getQuaternionFromEuler([0, 0, theta])
        )
        
        # 重置轮子状态
        for joint in self.wheel_joints:
            p.resetJointState(self.rover_id, joint, targetValue=0, targetVelocity=0)
        
        # 重置记录
        self.trace_log = []
        self.current_time = 0.0
        
        return self.get_state()
    
    def get_state(self):
        """
        获取月球车当前状态
        :return: [x, y, theta, vx, vy, omega]
        """
        # 获取位置和姿态
        pos, orn = p.getBasePositionAndOrientation(self.rover_id)
        euler = p.getEulerFromQuaternion(orn)
        theta = euler[2]  # yaw角
        
        # 获取线速度和角速度
        linear_vel, angular_vel = p.getBaseVelocity(self.rover_id)
        
        # 组合状态向量
        state = np.array([
            pos[0], pos[1], theta,  # 位置和姿态
            linear_vel[0], linear_vel[1], angular_vel[2]  # 速度
        ])
        
        return state
    
    def step(self, target_linear_v, target_angular_w, sim_steps=10):
        """
        执行动力学仿真步进
        :param target_linear_v: 目标线速度 (m/s)
        :param target_angular_w: 目标角速度 (rad/s)
        :param sim_steps: 每步执行的物理仿真次数
        :return: 当前状态
        """
        # 1. 差速驱动运动学逆解
        # 假设轮距 width = 0.5m, 轮径 radius = 0.15m
        wheel_base = 0.5  # 轮距
        wheel_radius = 0.15  # 轮径
        
        # 计算左右轮速度
        v_left = target_linear_v - target_angular_w * wheel_base / 2
        v_right = target_linear_v + target_angular_w * wheel_base / 2
        
        # 转换为角速度 (rad/s)
        omega_left = v_left / wheel_radius
        omega_right = v_right / wheel_radius
        
        # 2. 设置电机控制指令
        # 左前轮和左后轮设为相同速度
        # 右前轮和右后轮设为相同速度
        target_velocities = [
            omega_left,   # 左前轮
            omega_left,   # 左后轮
            omega_right,  # 右前轮
            omega_right   # 右后轮
        ]
        
        # 使用位置控制模式模拟速度控制
        p.setJointMotorControlArray(
            self.rover_id,
            self.wheel_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_velocities,
            forces=[self.max_force] * self.num_wheels
        )
        
        # 3. 执行物理仿真
        for _ in range(sim_steps):
            p.stepSimulation()
            time.sleep(self.dt / sim_steps)  # 保持仿真速度
        
        # 4. 更新时间
        self.current_time += self.dt
        
        # 5. 获取当前状态
        state = self.get_state()
        
        # 6. 记录状态
        self.trace_log.append({
            'time': self.current_time,
            'state': state.copy(),
            'target_v': target_linear_v,
            'target_w': target_angular_w
        })
        
        return state
    
    def save_experiment_data(self, filename="lunar_rover_experiment.csv"):
        """
        保存实验数据
        :param filename: 保存文件名
        """
        import pandas as pd
        
        # 转换为DataFrame
        data = []
        for record in self.trace_log:
            data.append({
                'time': record['time'],
                'x': record['state'][0],
                'y': record['state'][1],
                'theta': record['state'][2],
                'vx': record['state'][3],
                'vy': record['state'][4],
                'omega': record['state'][5],
                'target_v': record['target_v'],
                'target_w': record['target_w']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"实验数据已保存至 {filename}")
    
    def visualize_trajectory(self, save_path="rover_trajectory.png"):
        """
        可视化月球车轨迹
        :param save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        if not self.trace_log:
            print("没有轨迹数据可可视化")
            return
        
        # 提取轨迹数据
        x = [record['state'][0] for record in self.trace_log]
        y = [record['state'][1] for record in self.trace_log]
        
        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'b-', linewidth=2, label='月球车轨迹')
        plt.scatter(x[0], y[0], c='g', marker='o', s=100, label='起点')
        plt.scatter(x[-1], y[-1], c='r', marker='x', s=100, label='终点')
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.title('月球车运动轨迹')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹可视化已保存到 {save_path}")
        
        plt.close()
    
    def disconnect(self):
        """
        断开PyBullet连接
        """
        p.disconnect(self.client)

# 测试脚本
if __name__ == "__main__":
    print("启动月面巡视器动力学仿真...")
    
    # 创建仿真环境
    rover_sim = PyBulletLunarRover(
        terrain_mesh_path="NAC_DTM_CHANGE4.tiff",
        gui=True
    )
    
    try:
        # 重置状态
        rover_sim.reset(x=0, y=0, theta=0)
        
        print("开始仿真，按Ctrl+C停止...")
        
        # 仿真主循环
        for i in range(500):
            # 生成S形轨迹指令
            t = i * 0.01
            target_v = 0.5  # 线速度0.5m/s
            target_w = 0.5 * np.sin(t * 2)  # 角速度随时间正弦变化
            
            # 执行步进
            state = rover_sim.step(target_v, target_w)
            
            # 每50步打印一次状态
            if i % 50 == 0:
                print(f"时间: {rover_sim.current_time:.2f}s, 位置: ({state[0]:.2f}, {state[1]:.2f})m, 航向: {state[2]:.2f}rad")
        
        # 保存实验数据
        rover_sim.save_experiment_data()
        
        # 可视化轨迹
        rover_sim.visualize_trajectory()
        
        print("仿真完成！")
        
    except KeyboardInterrupt:
        print("用户中断仿真")
    finally:
        # 断开连接
        rover_sim.disconnect()