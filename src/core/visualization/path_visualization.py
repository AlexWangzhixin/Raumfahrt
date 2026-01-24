#!/usr/bin/env python3
"""
路径可视化模块
基于嫦娥6号数据的月球车路径可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class PathVisualizer:
    """
    路径可视化类
    """
    
    def __init__(self, config=None):
        """
        初始化路径可视化器
        
        Args:
            config: 配置参数
        """
        # 默认配置
        default_config = {
            'figsize': (12, 10),
            'dpi': 300,
            'output_dir': 'data/visualizations/path',
            'show_plot': False,
        }
        
        # 合并配置
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # 确保输出目录存在
        import os
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        print("路径可视化器初始化完成")
    
    def visualize_path(self, path, obstacles=None, key_nodes=None, process_steps=None, terrain_data=None, save=True, filename='path_visualization'):
        """
        可视化路径
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            key_nodes: 关键节点索引
            process_steps: 处理步骤
            terrain_data: 地形数据
            save: 是否保存
            filename: 保存文件名
        """
        print(f"开始可视化路径: 路径点数量={len(path)}")
        
        # 检查路径是否为空
        if not path:
            print("路径为空，无法可视化")
            return
        
        # 创建拓扑图视图
        self.visualize_topological_map(path, key_nodes, save=save, filename=f'{filename}_topological')
        
        # 创建流程图视图
        self.visualize_flowchart(path, process_steps, save=save, filename=f'{filename}_flowchart')
        
        # 创建地理分布图视图
        self.visualize_geographic_map(path, obstacles, terrain_data, save=save, filename=f'{filename}_geographic')
        
        print("路径可视化完成")
    
    def visualize_topological_map(self, path, key_nodes=None, save=True, filename='topological_map'):
        """
        可视化拓扑图
        
        Args:
            path: 路径点列表
            key_nodes: 关键节点索引
            save: 是否保存
            filename: 保存文件名
        """
        plt.figure(figsize=self.config['figsize'])
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for i, point in enumerate(path):
            node_color = 'lightblue'
            node_size = 500
            
            # 关键节点使用不同颜色
            if key_nodes and i in key_nodes:
                node_color = 'yellow'
                node_size = 800
            
            # 起点和终点使用不同颜色
            if i == 0:
                node_color = 'green'
                node_size = 1000
            elif i == len(path) - 1:
                node_color = 'red'
                node_size = 1000
            
            G.add_node(i, pos=(point[0], point[1]), color=node_color, size=node_size)
        
        # 添加边
        for i in range(len(path) - 1):
            G.add_edge(i, i+1, weight=1)
        
        # 获取节点位置和颜色
        pos = nx.get_node_attributes(G, 'pos')
        node_colors = [G.nodes[n]['color'] for n in G.nodes]
        node_sizes = [G.nodes[n]['size'] for n in G.nodes]
        
        # 绘制图
        nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, arrows=True)
        
        # 添加边标签
        edge_labels = {(u, v): f'{i+1}' for i, (u, v) in enumerate(G.edges)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title('路径拓扑图')
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.grid(True)
        plt.axis('equal')
        
        if save:
            save_path = f"{self.config['output_dir']}/{filename}.svg"
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"拓扑图保存到: {save_path}")
        
        if not self.config['show_plot']:
            plt.close()
    
    def visualize_flowchart(self, path, process_steps=None, save=True, filename='flowchart'):
        """
        可视化流程图
        
        Args:
            path: 路径点列表
            process_steps: 处理步骤
            save: 是否保存
            filename: 保存文件名
        """
        plt.figure(figsize=(15, 8))
        
        # 创建流程图
        G = nx.DiGraph()
        
        # 添加处理步骤节点
        if process_steps:
            for i, step in enumerate(process_steps):
                G.add_node(f'step_{i}', label=step, pos=(i*2, 2))
            
            # 添加步骤之间的边
            for i in range(len(process_steps) - 1):
                G.add_edge(f'step_{i}', f'step_{i+1}')
        
        # 添加路径节点
        for i, point in enumerate(path):
            G.add_node(f'path_{i}', label=f'Waypoint {i}', pos=(i, 0))
        
        # 添加路径边
        for i in range(len(path) - 1):
            G.add_edge(f'path_{i}', f'path_{i+1}')
        
        # 添加步骤和路径之间的连接
        if process_steps and path:
            G.add_edge('step_0', 'path_0')
            G.add_edge(f'path_{len(path)-1}', f'step_{len(process_steps)-1}')
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制图
        nx.draw(G, pos, with_labels=False, arrows=True, node_shape='s', node_size=1000, node_color='lightblue')
        
        # 添加节点标签
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        plt.title('路径处理流程图')
        plt.grid(True)
        plt.axis('equal')
        
        if save:
            save_path = f"{self.config['output_dir']}/{filename}.svg"
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"流程图保存到: {save_path}")
        
        if not self.config['show_plot']:
            plt.close()
    
    def visualize_geographic_map(self, path, obstacles=None, terrain_data=None, save=True, filename='geographic_map'):
        """
        可视化地理分布图
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            terrain_data: 地形数据
            save: 是否保存
            filename: 保存文件名
        """
        plt.figure(figsize=self.config['figsize'])
        
        # 提取路径坐标
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # 绘制地形（如果有）
        if terrain_data and 'elevation' in terrain_data:
            elevation = terrain_data['elevation']
            plt.imshow(elevation, cmap='terrain', origin='lower', alpha=0.5)
        
        # 绘制障碍物
        if obstacles:
            for obstacle in obstacles:
                pos = obstacle['position']
                size = obstacle['size']
                # 绘制障碍物
                circle = plt.Circle((pos[0], pos[1]), max(size)/2, color='red', alpha=0.3)
                plt.gca().add_patch(circle)
                # 绘制障碍物标签
                plt.text(pos[0], pos[1], f"Obs {obstacle.get('id', '')}", ha='center', va='center')
        
        # 绘制路径
        plt.plot(x, y, 'b-', linewidth=2, label='规划路径')
        
        # 绘制路径点
        plt.scatter(x, y, c='blue', s=50, label='路径点')
        
        # 绘制起点和终点
        plt.scatter(x[0], y[0], c='green', marker='o', s=200, label='起点')
        plt.scatter(x[-1], y[-1], c='red', marker='x', s=200, label='终点')
        
        # 绘制关键节点
        if 'key_nodes' in locals() and key_nodes:
            key_x = x[key_nodes]
            key_y = y[key_nodes]
            plt.scatter(key_x, key_y, c='yellow', marker='*', s=150, label='关键节点')
        
        plt.title('路径地理分布图')
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        if save:
            save_path = f"{self.config['output_dir']}/{filename}.svg"
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"地理分布图保存到: {save_path}")
        
        if not self.config['show_plot']:
            plt.close()
    
    def visualize_interactive(self, path, obstacles=None, terrain_data=None):
        """
        可视化交互式地图
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            terrain_data: 地形数据
        """
        # 简化版本：创建一个可交互的地图
        plt.figure(figsize=self.config['figsize'])
        
        # 提取路径坐标
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # 绘制地形（如果有）
        if terrain_data and 'elevation' in terrain_data:
            elevation = terrain_data['elevation']
            plt.imshow(elevation, cmap='terrain', origin='lower', alpha=0.5)
        
        # 绘制障碍物
        if obstacles:
            for obstacle in obstacles:
                pos = obstacle['position']
                size = obstacle['size']
                circle = plt.Circle((pos[0], pos[1]), max(size)/2, color='red', alpha=0.3)
                plt.gca().add_patch(circle)
        
        # 绘制路径
        plt.plot(x, y, 'b-', linewidth=2, label='规划路径')
        plt.scatter(x, y, c='blue', s=50, label='路径点')
        plt.scatter(x[0], y[0], c='green', marker='o', s=200, label='起点')
        plt.scatter(x[-1], y[-1], c='red', marker='x', s=200, label='终点')
        
        plt.title('交互式路径地图')
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 启用交互模式
        plt.ion()
        plt.show()
        
        # 等待用户关闭
        input("按Enter键关闭窗口...")
        plt.close()
    
    def save_all_modes(self, path, obstacles=None, key_nodes=None, process_steps=None, terrain_data=None, filename='all_modes'):
        """
        保存所有视图模式
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            key_nodes: 关键节点索引
            process_steps: 处理步骤
            terrain_data: 地形数据
            filename: 保存文件名
        """
        print("保存所有视图模式")
        
        # 保存拓扑图
        self.visualize_topological_map(path, key_nodes, save=True, filename=f'{filename}_topological')
        
        # 保存流程图
        self.visualize_flowchart(path, process_steps, save=True, filename=f'{filename}_flowchart')
        
        # 保存地理分布图
        self.visualize_geographic_map(path, obstacles, terrain_data, save=True, filename=f'{filename}_geographic')
        
        print("所有视图模式保存完成")