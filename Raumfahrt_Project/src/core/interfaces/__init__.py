#!/usr/bin/env python3
"""
定义抽象基类 (Interface)
为各个模块提供统一的接口标准
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class EnvironmentInterface(ABC):
    """
    环境建模接口
    定义环境模型的基本方法
    """
    
    @abstractmethod
    def get_elevation(self, position: Tuple[float, float]) -> float:
        """
        获取指定位置的高程
        
        Args:
            position: (x, y) 位置坐标
        
        Returns:
            高程值 (m)
        """
        pass
    
    @abstractmethod
    def get_soil_properties(self, position: Tuple[float, float]) -> Dict[str, float]:
        """
        获取指定位置的土壤参数
        
        Args:
            position: (x, y) 位置坐标
        
        Returns:
            土壤参数字典
        """
        pass
    
    @abstractmethod
    def is_obstacle(self, position: Tuple[float, float]) -> bool:
        """
        检查指定位置是否为障碍物
        
        Args:
            position: (x, y) 位置坐标
        
        Returns:
            是否为障碍物
        """
        pass

class DynamicsInterface(ABC):
    """
    动力学建模接口
    定义动力学模型的基本方法
    """
    
    @abstractmethod
    def update_state(self, control_input: Dict[str, Any], time_step: float) -> Dict[str, np.ndarray]:
        """
        更新系统状态
        
        Args:
            control_input: 控制输入
            time_step: 时间步长 (s)
        
        Returns:
            更新后的状态
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        获取当前状态
        
        Returns:
            当前状态
        """
        pass
    
    @abstractmethod
    def reset(self, initial_state: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        重置系统状态
        
        Args:
            initial_state: 初始状态，如果为None则使用默认值
        
        Returns:
            重置后的状态
        """
        pass

class PlanningInterface(ABC):
    """
    路径规划接口
    定义路径规划算法的基本方法
    """
    
    @abstractmethod
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float], **kwargs) -> List[Tuple[float, float]]:
        """
        规划路径
        
        Args:
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            **kwargs: 其他参数
        
        Returns:
            路径点列表 [(x1, y1), (x2, y2), ...]
        """
        pass
    
    @abstractmethod
    def update_map(self, map_data: Any) -> None:
        """
        更新地图数据
        
        Args:
            map_data: 地图数据
        """
        pass
    
    @abstractmethod
    def get_cost(self, path: List[Tuple[float, float]]) -> float:
        """
        计算路径代价
        
        Args:
            path: 路径点列表
        
        Returns:
            路径代价
        """
        pass

class PerceptionInterface(ABC):
    """
    感知系统接口
    定义感知系统的基本方法
    """
    
    @abstractmethod
    def perceive(self, **kwargs) -> Dict[str, Any]:
        """
        执行感知操作
        
        Args:
            **kwargs: 感知参数
        
        Returns:
            感知结果
        """
        pass
    
    @abstractmethod
    def update(self, sensor_data: Dict[str, Any]) -> None:
        """
        更新传感器数据
        
        Args:
            sensor_data: 传感器数据
        """
        pass
    
    @abstractmethod
    def get_map(self) -> Any:
        """
        获取感知地图
        
        Returns:
            感知地图
        """
        pass

class ControlInterface(ABC):
    """
    控制系统接口
    定义控制系统的基本方法
    """
    
    @abstractmethod
    def compute_control(self, state: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算控制量
        
        Args:
            state: 当前状态
            reference: 参考值
        
        Returns:
            控制量
        """
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict[str, float]) -> None:
        """
        更新控制参数
        
        Args:
            parameters: 控制参数
        """
        pass

# 导出所有接口
__all__ = [
    'EnvironmentInterface',
    'DynamicsInterface',
    'PlanningInterface',
    'PerceptionInterface',
    'ControlInterface',
]
