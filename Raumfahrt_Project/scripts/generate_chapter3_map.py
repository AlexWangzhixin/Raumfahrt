#!/usr/bin/env python3
"""
生成第3章地图文件
"""

import numpy as np
import os

def main():
    """
    主函数
    """
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 生成默认地图数据
    default_map = {
        'elevation_map': np.random.rand(100, 100) * 5 - 2.5,
        'obstacle_map': np.zeros((100, 100)),
        'traversability_map': np.random.rand(100, 100),
        'physics_map': np.zeros((100, 100, 5)),
        'map_resolution': 0.1,
        'map_size': (10.0, 10.0)
    }
    
    # 设置默认物理属性
    # [kc, kphi, n, c, phi]
    default_map['physics_map'][:, :, 0] = 1.4e3  # kc
    default_map['physics_map'][:, :, 1] = 8.2e5  # kphi
    default_map['physics_map'][:, :, 2] = 1.0    # n
    default_map['physics_map'][:, :, 3] = 0.17e3 # c
    default_map['physics_map'][:, :, 4] = 30     # phi
    
    # 保存地图数据
    map_path = os.path.join('data', 'chapter3_map.npz')
    np.savez(map_path, **default_map)
    
    print('第3章地图生成完成')
    print('地图文件路径:', os.path.abspath(map_path))

if __name__ == "__main__":
    main()
