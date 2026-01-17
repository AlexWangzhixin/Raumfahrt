import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pybullet as p
import pybullet_data
import os

class LunarTerrainModel:
    """
    【第三章核心功能】月面地形建模
    功能：
    1. 加载LROC NAC DTM数据
    2. 生成PyBullet可用的高度场
    3. 设置月面物理参数（摩擦、弹性等）
    """
    def __init__(self, dtm_path="NAC_DTM_CHANGE4.tiff", resolution=0.5, size=20.0):
        """
        初始化月面地形模型
        :param dtm_path: 数字地形模型路径
        :param resolution: 地形分辨率 (m/pixel)
        :param size: 地形尺寸 (m)
        """
        self.dtm_path = dtm_path
        self.resolution = resolution
        self.size = size
        self.terrain_data = None
        self.height_field = None
        
    def load_dtm_data(self):
        """
        加载数字地形模型数据
        """
        try:
            # 使用PIL加载TIFF文件
            img = Image.open(self.dtm_path)
            # 转换为numpy数组
            self.terrain_data = np.array(img)
            
            # 过滤无效值（-3.4e+38是float32的最小值，表示无效数据）
            invalid_mask = self.terrain_data <= -1e38
            valid_data = self.terrain_data[~invalid_mask]
            
            if len(valid_data) > 0:
                # 用有效值的平均值填充无效值
                valid_mean = valid_data.mean()
                self.terrain_data[invalid_mask] = valid_mean
                print(f"成功加载DTM数据：")
                print(f"  原始形状={self.terrain_data.shape}")
                print(f"  无效值占比={np.sum(invalid_mask)/self.terrain_data.size*100:.2f}%")
                print(f"  有效数据范围={valid_data.min():.2f}~{valid_data.max():.2f}")
                print(f"  填充后范围={self.terrain_data.min():.2f}~{self.terrain_data.max():.2f}")
            else:
                print("警告：所有数据都是无效值，生成默认地形")
                self._generate_default_terrain()
                return False
                
            return True
        except Exception as e:
            print(f"加载DTM数据失败：{e}")
            # 如果加载失败，生成默认地形
            self._generate_default_terrain()
            return False
    
    def _generate_default_terrain(self):
        """
        生成默认的月面地形（用于测试）
        """
        # 创建一个200x200的随机地形
        np.random.seed(42)
        self.terrain_data = np.random.rand(200, 200) * 2 - 1
        print("生成默认随机地形数据")
    
    def process_terrain(self, smoothing=True):
        """
        处理地形数据，生成PyBullet可用的高度场
        :param smoothing: 是否平滑地形
        """
        if self.terrain_data is None:
            self.load_dtm_data()
        
        # 调整地形尺寸
        target_size = int(self.size / self.resolution)
        
        # 缩放地形数据到目标尺寸
        from scipy.ndimage import zoom
        scaled_terrain = zoom(self.terrain_data, (target_size / self.terrain_data.shape[0], 
                                                 target_size / self.terrain_data.shape[1]),
                            order=1)  # 线性插值
        
        # 平滑地形（可选）
        if smoothing:
            from scipy.ndimage import gaussian_filter
            scaled_terrain = gaussian_filter(scaled_terrain, sigma=1)
        
        # 归一化地形高度到合理范围
        min_height = scaled_terrain.min()
        max_height = scaled_terrain.max()
        scaled_terrain = (scaled_terrain - min_height) / (max_height - min_height) * 2 - 1
        
        # 转换为PyBullet高度场格式（需要是flat array）
        self.height_field = scaled_terrain.flatten()
        self.height_field_shape = scaled_terrain.shape
        
        print(f"处理完成：高度场形状={self.height_field_shape}, 范围={self.height_field.min()}~{self.height_field.max()}")
        return self.height_field
    
    def create_pybullet_terrain(self, client_id):
        """
        在PyBullet中创建月面地形
        :param client_id: PyBullet客户端ID
        :return: 地形ID
        """
        if self.height_field is None:
            self.process_terrain()
        
        # 设置高度场
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.resolution, self.resolution, 1.0],
            heightfieldTextureScaling=int(self.size / 0.1),  # 纹理缩放
            heightfieldData=self.height_field,
            numHeightfieldRows=self.height_field_shape[0],
            numHeightfieldColumns=self.height_field_shape[1]
        )
        
        # 创建地形实例
        terrain_id = p.createMultiBody(
            baseMass=0,  # 静态地形
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0]
        )
        
        # 设置月面物理参数
        p.changeDynamics(terrain_id, -1, 
                        lateralFriction=0.6,    # 横向摩擦系数（月壤）
                        spinningFriction=0.1,   # 旋转摩擦系数
                        rollingFriction=0.05,    # 滚动摩擦系数
                        restitution=0.05,        # 弹性恢复系数（月面几乎无弹性）
                        frictionAnchor=True      # 摩擦锚定
                        )
        
        print(f"成功创建PyBullet地形，ID={terrain_id}")
        return terrain_id
    
    def visualize_terrain(self, save_path="lunar_terrain.png"):
        """
        可视化地形数据
        :param save_path: 保存路径
        """
        if self.terrain_data is None:
            self.load_dtm_data()
        
        plt.figure(figsize=(10, 10))
        
        # 提取有效数据范围用于归一化
        valid_min = np.min(self.terrain_data[self.terrain_data > -1e38])
        valid_max = np.max(self.terrain_data[self.terrain_data > -1e38])
        
        # 归一化到0-1范围用于显示
        normalized_terrain = (self.terrain_data - valid_min) / (valid_max - valid_min)
        
        # 使用viridis colormap，更适合地形显示
        plt.imshow(normalized_terrain, cmap='viridis', origin='upper')
        
        # 添加颜色条，显示实际高度值
        cbar = plt.colorbar(label='Height (m)')
        # 调整颜色条刻度显示实际高度
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels([f'{valid_min:.1f}', f'{(valid_min+valid_max*0.25):.1f}', 
                           f'{(valid_min+valid_max*0.5):.1f}', f'{(valid_min+valid_max*0.75):.1f}', 
                           f'{valid_max:.1f}'])
        
        # 使用英文标题避免中文字体问题
        plt.title('Lunar Terrain Data')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"地形可视化已保存到 {save_path}")
        
        plt.close()

# 测试地形建模
if __name__ == "__main__":
    # 创建月面地形模型
    lunar_terrain = LunarTerrainModel(dtm_path="NAC_DTM_CHANGE4.tiff", resolution=0.5, size=20.0)
    
    # 加载并处理地形
    lunar_terrain.load_dtm_data()
    lunar_terrain.process_terrain()
    
    # 可视化地形
    lunar_terrain.visualize_terrain()
    
    print("月面地形建模完成！")