# -*- coding: utf-8 -*-
"""
感知系统模块 (Perception System)
实现YOLOv5目标检测和DisNet距离估计的融合感知框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from config import config


class YOLOv5Backbone(nn.Module):
    """
    YOLOv5骨干网络 (简化版CSPDarknet53)
    用于从RGB图像中提取特征并检测障碍物
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        """
        初始化YOLOv5骨干网络

        Args:
            in_channels: 输入通道数 (RGB=3)
            num_classes: 目标类别数 (障碍物和目标点)
        """
        super(YOLOv5Backbone, self).__init__()

        self.num_classes = num_classes

        # Focus层：将图像空间信息压缩到通道维度
        self.focus = Focus(in_channels, 32)

        # CSP模块堆叠
        self.csp1 = CSPBlock(32, 64, n=1)
        self.csp2 = CSPBlock(64, 128, n=3)
        self.csp3 = CSPBlock(128, 256, n=3)
        self.csp4 = CSPBlock(256, 512, n=1)

        # SPP空间金字塔池化
        self.spp = SPP(512, 512)

        # 检测头
        # 输出: [batch, num_anchors * (5 + num_classes), height, width]
        # 5 = (x, y, w, h, confidence)
        self.detect = nn.Conv2d(512, 3 * (5 + num_classes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [batch, 3, height, width]

        Returns:
            检测结果 [batch, num_detections, 5 + num_classes]
        """
        # 特征提取
        x = self.focus(x)
        x = self.csp1(x)
        x = self.csp2(x)
        x = self.csp3(x)
        x = self.csp4(x)
        x = self.spp(x)

        # 检测
        out = self.detect(x)

        return out


class Focus(nn.Module):
    """
    Focus层：通过空间到通道的重排提取特征
    将4个相邻像素堆叠到通道维度
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Focus, self).__init__()
        self.conv = ConvBlock(in_channels * 4, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间到通道: [b, c, h, w] -> [b, 4c, h/2, w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat([patch_top_left, patch_top_right,
                       patch_bot_left, patch_bot_right], dim=1)
        return self.conv(x)


class ConvBlock(nn.Module):
    """卷积块：Conv + BatchNorm + SiLU"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """
    CSP (Cross Stage Partial) 模块
    通过分割特征图并在bottleneck后合并来减少计算量
    """

    def __init__(self, in_channels: int, out_channels: int, n: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            n: bottleneck重复次数
        """
        super(CSPBlock, self).__init__()
        hidden_channels = out_channels // 2

        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv3 = ConvBlock(hidden_channels * 2, out_channels, 1, 1)

        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels) for _ in range(n)]
        )

        self.downsample = nn.MaxPool2d(2, 2) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.bottlenecks(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        x = self.downsample(x)
        return x


class Bottleneck(nn.Module):
    """Bottleneck模块"""

    def __init__(self, in_channels: int, out_channels: int):
        super(Bottleneck, self).__init__()
        hidden_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class SPP(nn.Module):
    """
    空间金字塔池化模块
    在多个尺度上池化特征
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in [5, 9, 13]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        pooled = [x] + [pool(x) for pool in self.pools]
        x = torch.cat(pooled, dim=1)
        return self.conv2(x)


class DisNet(nn.Module):
    """
    DisNet距离估计网络
    基于目标检测框和深度图估计目标距离

    距离计算公式: d = f * H / h
    其中:
        f: 相机焦距
        H: 目标实际高度
        h: 目标在图像中的高度（像素）
    """

    def __init__(self, input_channels: int = 4):
        """
        初始化DisNet

        Args:
            input_channels: 输入通道数 (RGB + Depth = 4)
        """
        super(DisNet, self).__init__()

        # 特征提取网络
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 距离回归头
        self.distance_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.ReLU(inplace=True)  # 距离必须为正
        )

        # 相机参数
        self.focal_length = config.CAMERA_FOCAL_LENGTH
        self.default_target_height = 0.5  # 默认目标高度 (m)

    def forward(self, x: torch.Tensor, bbox_height: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征图 [batch, channels, height, width]
            bbox_height: 目标边界框高度（像素）

        Returns:
            估计距离 [batch, 1]
        """
        features = self.features(x)
        distance = self.distance_head(features)

        # 如果提供了边界框高度，使用几何公式调整
        if bbox_height is not None:
            # d = f * H / h
            geometric_distance = self.focal_length * self.default_target_height / \
                                 (bbox_height + 1e-6)
            # 融合网络预测和几何计算
            distance = 0.5 * distance + 0.5 * geometric_distance.unsqueeze(1)

        return distance


class PerceptionSystem:
    """
    感知系统
    整合YOLOv5目标检测和DisNet距离估计
    """

    def __init__(self, device: torch.device = None):
        """
        初始化感知系统

        Args:
            device: 计算设备
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 初始化网络
        self.yolo = YOLOv5Backbone().to(self.device)
        self.disnet = DisNet().to(self.device)

        # 设置为评估模式
        self.yolo.eval()
        self.disnet.eval()

        # 相机参数
        self.focal_length = config.CAMERA_FOCAL_LENGTH
        self.principal_point = (
            config.CAMERA_RESOLUTION[0] / 2,
            config.CAMERA_RESOLUTION[1] / 2
        )

    def detect_obstacles(self, rgb_image: np.ndarray,
                         confidence_threshold: float = 0.5) -> List[Dict]:
        """
        检测图像中的障碍物

        Args:
            rgb_image: RGB图像 [height, width, 3]
            confidence_threshold: 置信度阈值

        Returns:
            检测结果列表，每个元素包含:
                - bbox: (x_min, y_min, x_max, y_max)
                - confidence: 置信度
                - class_id: 类别ID
        """
        # 预处理图像
        image_tensor = self._preprocess_image(rgb_image)

        # 推理
        with torch.no_grad():
            detections = self.yolo(image_tensor)

        # 后处理（简化版NMS）
        results = self._postprocess_detections(detections, confidence_threshold)

        return results

    def estimate_distance(self, rgb_image: np.ndarray,
                          depth_image: np.ndarray,
                          bbox: Tuple[float, float, float, float]) -> float:
        """
        估计目标距离

        Args:
            rgb_image: RGB图像
            depth_image: 深度图像
            bbox: 目标边界框 (x_min, y_min, x_max, y_max)

        Returns:
            估计距离 (m)
        """
        # 提取边界框区域
        x_min, y_min, x_max, y_max = [int(v) for v in bbox]

        # 裁剪区域
        rgb_crop = rgb_image[y_min:y_max, x_min:x_max]
        depth_crop = depth_image[y_min:y_max, x_min:x_max]

        if rgb_crop.size == 0 or depth_crop.size == 0:
            return config.MAX_DETECTION_DISTANCE

        # 合并RGB和深度
        combined = np.concatenate([
            rgb_crop,
            np.expand_dims(depth_crop, axis=-1)
        ], axis=-1)

        # 调整大小
        combined = self._resize_image(combined, (64, 64))

        # 转换为张量
        combined_tensor = torch.FloatTensor(combined).permute(2, 0, 1).unsqueeze(0)
        combined_tensor = combined_tensor.to(self.device)

        # 计算边界框高度
        bbox_height = torch.tensor([y_max - y_min], dtype=torch.float32).to(self.device)

        # 推理
        with torch.no_grad():
            distance = self.disnet(combined_tensor, bbox_height)

        return distance.item()

    def compute_target_direction(self, target_pixel: Tuple[int, int],
                                  rover_theta: float) -> float:
        """
        计算目标方向角

        Args:
            target_pixel: 目标在图像中的像素坐标 (x, y)
            rover_theta: 月球车当前航向角

        Returns:
            目标方向角 (rad)，范围 [-π, π]
        """
        # 相机内参
        cx, cy = self.principal_point
        f = self.focal_length * config.CAMERA_RESOLUTION[0]  # 像素焦距

        # 计算相对于图像中心的偏移
        dx = target_pixel[0] - cx
        dy = target_pixel[1] - cy

        # 计算方位角（相对于相机光轴）
        azimuth = np.arctan2(dx, f)

        # 转换到世界坐标系（考虑月球车航向）
        target_direction = rover_theta + azimuth

        # 归一化到 [-π, π]
        target_direction = np.arctan2(np.sin(target_direction),
                                       np.cos(target_direction))

        return target_direction

    def extract_state_features(self, depth_image: np.ndarray,
                               target_direction: float,
                               target_distance: float) -> np.ndarray:
        """
        提取状态特征向量

        构造特征向量 st = [DepthMap, θt, dt]

        Args:
            depth_image: 深度图像
            target_direction: 目标方向角
            target_distance: 目标距离

        Returns:
            状态特征向量
        """
        # 调整深度图大小
        resized_depth = self._resize_image(depth_image, config.DEPTH_IMAGE_SIZE)

        # 归一化
        normalized_depth = resized_depth / config.MAX_DETECTION_DISTANCE
        normalized_depth = np.clip(normalized_depth, 0, 1)

        return normalized_depth

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小为模型输入尺寸
        resized = self._resize_image(image, (640, 640))

        # 归一化到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # 转换为张量 [batch, channels, height, width]
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor

    def _postprocess_detections(self, detections: torch.Tensor,
                                confidence_threshold: float) -> List[Dict]:
        """后处理检测结果"""
        # 简化实现：返回模拟检测结果
        results = []
        # 实际实现中应解析YOLO输出并进行NMS
        return results

    def _resize_image(self, image: np.ndarray,
                      target_size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小"""
        from scipy.ndimage import zoom

        if len(image.shape) == 2:
            # 灰度图
            current_size = image.shape
            zoom_factors = (target_size[1] / current_size[0],
                            target_size[0] / current_size[1])
            return zoom(image, zoom_factors, order=1)
        else:
            # 彩色图
            current_size = image.shape[:2]
            zoom_factors = (target_size[1] / current_size[0],
                            target_size[0] / current_size[1], 1)
            return zoom(image, zoom_factors, order=1)


def calculate_ciou_loss(pred_boxes: torch.Tensor,
                        target_boxes: torch.Tensor) -> torch.Tensor:
    """
    计算CIoU (Complete Intersection over Union) 损失

    CIoU = IoU - ρ²(b, b_gt) / c² - αv

    其中:
        ρ: 中心点距离
        c: 最小包围框对角线长度
        α: 权重参数
        v: 宽高比一致性

    Args:
        pred_boxes: 预测框 [N, 4] (x1, y1, x2, y2)
        target_boxes: 目标框 [N, 4]

    Returns:
        CIoU损失
    """
    # 计算IoU
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])

    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # 计算中心点
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    # 中心点距离的平方
    rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # 最小包围框对角线
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # 宽高比一致性
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]

    v = (4 / np.pi ** 2) * torch.pow(
        torch.atan(target_w / (target_h + 1e-7)) -
        torch.atan(pred_w / (pred_h + 1e-7)), 2
    )

    alpha = v / (1 - iou + v + 1e-7)

    # CIoU
    ciou = iou - rho2 / (c2 + 1e-7) - alpha * v

    return 1 - ciou.mean()


# 测试代码
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建感知系统
    perception = PerceptionSystem(device)
    print("感知系统创建成功")

    # 测试目标检测
    print("\n=== 测试目标检测 ===")
    rgb_image = np.random.rand(480, 640, 3).astype(np.float32)
    detections = perception.detect_obstacles(rgb_image)
    print(f"检测到 {len(detections)} 个目标")

    # 测试距离估计
    print("\n=== 测试距离估计 ===")
    depth_image = np.random.rand(480, 640).astype(np.float32) * 5
    bbox = (100, 100, 200, 250)
    distance = perception.estimate_distance(rgb_image, depth_image, bbox)
    print(f"估计距离: {distance:.2f} m")

    # 测试方向计算
    print("\n=== 测试方向计算 ===")
    target_pixel = (400, 240)
    rover_theta = 0.5
    direction = perception.compute_target_direction(target_pixel, rover_theta)
    print(f"目标方向角: {np.degrees(direction):.2f}°")

    # 测试特征提取
    print("\n=== 测试特征提取 ===")
    state_features = perception.extract_state_features(depth_image, direction, distance)
    print(f"状态特征形状: {state_features.shape}")

    # 测试CIoU损失
    print("\n=== 测试CIoU损失 ===")
    pred_boxes = torch.rand(10, 4)
    pred_boxes[:, 2:] += pred_boxes[:, :2]  # 确保x2>x1, y2>y1
    target_boxes = torch.rand(10, 4)
    target_boxes[:, 2:] += target_boxes[:, :2]
    ciou_loss = calculate_ciou_loss(pred_boxes, target_boxes)
    print(f"CIoU损失: {ciou_loss.item():.4f}")

    print("\n感知系统测试完成!")
