#!/usr/bin/env python3
"""
下载月球DEM数据
数据来源：NASA PDS LOLA 或 USGS Astrogeology
"""

import os
import sys
from pathlib import Path
import urllib.request
import urllib.error
# from tqdm import tqdm  # 可选进度条

# 数据目录
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lunar_dem"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 月球DEM数据源
DEM_SOURCES = {
    "slodem2015_preview": {
        "url": "https://pgda.gsfc.nasa.gov/products/54",
        "description": "SLDEM2015 预览/文档",
        "local_file": "slodem2015_info.html"
    },
    "lola_256ppd": {
        "url": "http://imbrium.mit.edu/DATA/LDEM_1024/",
        "description": "LOLA 256ppd 全球DEM",
        "local_file": "lola_256ppd_link.html"
    },
    "usgs_moon": {
        "url": "https://astrogeology.usgs.gov/search/map/Moon/MOLA/LOLAKaguya_DEMmerge_60N60S_1024",
        "description": "USGS LOLA+Kaguya合并DEM",
        "local_file": "usgs_dem_link.html"
    }
}


def download_file(url, local_path, timeout=30):
    """
    下载文件，带进度条
    
    Args:
        url: 下载链接
        local_path: 本地保存路径
        timeout: 超时时间
        
    Returns:
        bool: 是否成功
    """
    try:
        print(f"正在下载: {url}")
        print(f"保存到: {local_path}")
        
        # 创建请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        }
        request = urllib.request.Request(url, headers=headers)
        
        # 打开连接
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            # 读取数据
            if total_size > 0:
                chunk_size = 8192
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\r进度: {progress:.1f}%", end='', flush=True)
                print()  # 换行
            else:
                # 没有内容长度信息，直接下载
                data = response.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
                print(f"已下载: {len(data)} bytes")
        
        print(f"✓ 下载成功: {local_path}")
        return True
        
    except urllib.error.URLError as e:
        print(f"✗ URL错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("月球DEM数据下载工具")
    print("=" * 60)
    print(f"\n数据将保存到: {DATA_DIR}")
    print()
    
    # 检查可用数据源
    print("可用的数据源:")
    for i, (key, info) in enumerate(DEM_SOURCES.items(), 1):
        print(f"  {i}. {key}")
        print(f"     {info['description']}")
        print(f"     URL: {info['url']}")
        print()
    
    # 尝试下载数据源信息页面
    print("尝试获取数据源信息...")
    print("-" * 60)
    
    success_count = 0
    for key, info in DEM_SOURCES.items():
        local_path = DATA_DIR / info['local_file']
        if download_file(info['url'], local_path):
            success_count += 1
        print()
    
    print("-" * 60)
    print(f"完成: {success_count}/{len(DEM_SOURCES)} 个数据源信息已获取")
    print("=" * 60)
    
    # 打印手动下载指南
    print("\n手动下载指南:")
    print("由于网络限制，建议手动下载以下数据:")
    print()
    print("1. SLDEM2015 (推荐)")
    print("   网址: http://imbrium.mit.edu/DATA/SLDEM2015/")
    print("   分辨率: 512 pixels/degree (~60m/像素)")
    print("   覆盖范围: ±60° 纬度")
    print()
    print("2. LOLA 1024ppd DEM")
    print("   网址: http://imbrium.mit.edu/DATA/LDEM_1024/")
    print("   分辨率: 1024 pixels/degree (~30m/像素)")
    print("   格式: IMG 或 JP2")
    print()
    print("3. USGS Astrogeology")
    print("   网址: https://astrogeology.usgs.gov/search")
    print("   搜索: Moon LOLA Kaguya DEM")
    print()
    print(f"下载后请将数据文件放入: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
