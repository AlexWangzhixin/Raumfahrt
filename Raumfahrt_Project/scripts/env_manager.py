#!/usr/bin/env python3
"""
环境管理工具
用于检查和管理Raumfahrt项目的Python依赖
"""

import sys
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def get_package_version(package_name):
    """获取包版本"""
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'unknown')
    except ImportError:
        return None


def check_environment():
    """检查当前环境状态"""
    print("=" * 60)
    print("Raumfahrt 项目环境检查")
    print("=" * 60)
    
    # Python版本
    print(f"\nPython版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 核心依赖
    dependencies = {
        'numpy': '科学计算',
        'scipy': '科学计算 (高斯滤波等)',
        'matplotlib': '绘图',
        'PIL': '图像处理 (Pillow)',
        'yaml': 'YAML配置 (PyYAML)',
    }
    
    print("\n核心依赖状态:")
    print("-" * 60)
    
    all_installed = True
    for pkg, desc in dependencies.items():
        version = get_package_version(pkg)
        if version:
            print(f"  [OK] {pkg:15s} {version:10s} ({desc})")
        else:
            print(f"  [MISSING] {pkg:15s} {'not installed':10s} ({desc})")
            all_installed = False
    
    print("-" * 60)
    
    if all_installed:
        print("\n[OK] All core dependencies installed!")
    else:
        print("\n[WARNING] Some dependencies missing, run: pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_installed


def install_requirements():
    """安装requirements.txt中的所有依赖"""
    print("=" * 60)
    print("安装项目依赖")
    print("=" * 60)
    
    if not REQUIREMENTS_FILE.exists():
        print(f"错误: 找不到 {REQUIREMENTS_FILE}")
        return False
    
    print(f"\n从 {REQUIREMENTS_FILE} 安装依赖...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)
        ])
        print("\n[OK] 依赖安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 安装失败: {e}")
        return False


def install_package(package_name):
    """安装单个包"""
    print(f"\n安装 {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name
        ])
        print(f"[OK] {package_name} 安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {package_name} 安装失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Raumfahrt项目环境管理工具')
    parser.add_argument('action', choices=['check', 'install', 'install-scipy'], 
                       help='操作: check(检查环境), install(安装所有依赖), install-scipy(安装scipy)')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_environment()
    elif args.action == 'install':
        install_requirements()
    elif args.action == 'install-scipy':
        install_package('scipy')


if __name__ == "__main__":
    main()
