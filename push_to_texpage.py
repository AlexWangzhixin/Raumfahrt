#!/usr/bin/env python3
"""
推送到TexPage并编译的脚本
创建完整的推送包（包含所有依赖文件）
"""

import os
import glob
import shutil
from pathlib import Path
from datetime import datetime

# TexPage项目URL
TEXPAGE_PROJECT_URL = "https://www.texpage.com/project/user/174b0065-728d-4b2a-b993-a94f361552a8/b452641c-8e0c-4f38-9d16-658b0018f7bd"

# 项目路径
THESIS_DIR = Path("docs/thesis/Thesis_Project")
CHAPTERS_DIR = THESIS_DIR / "chapters"
SETUP_DIR = THESIS_DIR / "setup"
FIGURES_DIR = THESIS_DIR / "figures"

# 输出目录
OUTPUT_DIR = Path("texpage_upload_package")

def read_file_content(filepath):
    """读取文件内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def get_chapter_files():
    """获取第五章和第六章的所有文件"""
    files = []
    
    # 第五章文件
    ch5_dir = CHAPTERS_DIR / "5.基于数字孪生的月面岩石识别与避障"
    ch5_files = sorted(glob.glob(str(ch5_dir / "*.tex")))
    files.extend(ch5_files)
    
    # 第六章文件
    ch6_dir = CHAPTERS_DIR / "6.基于数字孪生的巡视器路径规划"
    ch6_files = sorted(glob.glob(str(ch6_dir / "*.tex")))
    files.extend(ch6_files)
    
    return files

def get_figure_files():
    """获取第五章和第六章的图片文件"""
    figures = []
    # ch5 figures
    ch5_figs = glob.glob(str(FIGURES_DIR / "ch5" / "*"))
    figures.extend([f for f in ch5_figs if os.path.isfile(f)])
    # ch6 figures
    ch6_figs = glob.glob(str(FIGURES_DIR / "ch6" / "*"))
    figures.extend([f for f in ch6_figs if os.path.isfile(f)])
    return figures

def create_upload_package():
    """创建完整的上传包"""
    print("=" * 60)
    print("创建TexPage上传包")
    print("=" * 60)
    
    # 清理并创建输出目录
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # 创建figures子目录
    figures_output = OUTPUT_DIR / "figures"
    figures_output.mkdir(parents=True)
    (figures_output / "ch5").mkdir(parents=True)
    (figures_output / "ch6").mkdir(parents=True)
    
    # 1. 读取setup文件
    print("\n1. 读取setup文件...")
    preamble = read_file_content(SETUP_DIR / "preamble.tex")
    macros = read_file_content(SETUP_DIR / "macros.tex")
    print("   - preamble.tex")
    print("   - macros.tex")
    
    # 2. 读取第五章和第六章内容
    print("\n2. 读取章节文件...")
    ch5_files = sorted(glob.glob(str(CHAPTERS_DIR / "5.基于数字孪生的月面岩石识别与避障" / "*.tex")))
    ch6_files = sorted(glob.glob(str(CHAPTERS_DIR / "6.基于数字孪生的巡视器路径规划" / "*.tex")))
    
    ch5_content = ""
    for f in ch5_files:
        print(f"   - {os.path.basename(f)}")
        ch5_content += f"\n% === {os.path.basename(f)} ===\n"
        ch5_content += read_file_content(f)
    
    ch6_content = ""
    for f in ch6_files:
        print(f"   - {os.path.basename(f)}")
        ch6_content += f"\n% === {os.path.basename(f)} ===\n"
        ch6_content += read_file_content(f)
    
    # 3. 构建完整的文档
    print("\n3. 构建完整的LaTeX文档...")
    full_doc = f"""% !TeX program = lualatex
% !TeX encoding = UTF-8
% 中文论文排版：使用 ctex（支持章节/中文标题/中文标点等）
\\documentclass[UTF8,zihao=-4,a4paper,oneside,openany]{{ctexrep}}

% 关闭快速编译
\\newif\\iffastcompile
\\fastcompilefalse

{preamble}

{macros}

\\title{{面向月面巡视器自主行走的数字孪生技术研究}}
\\author{{作者姓名}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

{ch5_content}

{ch6_content}

\\end{{document}}
"""
    
    # 4. 保存主文件
    main_file = OUTPUT_DIR / "main.tex"
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(full_doc)
    print(f"   主文件已保存: main.tex ({len(full_doc)} 字符)")
    
    # 5. 复制图片文件
    print("\n4. 复制图片文件...")
    fig_files = get_figure_files()
    for fig in fig_files:
        fig_name = os.path.basename(fig)
        # 确定目标目录
        if "ch5" in fig:
            dest_dir = figures_output / "ch5"
        else:
            dest_dir = figures_output / "ch6"
        dest_file = dest_dir / fig_name
        shutil.copy2(fig, dest_file)
        print(f"   - figures/{'ch5' if 'ch5' in fig else 'ch6'}/{fig_name}")
    
    # 6. 创建说明文件
    print("\n5. 创建说明文件...")
    readme_content = f"""# TexPage 上传包

## 创建时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件说明
- main.tex: 主LaTeX文件（包含第五章和第六章）
- figures/: 图片目录
  - ch5/: 第五章图片
  - ch6/: 第六章图片

## 章节内容
### 第五章：基于数字孪生的月面岩石识别与避障
- 5.1_月面岩石识别需求与挑战.tex
- 5.2_数字孪生平台下的岩石识别框架.tex
- 5.3_增强型ORB-SLAM2系统.tex
- 5.4_基于岩石识别的避障策略.tex
- 5.5_算法验证与性能分析.tex

### 第六章：基于数字孪生的巡视器路径规划
- 6.1_路径规划问题描述.tex
- 6.2_数字孪生环境下的路径规划框架.tex
- 6.3_基于A-D3QN的混合路径规划算法.tex
- 6.4_算法验证与性能分析.tex

## 上传说明
1. 登录TexPage: {TEXPAGE_PROJECT_URL}
2. 将main.tex内容复制到项目主文件
3. 上传figures目录下的所有图片到对应路径

## 编译说明
- 编译器: LuaLaTeX
- 需要完整的ctex宏包支持
"""
    
    readme_file = OUTPUT_DIR / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("   - README.md")
    
    # 7. 创建zip包
    print("\n6. 创建zip压缩包...")
    zip_file = f"texpage_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.make_archive(zip_file, 'zip', OUTPUT_DIR)
    print(f"   压缩包: {zip_file}.zip")
    
    # 8. 打印总结
    print("\n" + "=" * 60)
    print("上传包创建完成!")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR.absolute()}")
    print(f"压缩包: {zip_file}.zip")
    print(f"\n请访问: {TEXPAGE_PROJECT_URL}")
    print("上传main.tex文件和figures目录下的图片文件。")
    
    return str(OUTPUT_DIR)

if __name__ == "__main__":
    create_upload_package()