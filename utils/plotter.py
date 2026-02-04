import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
from matplotlib import font_manager


# --- 解决 Matplotlib 中文乱码的核心逻辑 ---
def configure_chinese_font():
    # 优先尝试的字体列表 (Windows, Mac, Linux)
    font_candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC', 'WenQuanYi Micro Hei']

    # 获取系统已安装的字体
    system_fonts = {f.name for f in font_manager.fontManager.ttflist}

    selected_font = None
    for font in font_candidates:
        if font in system_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        # print(f"✅ Plotter 已加载中文字体: {selected_font}")
    else:
        # 如果实在找不到，就回退，或者你可以指定一个 ttf 文件的绝对路径
        print("⚠️ Plotter 未检测到常见中文字体，雷达图中文可能显示为方框。")


# 初始化配置
configure_chinese_font()
# 使用非交互式后端，防止多线程报错
matplotlib.use('Agg')


def create_radar_chart(scores, categories, save_dir="static"):
    """
    生成雷达图并返回本地路径
    :param scores: list [80, 60, 90, 70, 85]
    :param categories: list ['技术', '经验', ...]
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据闭环 (Matplotlib雷达图特性，首尾相连)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    scores_loop = scores + [scores[0]]
    angles_loop = angles + [angles[0]]
    categories_loop = categories + [categories[0]]

    # 2. 绘图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制线条和填充
    ax.plot(angles_loop, scores_loop, linewidth=2, linestyle='solid', color='#3b82f6')
    ax.fill(angles_loop, scores_loop, '#3b82f6', alpha=0.25)

    # 设置标签
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)

    # 设置刻度范围
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([])  # 隐藏具体数字，保持清爽

    # 标题
    plt.title("人岗匹配度多维分析", y=1.08, fontsize=14, fontweight='bold', color='#1e293b')

    # 3. 保存文件
    filename = f"radar_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)  # 关键：释放内存

    return filepath