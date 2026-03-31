"""
生成推文配图 — 专业 matplotlib 图表，基于真实实验数据。
风格参考 Ollama benchmark：简洁、粗体数字、高对比度。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# 全局样式：干净、专业
plt.rcParams.update({
    'figure.facecolor': '#f5f5f5',
    'axes.facecolor': '#f5f5f5',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
})


def chart1_lora_rank():
    """LoRA Rank vs Online Learning — 核心发现图"""
    ranks = ['8', '16', '32', '64', '128', 'Full\n(512)']
    slopes = [-0.0364, -0.0360, -0.0363, -0.0362, -0.0356, -0.0045]
    abs_slopes = [abs(s) for s in slopes]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1a1a1a'] * 5 + ['#bbbbbb']
    bars = ax.bar(ranks, abs_slopes, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

    # 数字标注
    for i, (bar, slope) in enumerate(zip(bars, slopes)):
        y = bar.get_height()
        label = f'{slope:.4f}'
        fontweight = 'bold' if i == 0 else 'normal'
        fontsize = 16 if i == 0 else 13
        color = '#1a1a1a' if i < 5 else '#888888'
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.0005,
                label, ha='center', va='bottom', fontsize=fontsize,
                fontweight=fontweight, color=color)

    # rank=8 加 8.1x 标注
    ax.annotate('8.1× better', xy=(0, abs_slopes[0]), xytext=(1.5, 0.042),
                fontsize=15, fontweight='bold', color='#d63031',
                arrowprops=dict(arrowstyle='->', color='#d63031', lw=2),
                ha='center')

    ax.set_xlabel('LoRA Rank', fontweight='bold')
    ax.set_ylabel('Online Learning Slope (|slope|, higher = better)', fontweight='bold')
    ax.set_title('Low-Rank Regularization in TTT:\nLess Capacity = Better Learning')
    ax.set_ylim(0, 0.048)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'chart_lora_rank.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


def chart2_memory():
    """100M Context 显存对比 — 标准 Transformer vs NEXUS"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Standard\nTransformer', 'NEXUS\n(1B model)']
    memory = [16700, 2.6]  # GB

    colors = ['#bbbbbb', '#1a1a1a']
    bars = ax.bar(models, memory, color=colors, width=0.5, edgecolor='white', linewidth=2)

    # 数字标注
    ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 200,
            '16,700 GB', ha='center', va='bottom', fontsize=20, fontweight='bold', color='#888888')
    ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() * 0.5,
            '120× A100', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

    ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 200,
            '2.6 GB', ha='center', va='bottom', fontsize=20, fontweight='bold', color='#1a1a1a')
    ax.text(bars[1].get_x() + bars[1].get_width()/2, max(bars[1].get_height(), 800),
            '1× RTX 4060', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

    # 6400x 标注
    ax.annotate('6,400× less memory', xy=(1, 2.6), xytext=(0.5, 10000),
                fontsize=16, fontweight='bold', color='#d63031',
                arrowprops=dict(arrowstyle='->', color='#d63031', lw=2.5),
                ha='center')

    ax.set_ylabel('KV Cache Memory for 100M Context (GB)', fontweight='bold')
    ax.set_title('100M Token Context: Memory Comparison')
    ax.set_ylim(0, 20000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'chart_memory.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


def chart3_ttt_online():
    """TTT 在线学习：序列越长 Loss 越低"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：NEXUS TTT 在线学习（斜率为负）
    quarters = ['Q1\n(0-256)', 'Q2\n(256-512)', 'Q3\n(512-768)', 'Q4\n(768-1024)']
    nexus_losses = [7.8958, 7.8089, 7.8092, 7.8005]

    ax1.bar(quarters, nexus_losses, color='#1a1a1a', width=0.6, edgecolor='white', linewidth=1.5)
    for i, (q, l) in enumerate(zip(quarters, nexus_losses)):
        ax1.text(i, l + 0.002, f'{l:.3f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold' if i == 3 else 'normal')

    # 趋势线
    x = np.arange(4)
    z = np.polyfit(x, nexus_losses, 1)
    ax1.plot(x, np.polyval(z, x), '--', color='#d63031', linewidth=2.5, label=f'slope = {z[0]:+.4f}')
    ax1.legend(fontsize=13, loc='upper right')

    ax1.set_title('NEXUS: Loss ↓ while reading', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Cross-Entropy Loss', fontweight='bold')
    ax1.set_ylim(7.78, 7.92)
    ax1.set_xlabel('Sequence Position', fontweight='bold')

    # 右图：对比柱状图 — 斜率对比
    models = ['NEXUS\n(TTT)', 'Standard\nTransformer']
    slopes = [-0.029, +0.001]
    colors_bar = ['#1a1a1a', '#bbbbbb']

    bars2 = ax1_bars = ax2.bar(models, slopes, color=colors_bar, width=0.5, edgecolor='white', linewidth=2)
    for bar, s in zip(bars2, slopes):
        y_pos = s - 0.003 if s < 0 else s + 0.002
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{s:+.3f}', ha='center', va='top' if s < 0 else 'bottom',
                fontsize=18, fontweight='bold',
                color='#1a1a1a' if s < 0 else '#888888')

    ax2.axhline(y=0, color='#333333', linewidth=1)
    ax2.set_title('Online Learning Slope\n(negative = learns)', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Loss Slope', fontweight='bold')
    ax2.set_ylim(-0.045, 0.015)

    # 标注
    ax2.text(0, -0.040, '✅ Adapts to context', ha='center', fontsize=12, color='#27ae60', fontweight='bold')
    ax2.text(1, 0.010, '❌ Static', ha='center', fontsize=12, color='#e74c3c', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'chart_ttt_online.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


def chart4_architecture():
    """架构总览信息图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'NEXUS Architecture', fontsize=24, fontweight='bold',
            ha='center', va='center', color='#1a1a1a')
    ax.text(5, 9.0, 'Next-gen Transformer · 6 Frontier Techniques · 1 Unified Model',
            fontsize=13, ha='center', va='center', color='#666666')

    # 组件框
    components = [
        (5, 7.5, 'DiffAttn + MLA', 'Noise cancellation + KV compression 4×', '#2c3e50'),
        (5, 6.0, 'TTT-Linear (LoRA rank=8)', 'Online learning during inference · 8× better', '#e74c3c'),
        (5, 4.5, 'MoE-SwiGLU (8 experts, top-2)', 'Sparse activation · 32% compute per token', '#27ae60'),
    ]

    for x, y, title, sub, color in components:
        box = plt.Rectangle((1, y - 0.55), 8, 1.1, facecolor=color, alpha=0.12,
                            edgecolor=color, linewidth=2, transform=ax.transData)
        ax.add_patch(box)
        ax.text(x, y + 0.15, title, fontsize=15, fontweight='bold', ha='center', color=color)
        ax.text(x, y - 0.25, sub, fontsize=11, ha='center', color='#555555')

        # 残差箭头
        if y > 5:
            ax.annotate('', xy=(9.3, y - 0.55), xytext=(9.3, y + 0.55),
                       arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5))

    # 底部统计
    stats = [
        ('1B Model', '2.6 GB VRAM'),
        ('100M Context', 'Sliding Window'),
        ('LoRA rank=8', '8× Online Learning'),
    ]
    for i, (title, value) in enumerate(stats):
        x = 1.7 + i * 3
        box = plt.Rectangle((x - 1.2, 2.2), 2.8, 1.2, facecolor='#1a1a1a', alpha=0.9,
                            edgecolor='none', transform=ax.transData)
        ax.add_patch(box)
        ax.text(x + 0.2, 3.0, title, fontsize=13, fontweight='bold', ha='center', color='white')
        ax.text(x + 0.2, 2.5, value, fontsize=11, ha='center', color='#cccccc')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'chart_architecture.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


if __name__ == '__main__':
    print('生成推文配图...')
    chart1_lora_rank()
    chart2_memory()
    chart3_ttt_online()
    chart4_architecture()
    print('\n全部完成！图片在 assets/ 目录下')
