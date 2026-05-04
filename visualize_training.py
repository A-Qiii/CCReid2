"""
visualize_training.py
====================
读取 TensorBoard .tfevents 日志，生成可直接放入毕业论文的高质量 PNG 图表。

使用方法：
    python visualize_training.py --log_dir ./logs/exp_two_stage --output_dir ./vis_output

依赖：
    pip install tensorboard matplotlib numpy
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 服务器环境必须设置
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==================== 全局绘图风格（论文级） ====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

COLORS = {
    'stage1': '#2196F3',   # 蓝色
    'stage2': '#FF5722',   # 橙红色
    'pos':    '#4CAF50',   # 绿色（正样本相似度）
    'neg':    '#F44336',   # 红色（负样本相似度）
    'acc':    '#9C27B0',   # 紫色
}


def load_scalars(log_dir):
    """加载指定目录下所有 tfevents 文件中的标量数据"""
    if not os.path.exists(log_dir):
        return {}
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            'step': np.array([e.step for e in events]),
            'value': np.array([e.value for e in events])
        }
    return data


def smooth(values, weight=0.6):
    """指数移动平均平滑，用于减少训练曲线锯齿"""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


def plot_stage1(data_s1, output_dir):
    """
    图1：Stage 1 InfoNCE 损失曲线
    用途：验证提示词 [X] 是否正常收敛，是毕业论文中展示 Stage1 有效性的核心图
    """
    if 'Train/Stage1_Loss' not in data_s1:
        print("[跳过] 未找到 Stage1_Loss 数据")
        return

    steps = data_s1['Train/Stage1_Loss']['step']
    values = data_s1['Train/Stage1_Loss']['value']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, color=COLORS['stage1'], alpha=0.3, linewidth=1, label='Raw Loss')
    ax.plot(steps, smooth(values), color=COLORS['stage1'], linewidth=2.5, label='Smoothed Loss')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('InfoNCE Loss')
    ax.set_title('Stage 1: Hybrid Prompt Learning - InfoNCE Loss Curve')
    ax.legend()

    # 标注最小损失点
    min_idx = np.argmin(smooth(values))
    ax.annotate(f'Min: {smooth(values)[min_idx]:.3f}',
                xy=(steps[min_idx], smooth(values)[min_idx]),
                xytext=(steps[min_idx], smooth(values)[min_idx] + 0.3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_stage1_infonce_loss.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {save_path}")


def plot_stage1_cosine_probe(data_s1, output_dir):
    """
    图2：Stage 1 余弦相似度探针（正负样本分离程度）
    用途：直观验证 [X] 提示词是否真正实现了同身份特征拉近、异身份特征推开
    这是论文中论证创新点有效性的关键定性图
    """
    pos_key = 'Train/Stage1_PosSim'
    neg_key = 'Train/Stage1_NegSim'

    if pos_key not in data_s1 or neg_key not in data_s1:
        print("[跳过] 未找到余弦相似度探针数据（需在 loss_func 中额外写入 TensorBoard）")
        return

    steps = data_s1[pos_key]['step']
    pos_vals = data_s1[pos_key]['value']
    neg_vals = data_s1[neg_key]['value']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, smooth(pos_vals, 0.7), color=COLORS['pos'], linewidth=2.5,
            label='Positive Pair Cosine Sim (↑ target: 1.0)')
    ax.plot(steps, smooth(neg_vals, 0.7), color=COLORS['neg'], linewidth=2.5,
            label='Negative Pair Cosine Sim (↓ target: 0.0)')
    ax.fill_between(steps, smooth(pos_vals, 0.7), smooth(neg_vals, 0.7),
                    alpha=0.1, color='green', label='Margin Gap')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Stage 1: Text-Image Alignment Quality (Cosine Probe)')
    ax.set_ylim([-0.1, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig2_stage1_cosine_probe.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {save_path}")


def plot_stage2(data_s2, output_dir):
    """
    图3：Stage 2 各损失分量曲线（四合一）
    用途：展示解耦训练过程中各损失组件的变化，验证 MIPL 外科手术效果
    """
    keys_map = {
        'Train/Stage2_TotalLoss': ('Total Loss', '#212121'),
        'Train/Stage2_L_ce':      ('L_ce (Classification)', '#2196F3'),
        'Train/Stage2_L_tri':     ('L_tri (Triplet)',        '#4CAF50'),
        'Train/Stage2_L_sc':      ('L_sc (Semantic Align)', '#FF9800'),
        'Train/Stage2_L_de':      ('L_de (Ortho Decoupling)', '#F44336'),
        'Train/Stage2_L_Guide':   ('L_Guide (ID Anchor)',    '#9C27B0'),
    }

    available = {k: v for k, v in keys_map.items() if k in data_s2}
    if not available:
        print("[跳过] 未找到 Stage2 损失数据")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (key, (label, color)) in enumerate(available.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        steps = data_s2[key]['step']
        values = data_s2[key]['value']
        ax.plot(steps, values, color=color, alpha=0.25, linewidth=1)
        ax.plot(steps, smooth(values), color=color, linewidth=2.5, label=label)
        ax.set_title(label)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss Value')
        ax.legend(loc='upper right', fontsize=9)

    # 隐藏多余的子图
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Stage 2: MIPL Disentanglement Loss Components', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig3_stage2_loss_components.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {save_path}")


def plot_stage2_acc(data_s2, output_dir):
    """
    图4：Stage 2 训练分类准确率曲线
    用途：监控 Stage 2 基础判别能力是否正常，排查过拟合/欠拟合
    """
    if 'Train/Stage2_BaseAcc' not in data_s2:
        print("[跳过] 未找到 Stage2_BaseAcc 数据")
        return

    steps = data_s2['Train/Stage2_BaseAcc']['step']
    values = data_s2['Train/Stage2_BaseAcc']['value']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, color=COLORS['acc'], alpha=0.3, linewidth=1)
    ax.plot(steps, smooth(values, 0.8), color=COLORS['acc'], linewidth=2.5, label='Train Accuracy')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Stage 2: Training Classification Accuracy')
    ax.set_ylim([0, 1.05])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig4_stage2_train_accuracy.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {save_path}")


def plot_two_stage_overview(data_s1, data_s2, output_dir):
    """
    图5：两阶段训练总览图（横向拼接）
    用途：毕业论文中一张图展示完整训练流程，高度概括实验设置
    """
    has_s1 = 'Train/Stage1_Loss' in data_s1
    has_s2 = 'Train/Stage2_TotalLoss' in data_s2

    if not has_s1 and not has_s2:
        print("[跳过] Stage1/Stage2 数据均不存在")
        return

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    if has_s1:
        ax1 = fig.add_subplot(gs[0])
        s1_steps = data_s1['Train/Stage1_Loss']['step']
        s1_vals = data_s1['Train/Stage1_Loss']['value']
        ax1.plot(s1_steps, smooth(s1_vals), color=COLORS['stage1'], linewidth=2.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('InfoNCE Loss')
        ax1.set_title('Stage 1: Prompt Learning\n(InfoNCE Convergence)', fontweight='bold')
        ax1.text(0.05, 0.95, '❄ Image/Text Encoder Frozen\n✅ [X] Prompt Learnable',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    if has_s2:
        ax2 = fig.add_subplot(gs[1])
        s2_steps = data_s2['Train/Stage2_TotalLoss']['step']
        s2_vals = data_s2['Train/Stage2_TotalLoss']['value']
        ax2.plot(s2_steps, smooth(s2_vals), color=COLORS['stage2'], linewidth=2.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total Loss')
        ax2.set_title('Stage 2: MIPL Disentanglement\n(Visual Finetuning)', fontweight='bold')
        ax2.text(0.05, 0.95, '❄ Text Encoder Frozen\n✅ ViT + Cloth_Proj Learnable',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.5))

    fig.suptitle('Two-Stage CC-ReID Training Overview', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig5_two_stage_overview.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {save_path}")


def plot_decoupling_health(data_s2, output_dir):
    """
    图6：解耦健康度监控（L_sc 与 L_de 联合图）
    用途：直观展示 MIPL 外科手术是否真正在工作
    - L_sc 下降 → 衣服投影器 cloth_proj 对齐文本衣服特征，提取纯净
    - L_de 下降 → 视觉特征与衣服特征的余弦相似度降低，正交解耦成功
    """
    sc_key = 'Train/Stage2_L_sc'
    de_key = 'Train/Stage2_L_de'

    if sc_key not in data_s2 or de_key not in data_s2:
        print("[跳过] 未找到 L_sc / L_de 数据（需在 make_loss.py 中写入 TensorBoard）")
        return

    steps_sc = data_s2[sc_key]['step']
    vals_sc = data_s2[sc_key]['value']
    steps_de = data_s2[de_key]['step']
    vals_de = data_s2[de_key]['value']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps_sc, smooth(vals_sc), color='#FF9800', linewidth=2.5)
    ax1.set_title('L_sc: Semantic Alignment Loss\n(Clothing Feature Consistency)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.text(0.6, 0.85, '↓ means cloth_proj\naligns cloth text',
             transform=ax1.transAxes, fontsize=9, color='#FF9800',
             bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))

    ax2.plot(steps_de, smooth(vals_de), color='#F44336', linewidth=2.5)
    ax2.set_title('L_de: Orthogonal Decoupling Loss\n(Clothing Suppression Intensity)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('ReLU(cos(F_ori, F_img2clo))')
    ax2.set_ylim(bottom=0)
    ax2.text(0.6, 0.85, '↓ means clothing info\nremoved from F_ori',
             transform=ax2.transAxes, fontsize=9, color='#F44336',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

    fig.suptitle('MIPL Surgical Decoupling Health Monitor', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig6_decoupling_health.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {save_path}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="CC-ReID 训练可视化工具")
    parser.add_argument('--log_dir', type=str, default='./logs/exp_two_stage',
                        help='TensorBoard 日志根目录（含 tensorboard/stage1 和 tensorboard/stage2 子目录）')
    parser.add_argument('--output_dir', type=str, default='./vis_output',
                        help='可视化图片输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载两个阶段的日志
    s1_log_dir = os.path.join(args.log_dir, 'tensorboard', 'stage1')
    s2_log_dir = os.path.join(args.log_dir, 'tensorboard', 'stage2')

    print(f">>> 正在加载 Stage 1 日志: {s1_log_dir}")
    data_s1 = load_scalars(s1_log_dir)
    print(f"    找到 {len(data_s1)} 个标量: {list(data_s1.keys())}")

    print(f">>> 正在加载 Stage 2 日志: {s2_log_dir}")
    data_s2 = load_scalars(s2_log_dir)
    print(f"    找到 {len(data_s2)} 个标量: {list(data_s2.keys())}")

    print("\n>>> 开始生成可视化图表...")

    # 依次生成所有图表
    plot_stage1(data_s1, args.output_dir)
    plot_stage1_cosine_probe(data_s1, args.output_dir)
    plot_stage2(data_s2, args.output_dir)
    plot_stage2_acc(data_s2, args.output_dir)
    plot_two_stage_overview(data_s1, data_s2, args.output_dir)
    plot_decoupling_health(data_s2, args.output_dir)

    print(f"\n✅ 全部可视化图表已保存至: {args.output_dir}")
    print("\n图表用途说明：")
    print("  fig1 - Stage1 InfoNCE 损失收敛曲线（论证提示词学习有效性）")
    print("  fig2 - Stage1 正负样本余弦相似度探针（论证特征对齐质量）")
    print("  fig3 - Stage2 各损失分量（论证 MIPL 解耦训练稳定性）")
    print("  fig4 - Stage2 训练分类准确率（监控过拟合/欠拟合）")
    print("  fig5 - 两阶段训练总览（论文中一图展示完整训练流程）")
    print("  fig6 - 解耦健康度监控 L_sc/L_de（论证外科手术解耦是否生效）")


if __name__ == '__main__':
    main()