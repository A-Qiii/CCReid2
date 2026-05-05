"""
visualize_training.py
读取 TensorBoard 日志，生成可直接放入毕业论文的高质量 PNG 图表。
每张图的右下角会显示当前实验的关键配置，便于排查和对比实验。

使用方法：
    python visualize_training.py --log_dir ./logs/pilot_B1 --output_dir ./logs/pilot_B1/vis
"""
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
    'stage1': '#2196F3',
    'stage2': '#FF5722',
    'pos':    '#4CAF50',
    'neg':    '#F44336',
    'acc':    '#9C27B0',
}


def load_scalars(log_dir):
    """
    只加载最新的一个 tfevents 文件，避免多次运行导致曲线叠加。
    同时打印文件信息，方便确认读取的是正确的实验数据。
    """
    if not os.path.exists(log_dir):
        return {}

    import glob
    tfevents_files = sorted(
        glob.glob(os.path.join(log_dir, 'events.out.tfevents.*')),
        key=os.path.getmtime
    )
    if not tfevents_files:
        print(f"  [警告] {log_dir} 下没有找到任何 tfevents 文件")
        return {}

    # 列出所有找到的文件，方便排查
    print(f"  [文件列表] 共找到 {len(tfevents_files)} 个 tfevents 文件:")
    for f in tfevents_files:
        import time
        mtime = time.strftime('%Y-%m-%d %H:%M:%S',
                               time.localtime(os.path.getmtime(f)))
        size_kb = os.path.getsize(f) // 1024
        print(f"    {'→ 使用' if f == tfevents_files[-1] else '  忽略'} "
              f"{os.path.basename(f)}  [{mtime}]  {size_kb}KB")

    latest_file = tfevents_files[-1]

    try:
        ea = EventAccumulator(latest_file, size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            data[tag] = {
                'step':  np.array([e.step for e in events]),
                'value': np.array([e.value for e in events])
            }
        return data
    except Exception as e:
        print(f"  [错误] 读取 tfevents 失败: {e}")
        return {}


def load_text_config(log_dir):
    """从 TensorBoard 日志中读取 add_text 写入的配置信息（只读最新文件）"""
    if not os.path.exists(log_dir):
        return {}
    import glob
    tfevents_files = sorted(
        glob.glob(os.path.join(log_dir, 'events.out.tfevents.*')),
        key=os.path.getmtime
    )
    if not tfevents_files:
        return {}
    ea = EventAccumulator(tfevents_files[-1], size_guidance={'tensors': 0})
    ea.Reload()
    config = {}
    try:
        text_tags = ea.Tags().get('tensors', [])
        for tag in text_tags:
            if tag.startswith('config/'):
                events = ea.Tensors(tag)
                if events:
                    # TensorBoard 文本存在 tensor 里，取最新一条
                    val = events[-1].tensor_proto.string_val
                    if val:
                        key = tag.replace('config/', '')
                        config[key] = val[0].decode('utf-8')
    except Exception:
        pass
    return config


def make_config_label(config_s1, config_s2, exp_name):
    """生成显示在图上的配置文字标签"""
    lines = [f"Exp: {exp_name}"]
    if config_s2.get('ortho_weight'):
        lines.append(f"L_de weight: {config_s2['ortho_weight']}")
    if config_s2.get('sc_weight'):
        lines.append(f"L_sc weight: {config_s2['sc_weight']}")
    if config_s1.get('stage1_lr'):
        lines.append(f"Stage1 LR: {config_s1['stage1_lr']}")
    if config_s2.get('stage2_lr'):
        lines.append(f"Stage2 LR: {config_s2['stage2_lr']}")
    return "\n".join(lines)


def add_config_watermark(ax, label, fontsize=8):
    """在子图右下角添加配置水印"""
    ax.text(0.99, 0.02, label,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8),
            family='monospace')


def smooth(values, weight=0.6):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


# ============================================================

def plot_stage1(data_s1, output_dir, config_label):
    if 'Train/Stage1_Loss' not in data_s1:
        print("[跳过] 未找到 Stage1_Loss 数据")
        return
    steps  = data_s1['Train/Stage1_Loss']['step']
    values = data_s1['Train/Stage1_Loss']['value']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, color=COLORS['stage1'], alpha=0.3, linewidth=1, label='Raw Loss')
    ax.plot(steps, smooth(values), color=COLORS['stage1'], linewidth=2.5, label='Smoothed Loss')

    min_idx = np.argmin(smooth(values))
    ax.annotate(f'Min: {smooth(values)[min_idx]:.3f}',
                xy=(steps[min_idx], smooth(values)[min_idx]),
                xytext=(steps[min_idx], smooth(values)[min_idx] + 0.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('InfoNCE Loss')
    ax.set_title('Stage 1: Hybrid Prompt Learning - InfoNCE Loss Curve')
    ax.legend()
    add_config_watermark(ax, config_label)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_stage1_infonce_loss.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {path}")


def plot_stage1_cosine_probe(data_s1, output_dir, config_label):
    pos_key = 'Train/Stage1_PosSim'
    neg_key = 'Train/Stage1_NegSim'
    if pos_key not in data_s1 or neg_key not in data_s1:
        print("[跳过] 未找到余弦相似度探针数据")
        return

    steps    = data_s1[pos_key]['step']
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
    add_config_watermark(ax, config_label)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_stage1_cosine_probe.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {path}")


def plot_stage2(data_s2, output_dir, config_label):
    keys_map = {
        'Train/Stage2_TotalLoss': ('Total Loss',                    '#212121'),
        'Train/Stage2_L_ce':      ('L_ce (Classification)',         '#2196F3'),
        'Train/Stage2_L_tri':     ('L_tri (Triplet)',               '#4CAF50'),
        'Train/Stage2_L_sc':      ('L_sc (Semantic Align)',         '#FF9800'),
        'Train/Stage2_L_de':      ('L_de (Ortho Decoupling)',       '#F44336'),
        'Train/Stage2_L_Guide':   ('L_Guide (ID Anchor)',           '#9C27B0'),
    }
    available = {k: v for k, v in keys_map.items() if k in data_s2}
    if not available:
        print("[跳过] 未找到 Stage2 损失数据")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, (key, (label, color)) in enumerate(available.items()):
        if idx >= len(axes): break
        ax = axes[idx]
        steps  = data_s2[key]['step']
        values = data_s2[key]['value']
        ax.plot(steps, values, color=color, alpha=0.25, linewidth=1)
        ax.plot(steps, smooth(values), color=color, linewidth=2.5, label=label)
        ax.set_title(label)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss Value')
        ax.legend(loc='upper right', fontsize=9)
        add_config_watermark(ax, config_label, fontsize=7)

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Stage 2: MIPL Disentanglement Loss Components', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_stage2_loss_components.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {path}")


def plot_stage2_acc(data_s2, output_dir, config_label):
    if 'Train/Stage2_BaseAcc' not in data_s2:
        print("[跳过] 未找到 Stage2_BaseAcc 数据")
        return
    steps  = data_s2['Train/Stage2_BaseAcc']['step']
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
    add_config_watermark(ax, config_label)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_stage2_train_accuracy.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {path}")


def plot_two_stage_overview(data_s1, data_s2, output_dir, config_label):
    has_s1 = 'Train/Stage1_Loss' in data_s1
    has_s2 = 'Train/Stage2_TotalLoss' in data_s2
    if not has_s1 and not has_s2:
        print("[跳过] Stage1/Stage2 数据均不存在")
        return

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    if has_s1:
        ax1 = fig.add_subplot(gs[0])
        s = data_s1['Train/Stage1_Loss']
        ax1.plot(s['step'], smooth(s['value']), color=COLORS['stage1'], linewidth=2.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('InfoNCE Loss')
        ax1.set_title('Stage 1: Prompt Learning\n(InfoNCE Convergence)', fontweight='bold')
        ax1.text(0.05, 0.95, '❄ Image/Text Encoder Frozen\n✅ [X] Prompt Learnable',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        add_config_watermark(ax1, config_label)

    if has_s2:
        ax2 = fig.add_subplot(gs[1])
        s = data_s2['Train/Stage2_TotalLoss']
        ax2.plot(s['step'], smooth(s['value']), color=COLORS['stage2'], linewidth=2.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total Loss')
        ax2.set_title('Stage 2: MIPL Disentanglement\n(Visual Finetuning)', fontweight='bold')
        ax2.text(0.05, 0.95, '❄ Text Encoder Frozen\n✅ ViT + Cloth_Proj Learnable',
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.5))
        add_config_watermark(ax2, config_label)

    fig.suptitle('Two-Stage CC-ReID Training Overview', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_two_stage_overview.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {path}")


def plot_decoupling_health(data_s2, output_dir, config_label):
    sc_key = 'Train/Stage2_L_sc'
    de_key = 'Train/Stage2_L_de'
    if sc_key not in data_s2 or de_key not in data_s2:
        print("[跳过] 未找到 L_sc / L_de 数据")
        return

    steps_sc = data_s2[sc_key]['step']
    vals_sc  = data_s2[sc_key]['value']
    steps_de = data_s2[de_key]['step']
    vals_de  = data_s2[de_key]['value']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps_sc, smooth(vals_sc), color='#FF9800', linewidth=2.5)
    ax1.set_title('L_sc: Semantic Alignment Loss\n(Clothing Feature Consistency)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.text(0.6, 0.85, '↓ means cloth_proj\naligns cloth text',
             transform=ax1.transAxes, fontsize=9, color='#FF9800',
             bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))
    add_config_watermark(ax1, config_label)

    ax2.plot(steps_de, smooth(vals_de), color='#F44336', linewidth=2.5)
    ax2.set_title('L_de: Orthogonal Decoupling Loss\n(Clothing Suppression Intensity)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('ReLU(cos(F_ori, F_img2clo))')
    ax2.set_ylim(bottom=0)
    ax2.text(0.6, 0.85, '↓ means clothing info\nremoved from F_ori',
             transform=ax2.transAxes, fontsize=9, color='#F44336',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))
    add_config_watermark(ax2, config_label)

    fig.suptitle('MIPL Surgical Decoupling Health Monitor', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_decoupling_health.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {path}")


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CC-ReID 训练可视化工具")
    parser.add_argument('--log_dir',    type=str, default='./logs/exp_two_stage')
    parser.add_argument('--output_dir', type=str, default='./vis_output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.normpath(args.log_dir))

    s1_log_dir = os.path.join(args.log_dir, 'tensorboard', 'stage1')
    s2_log_dir = os.path.join(args.log_dir, 'tensorboard', 'stage2')

    print(f">>> 正在加载 Stage 1 日志: {s1_log_dir}")
    data_s1   = load_scalars(s1_log_dir)
    config_s1 = load_text_config(s1_log_dir)
    print(f"    找到 {len(data_s1)} 个标量")

    print(f">>> 正在加载 Stage 2 日志: {s2_log_dir}")
    data_s2   = load_scalars(s2_log_dir)
    config_s2 = load_text_config(s2_log_dir)
    print(f"    找到 {len(data_s2)} 个标量")

    # 生成配置水印文字
    config_label = make_config_label(config_s1, config_s2, exp_name)
    print(f"\n>>> 配置水印内容:\n{config_label}\n")

    print(">>> 开始生成可视化图表...")
    plot_stage1(data_s1, args.output_dir, config_label)
    plot_stage1_cosine_probe(data_s1, args.output_dir, config_label)
    plot_stage2(data_s2, args.output_dir, config_label)
    plot_stage2_acc(data_s2, args.output_dir, config_label)
    plot_two_stage_overview(data_s1, data_s2, args.output_dir, config_label)
    plot_decoupling_health(data_s2, args.output_dir, config_label)

    print(f"\n✅ 全部图表已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()