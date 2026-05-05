"""
test_visualize.py
=================
测试阶段可视化脚本，生成三类论文级图表：
  1. 检索结果可视化（Top-K Retrieval）
  2. t-SNE 特征分布图（换装感知版）
     - 颜色 = 身份（同色=同人）
     - 形状 = 衣服（● = Gallery原装, ★ = Query换装）
     - 边框粗细 = Query(粗黑边框) / Gallery(无边框)

使用方法：
    python test_visualize.py \
        --config_file configs/prcc/vit_ccreid_prcc_final.yml \
        --weight_path ./logs/final_exp/stage2_disentangled_model_30.pth \
        --output_dir ./logs/final_exp/vis_test
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from configs import cfg
from datasets import make_dataloader
from datasets.prcc import PRCC
from datasets.ltcc import LTCC
from modeling import make_model
from processor.processor import do_inference
from processor.evaluator import eval_func


# ──────────────────────────────────────────────
def load_model_and_data(cfg, weight_path):
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    ckpt = torch.load(weight_path, map_location='cpu')
    num_clothes_ckpt = (ckpt['prompt_learner.ctx_cloth'].shape[0]
                        if 'prompt_learner.ctx_cloth' in ckpt else 1000)
    num_classes_ckpt = (ckpt['classifier.weight'].shape[0]
                        if 'classifier.weight' in ckpt else num_classes)
    print(f">>> checkpoint 检测: num_classes={num_classes_ckpt}, num_clothes={num_clothes_ckpt}")

    cfg.defrost()
    cfg.MODEL.NUM_CLOTHES = num_clothes_ckpt
    cfg.freeze()

    model = make_model(cfg, num_class=num_classes_ckpt,
                       camera_num=camera_num, view_num=view_num)
    model.load_state_dict(ckpt)
    model.to(cfg.MODEL.DEVICE)
    print(f">>> 权重加载成功: {weight_path}")
    return model, val_loader, num_query, num_classes_ckpt


def get_dataset_info(cfg):
    """获取 query/gallery 的图片路径和衣服ID"""
    if cfg.DATASETS.NAMES == 'prcc':
        dataset = PRCC(root=cfg.DATASETS.ROOT_DIR,
                       llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    elif cfg.DATASETS.NAMES == 'ltcc':
        dataset = LTCC(root=cfg.DATASETS.ROOT_DIR,
                       llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    else:
        raise ValueError(f"不支持的数据集: {cfg.DATASETS.NAMES}")

    q_paths    = [x[0] for x in dataset.query]
    g_paths    = [x[0] for x in dataset.gallery]
    q_clothids = np.asarray([x[3] for x in dataset.query])
    g_clothids = np.asarray([x[3] for x in dataset.gallery])
    return q_paths, g_paths, q_clothids, g_clothids


# ──────────────────────────────────────────────
# 图1：Top-K 检索结果可视化
# ──────────────────────────────────────────────
def plot_retrieval(feats, pids, camids, q_paths, g_paths,
                   num_query, output_dir, num_vis=10, top_k=5):
    print(">>> 正在生成检索结果可视化...")
    qf = feats[:num_query]
    gf = feats[num_query:]
    q_pids   = np.asarray(pids[:num_query])
    g_pids   = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    distmat = torch.cdist(qf, gf).numpy()
    indices = np.argsort(distmat, axis=1)

    np.random.seed(42)
    sampled = np.random.choice(num_query, min(num_vis, num_query), replace=False)

    fig, axes = plt.subplots(num_vis, top_k + 1,
                             figsize=((top_k + 1) * 2.2, num_vis * 2.8))
    if num_vis == 1:
        axes = axes[np.newaxis, :]

    for row, q_idx in enumerate(sampled):
        q_pid   = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order   = indices[q_idx]
        remove  = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        valid_order = order[~remove]

        ax = axes[row, 0]
        try:
            img = Image.open(q_paths[q_idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (64, 128), (200, 200, 200))
        ax.imshow(img)
        ax.set_title(f"Query\nID:{q_pid}", fontsize=8, fontweight='bold')
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor('#2196F3'); sp.set_linewidth(3)

        for col in range(top_k):
            ax = axes[row, col + 1]
            if col >= len(valid_order):
                ax.axis('off'); continue
            g_idx = valid_order[col]
            g_pid = g_pids[g_idx]
            hit   = (g_pid == q_pid)
            try:
                img = Image.open(g_paths[g_idx]).convert('RGB')
            except Exception:
                img = Image.new('RGB', (64, 128), (200, 200, 200))
            color = '#4CAF50' if hit else '#F44336'
            ax.imshow(img)
            ax.set_title(f"{'✓' if hit else '✗'} ID:{g_pid}",
                         fontsize=8, fontweight='bold', color=color)
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(3)

    fig.suptitle(
        f'CC-ReID Top-{top_k} Retrieval  |  '
        f'{cfg.DATASETS.NAMES.upper()}  |  Green=Correct  Red=Wrong',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, 'fig7_retrieval_results.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"[已保存] {path}")


# ──────────────────────────────────────────────
# 图2：换装感知 t-SNE 特征分布可视化
# 颜色=身份  形状=Query(★)/Gallery(●)  边框粗细=是否Query
# ──────────────────────────────────────────────
def plot_tsne_cloth_aware(feats, pids, camids, q_clothids, g_clothids,
                          num_query, output_dir, max_ids=20, max_samples=1000):
    """
    换装感知 t-SNE（对齐你的设计方案）：
    - 颜色  = 身份（同色=同一个人）
    - 形状  = 衣服（每个身份内部按衣服编号：○=衣1  □=衣2  △=衣3  ◇=衣4+）
    - 边框  = Query 有粗黑边框，Gallery 无边框

    健康状态（解耦成功）：
      问题①：同色不同形状是否聚在一起？→ 是 = 换装鲁棒，L_de/L_sc 有效
      问题②：★(Query) 是否落在自己颜色的●团内？→ 是 = 跨摄像头对齐，L_Guide 有效

    问题状态（解耦失败）：
      同色不同形状分散各处 → L_de 过强，破坏身份特征
      ★ 落入其他颜色的团 → 检索会错
    """
    print(">>> 正在计算换装感知 t-SNE（约1-2分钟）...")

    all_feats    = feats.numpy()
    all_pids     = np.asarray(pids)
    is_query     = np.array([True]*num_query + [False]*(len(pids)-num_query))
    all_clothids = np.concatenate([q_clothids, g_clothids])

    unique_pids = np.unique(all_pids)[:max_ids]
    mask = np.isin(all_pids, unique_pids)
    sf, sp, sq, sc = all_feats[mask], all_pids[mask], is_query[mask], all_clothids[mask]

    if len(sf) > max_samples:
        idx = np.random.choice(len(sf), max_samples, replace=False)
        sf, sp, sq, sc = sf[idx], sp[idx], sq[idx], sc[idx]

    coords = TSNE(n_components=2, random_state=42,
                  perplexity=30, max_iter=1000).fit_transform(sf)

    cmap      = plt.cm.get_cmap('tab20', max_ids)
    pid_color = {pid: cmap(i) for i, pid in enumerate(unique_pids)}

    # 为每个身份内部的衣服分配局部编号 → 形状
    # ○圆=衣1  □方=衣2  △三角=衣3  ◇菱形=衣4+
    marker_pool = ['o', 's', '^', 'D', 'p', 'h']

    # 建立 (pid, cloth_id) → 局部cloth编号 的映射
    pid_cloth_map = {}  # {pid: {cloth_id: local_idx}}
    for pid in unique_pids:
        pid_mask   = sp == pid
        pid_cloths = np.unique(sc[pid_mask])
        pid_cloth_map[pid] = {cid: i for i, cid in enumerate(sorted(pid_cloths))}

    fig, ax = plt.subplots(figsize=(12, 10))

    # 先画 Gallery（底层，无边框）
    for pid in unique_pids:
        c = pid_color[pid]
        for cid, local_idx in pid_cloth_map[pid].items():
            marker = marker_pool[min(local_idx, len(marker_pool)-1)]
            gm = (sp == pid) & (sc == cid) & ~sq
            if gm.any():
                ax.scatter(coords[gm, 0], coords[gm, 1],
                           c=[c], marker=marker, s=55, alpha=0.7,
                           edgecolors='none', zorder=2)

    # 再画 Query（顶层，粗黑边框）
    for pid in unique_pids:
        c = pid_color[pid]
        for cid, local_idx in pid_cloth_map[pid].items():
            marker = marker_pool[min(local_idx, len(marker_pool)-1)]
            qm = (sp == pid) & (sc == cid) & sq
            if qm.any():
                ax.scatter(coords[qm, 0], coords[qm, 1],
                           c=[c], marker=marker, s=200, alpha=1.0,
                           edgecolors='black', linewidths=1.5, zorder=5)

    # 图例区域
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 左下：形状图例（衣服编号）
    shape_legend = [
        Line2D([0],[0], marker=marker_pool[i], color='gray', markersize=9,
               linestyle='None', label=f'Cloth {i+1}')
        for i in range(min(4, max([len(v) for v in pid_cloth_map.values()])))
    ]
    shape_legend += [
        Line2D([0],[0], marker='o', color='gray', markersize=8,
               markeredgecolor='none', linestyle='None', label='Gallery (no border)'),
        Line2D([0],[0], marker='o', color='gray', markersize=10,
               markeredgecolor='black', markeredgewidth=1.5,
               linestyle='None', label='Query (black border)'),
    ]
    leg1 = ax.legend(handles=shape_legend, loc='lower left',
                     fontsize=8, framealpha=0.9,
                     title='Shape=Cloth  Border=Query/Gallery')

    # 右上：颜色图例（身份）
    color_handles = [
        Line2D([0],[0], marker='o', color=pid_color[pid], markersize=8,
               linestyle='None', label=f'ID {pid}')
        for pid in unique_pids
    ]
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=7, ncol=2, title='Color = Identity')

    ax.set_title(
        f't-SNE: Cloth-Aware Feature Distribution  |  {cfg.DATASETS.NAMES.upper()}\n'
        f'Color=Identity  Shape=Cloth  Large+Border=Query  Small=Gallery',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.grid(True, alpha=0.2)



    plt.tight_layout()
    path = os.path.join(output_dir, 'fig8_tsne_cloth_aware.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[已保存] {path}")
    print("  诊断①: 同色不同形状聚团? → 是=换装鲁棒(L_sc/L_de有效)")
    print("  诊断②: ★Query落在同色●团内? → 是=跨摄像头对齐(L_Guide有效)")
    print("  诊断③: 不同颜色的团清晰分开? → 是=身份判别力强(L_ce/L_tri有效)")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CC-ReID 测试阶段可视化")
    parser.add_argument("--config_file",  type=str, required=True)
    parser.add_argument("--weight_path",  type=str, default="")
    parser.add_argument("--output_dir",   type=str, default="./vis_test")
    parser.add_argument("--num_vis",      type=int, default=10)
    parser.add_argument("--top_k",        type=int, default=5)
    parser.add_argument("--max_tsne_ids", type=int, default=20)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.TRAIN_STAGE = 2
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)

    # 自动寻找权重
    weight_path = args.weight_path
    if not weight_path:
        candidates = [
            os.path.join(cfg.OUTPUT_DIR,
                         f"stage2_disentangled_model_{cfg.SOLVER.STAGE2_MAX_EPOCHS}.pth"),
            os.path.join(cfg.OUTPUT_DIR,
                         f"stage1_prompt_model_{cfg.SOLVER.STAGE1_MAX_EPOCHS}.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                weight_path = c
                break
    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"找不到权重文件，请用 --weight_path 手动指定。"
            f"OUTPUT_DIR={cfg.OUTPUT_DIR}")

    print(f"\n{'='*55}")
    print(f"数据集    : {cfg.DATASETS.NAMES.upper()}")
    print(f"权重文件  : {weight_path}")
    print(f"输出目录  : {args.output_dir}")
    print(f"{'='*55}\n")

    model, val_loader, num_query, num_classes = load_model_and_data(cfg, weight_path)
    feats, pids, camids = do_inference(cfg, model, val_loader, num_query)

    q_pids   = np.asarray(pids[:num_query])
    g_pids   = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    distmat  = torch.cdist(feats[:num_query], feats[num_query:]).numpy()

    use_cc      = (cfg.DATASETS.NAMES == 'ltcc')
    q_cloth = g_cloth = None
    if use_cc:
        ds = LTCC(root=cfg.DATASETS.ROOT_DIR,
                  llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
        q_cloth = np.asarray([x[3] for x in ds.query])
        g_cloth = np.asarray([x[3] for x in ds.gallery])

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids,
                         q_cloth, g_cloth, ltcc_cc_setting=use_cc)

    print("\n========== 测试结果 ==========")
    print(f"mAP    : {mAP:.1%}")
    print(f"Rank-1 : {cmc[0]:.1%}")
    print(f"Rank-5 : {cmc[4]:.1%}")
    print(f"Rank-10: {cmc[9]:.1%}")
    print("=" * 30)

    # 获取路径和衣服ID
    q_paths, g_paths, q_clothids, g_clothids = get_dataset_info(cfg)

    # 生成可视化
    plot_retrieval(feats, pids, camids, q_paths, g_paths,
                   num_query, args.output_dir,
                   num_vis=args.num_vis, top_k=args.top_k)

    plot_tsne_cloth_aware(feats, pids, camids,
                          q_clothids, g_clothids,
                          num_query, args.output_dir,
                          max_ids=args.max_tsne_ids)

    print(f"\n✅ 所有可视化已保存至: {args.output_dir}")
    print("  fig7 - 检索结果可视化（论文实验章节）")
    print("  fig8 - 换装感知 t-SNE 特征分布图（论文分析章节）")


if __name__ == '__main__':
    main()