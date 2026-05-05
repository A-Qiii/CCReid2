"""
test_visualize.py
=================
测试阶段可视化脚本，生成两类论文级图表：
  1. 检索结果可视化（Top-K Retrieval）：绿框=正确，红框=错误
  2. t-SNE 特征分布图：展示特征聚类效果

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

    # 从 checkpoint 直接读取真实维度，避免 NUM_CLOTHES=0 导致结构不匹配
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


def get_image_paths(cfg):
    if cfg.DATASETS.NAMES == 'prcc':
        dataset = PRCC(root=cfg.DATASETS.ROOT_DIR,
                       llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    elif cfg.DATASETS.NAMES == 'ltcc':
        dataset = LTCC(root=cfg.DATASETS.ROOT_DIR,
                       llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    else:
        raise ValueError(f"不支持的数据集: {cfg.DATASETS.NAMES}")
    q_paths = [x[0] for x in dataset.query]
    g_paths = [x[0] for x in dataset.gallery]
    return q_paths, g_paths


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

        order  = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        valid_order = order[~remove]

        # Query 列
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

        # Top-K 列
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
# 图2：t-SNE 特征分布可视化
# ──────────────────────────────────────────────
def plot_tsne(feats, pids, num_query, output_dir, max_ids=20, max_samples=1000):
    print(">>> 正在计算 t-SNE...")
    all_feats = feats.numpy()
    all_pids  = np.asarray(pids)
    is_query  = np.array([True]*num_query + [False]*(len(pids)-num_query))

    unique_pids = np.unique(all_pids)[:max_ids]
    mask = np.isin(all_pids, unique_pids)
    sf, sp, sq = all_feats[mask], all_pids[mask], is_query[mask]

    if len(sf) > max_samples:
        idx = np.random.choice(len(sf), max_samples, replace=False)
        sf, sp, sq = sf[idx], sp[idx], sq[idx]

    coords = TSNE(n_components=2, random_state=42,
                  perplexity=30, max_iter=1000).fit_transform(sf)

    cmap = plt.cm.get_cmap('tab20', max_ids)
    pid_color = {pid: cmap(i) for i, pid in enumerate(unique_pids)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for pid in unique_pids:
        c = pid_color[pid]
        gm = (sp == pid) & ~sq
        if gm.any():
            ax.scatter(coords[gm, 0], coords[gm, 1],
                       c=[c], marker='o', s=40, alpha=0.7,
                       edgecolors='none', label=f'ID {pid}')
        qm = (sp == pid) & sq
        if qm.any():
            ax.scatter(coords[qm, 0], coords[qm, 1],
                       c=[c], marker='*', s=150, alpha=1.0,
                       edgecolors='black', linewidths=0.5)

    ax.set_title(
        f't-SNE Feature Distribution  |  {cfg.DATASETS.NAMES.upper()}\n'
        f'★=Query  ●=Gallery  (Top-{max_ids} identities)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=7, ncol=2, markerscale=1.2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig8_tsne_feature_dist.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[已保存] {path}")


# ──────────────────────────────────────────────
# 主函数（注意：不包含 opts 参数，避免与 -- 参数冲突）
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CC-ReID 测试阶段可视化")
    parser.add_argument("--config_file",  type=str, required=True,
                        help="配置文件路径")
    parser.add_argument("--weight_path",  type=str, default="",
                        help="模型权重路径，留空则自动在 OUTPUT_DIR 下寻找最新权重")
    parser.add_argument("--output_dir",   type=str, default="./vis_test",
                        help="可视化图表输出目录")
    parser.add_argument("--num_vis",      type=int, default=10,
                        help="检索可视化展示的 Query 数量")
    parser.add_argument("--top_k",        type=int, default=5,
                        help="每个 Query 展示的检索结果数量")
    parser.add_argument("--max_tsne_ids", type=int, default=20,
                        help="t-SNE 图展示的最大身份数量")
    # 注意：此处不再添加 opts 参数，测试脚本所有配置通过 --config_file 指定
    args = parser.parse_args()

    # 加载配置（仅从文件读取，不支持命令行覆盖，避免歧义）
    cfg.merge_from_file(args.config_file)
    # 测试阶段强制设置 Stage=2（确保模型结构正确）
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
            f"找不到权重文件。请用 --weight_path 手动指定，或确认 OUTPUT_DIR={cfg.OUTPUT_DIR} 下存在权重。")

    print(f"\n{'='*50}")
    print(f"数据集    : {cfg.DATASETS.NAMES.upper()}")
    print(f"权重文件  : {weight_path}")
    print(f"输出目录  : {args.output_dir}")
    print(f"{'='*50}\n")

    # 加载模型并提取特征
    model, val_loader, num_query, num_classes = load_model_and_data(cfg, weight_path)
    feats, pids, camids = do_inference(cfg, model, val_loader, num_query)

    # 评测指标
    q_pids   = np.asarray(pids[:num_query])
    g_pids   = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    distmat  = torch.cdist(feats[:num_query], feats[num_query:]).numpy()

    use_cc = (cfg.DATASETS.NAMES == 'ltcc')
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

    # 获取图片路径并生成可视化
    q_paths, g_paths = get_image_paths(cfg)
    plot_retrieval(feats, pids, camids, q_paths, g_paths,
                   num_query, args.output_dir,
                   num_vis=args.num_vis, top_k=args.top_k)
    plot_tsne(feats, pids, num_query, args.output_dir,
              max_ids=args.max_tsne_ids)

    print(f"\n✅ 所有可视化已保存至: {args.output_dir}")
    print("  fig7 - 检索结果可视化")
    print("  fig8 - t-SNE 特征分布图")


if __name__ == '__main__':
    main()