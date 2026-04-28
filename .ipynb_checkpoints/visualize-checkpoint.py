import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from configs import cfg
from datasets import make_dataloader
from datasets.ltcc import LTCC
from modeling import make_model
from processor.processor import do_inference

def visualize(cfg):
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 挂载 60 轮物理权重
    weight_path = os.path.join(cfg.OUTPUT_DIR, "model_60.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到物理权重文件: {weight_path}")
    print(f">>> 成功挂载权重: {weight_path}")

    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.to('cuda')

    print(">>> 正在提取全量测试集特征...")
    feats, pids, camids = do_inference(cfg, model, val_loader, num_query)
    pids = np.asarray(pids)
    camids = np.asarray(camids)

    # --- 核心优化：直接利用新版 LTCC 类的 5 元组物理结构提取特征 ---
    dataset = LTCC(root=cfg.DATASETS.ROOT_DIR, llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    
    # x 的结构为: (img_path, pid, camid, cloth_id, cloth_text)
    q_paths = [x[0] for x in dataset.query]
    g_paths = [x[0] for x in dataset.gallery]
    q_clothids = np.asarray([x[3] for x in dataset.query])
    g_clothids = np.asarray([x[3] for x in dataset.gallery])
    # ---------------------------------

    qf, q_pids, q_camids = feats[:num_query], pids[:num_query], camids[:num_query]
    gf, g_pids, g_camids = feats[num_query:], pids[num_query:], camids[num_query:]

    print(">>> 正在计算欧氏距离矩阵...")
    distmat = torch.cdist(qf, gf).numpy()

    print(">>> 正在生成 CC-Setting 换衣地狱模式可视化检验图...")
    num_vis = 10  
    top_k = 5   

    fig, axes = plt.subplots(num_vis, top_k + 1, figsize=(22, 5 * num_vis))
    indices = np.argsort(distmat, axis=1)

    sampled_q_idxs = np.random.choice(num_query, num_vis, replace=False)

    for i, q_idx in enumerate(sampled_q_idxs):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_clothid = q_clothids[q_idx]

        q_img = Image.open(q_paths[q_idx]).convert('RGB')
        axes[i, 0].imshow(q_img)
        axes[i, 0].set_title(f"Query\nPID: {q_pid}\nCloth: {q_clothid}", fontweight='bold')
        axes[i, 0].axis('off')

        order = indices[q_idx]
        
        # 换衣模式掩码 (剔除同镜头与同衣服样本)
        remove = ((g_pids[order] == q_pid) & (g_camids[order] == q_camid)) | \
                 ((g_pids[order] == q_pid) & (g_clothids[order] == q_clothid))
        keep = np.invert(remove)
        valid_g_order = order[keep]

        for j in range(top_k):
            g_idx = valid_g_order[j]
            g_img = Image.open(g_paths[g_idx]).convert('RGB')
            g_pid_current = g_pids[g_idx]
            g_clothid_current = g_clothids[g_idx]

            ax = axes[i, j + 1]
            ax.imshow(g_img)
            ax.set_xticks([])
            ax.set_yticks([])

            if g_pid_current == q_pid:
                ax.set_title(f"True\nPID: {g_pid_current}\nCloth: {g_clothid_current}", color='green', fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(4)
            else:
                ax.set_title(f"False\nPID: {g_pid_current}\nCloth: {g_clothid_current}", color='red', fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)

    plt.tight_layout()
    save_path = os.path.join(cfg.OUTPUT_DIR, "vis_cc_results.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f">>> CC 模式物理透视图已成功落盘至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 强制对齐新系统的总控配置
    parser.add_argument("--config_file", default="configs/ltcc/vit_ccreid.yml", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    visualize(cfg)