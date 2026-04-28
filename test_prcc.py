import os
import argparse
import torch
import numpy as np
from configs import cfg
from datasets import make_dataloader
from datasets.ltcc import LTCC
from datasets.prcc import PRCC 
from modeling import make_model
from processor.processor import do_inference
from processor.evaluator import eval_func

def test(cfg):
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # 挂载 60 轮物理权重
    weight_path = os.path.join(cfg.OUTPUT_DIR, "model_60.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到物理权重文件: {weight_path}")
    print(f">>> 成功挂载权重: {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location='cpu')) 
    model.to('cuda')
    
    print(">>> 开始提取测试集特征...")
    feats, pids, camids = do_inference(cfg, model, val_loader, num_query)
    
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    
    # 物理切分 Query 和 Gallery
    qf, q_pids, q_camids = feats[:num_query], pids[:num_query], camids[:num_query]
    gf, g_pids, g_camids = feats[num_query:], pids[num_query:], camids[num_query:]
    
    print(">>> 正在计算特征欧氏距离矩阵...")
    distmat = torch.cdist(qf, gf).numpy()
    
    # --- 核心优化：根据数据集协议进行动态评测 ---
    if cfg.DATASETS.NAMES == 'ltcc':
        print(">>> 探测到 LTCC 数据集，启用带衣着标签的 CC-Setting 过滤模式...")
        dataset = LTCC(root=cfg.DATASETS.ROOT_DIR, llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
        q_clothids = np.asarray([x[3] for x in dataset.query])
        g_clothids = np.asarray([x[3] for x in dataset.gallery])
        
        cmc, mAP = eval_func(
            distmat=distmat, 
            q_pids=q_pids, g_pids=g_pids, 
            q_camids=q_camids, g_camids=g_camids,
            q_clothids=q_clothids, g_clothids=g_clothids,
            ltcc_cc_setting=True  # 核心开关：强制抹除同衣服作弊项
        )
        
    elif cfg.DATASETS.NAMES == 'prcc':
        print(">>> 探测到 PRCC 数据集，启用标准跨相机(Query=C, Gallery=A/B)评测模式...")
        # PRCC 官方协议：天然跨衣，无需 mask 同衣着项
        cmc, mAP = eval_func(
            distmat=distmat, 
            q_pids=q_pids, g_pids=g_pids, 
            q_camids=q_camids, g_camids=g_camids,
            q_clothids=None, g_clothids=None,
            ltcc_cc_setting=False
        )
    else:
        raise ValueError(f"未知的评测数据集: {cfg.DATASETS.NAMES}")
    
    print("\n========== 最终测试报告 ==========")
    print(f"mAP (平均精度均值) : {mAP:.1%}")
    print(f"Rank-1 (首位命中率): {cmc[0]:.1%}")
    print(f"Rank-5             : {cmc[4]:.1%}")
    print(f"Rank-10            : {cmc[9]:.1%}")
    print("===========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认值保留为您现在的 PRCC 配置文件
    parser.add_argument("--config_file", default="configs/prcc/vit_ccreid_prcc.yml", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    test(cfg)