import os
import argparse
import torch
import numpy as np
from configs import cfg
from datasets import make_dataloader
from datasets.ltcc import LTCC
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
    # 恢复为接收 3 个基础变量
    feats, pids, camids = do_inference(cfg, model, val_loader, num_query)
    
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    
    # --- 核心优化：直接利用新版 LTCC 类的 5 元组物理结构提取 cloth_id ---
    dataset = LTCC(root=cfg.DATASETS.ROOT_DIR, llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    
    # 你的 dataset.query 中每个元素 x 为: (img_path, pid, camid, cloth_id, cloth_text)
    # 直接提取索引 [3] 即可获得最精准的衣服标签
    q_clothids = np.asarray([x[3] for x in dataset.query])
    g_clothids = np.asarray([x[3] for x in dataset.gallery])
    # ---------------------------------
    
    # 物理切分 Query 和 Gallery
    qf, q_pids, q_camids = feats[:num_query], pids[:num_query], camids[:num_query]
    gf, g_pids, g_camids = feats[num_query:], pids[num_query:], camids[num_query:]
    
    print(">>> 正在计算特征欧氏距离矩阵...")
    distmat = torch.cdist(qf, gf).numpy()
    
    # 移交评测引擎
    print(">>> 开始计算 mAP 与 CMC 排位指标 (CC-Setting 换衣地狱模式)...")
    cmc, mAP = eval_func(
        distmat=distmat, 
        q_pids=q_pids, g_pids=g_pids, 
        q_camids=q_camids, g_camids=g_camids,
        q_clothids=q_clothids, g_clothids=g_clothids,
        ltcc_cc_setting=True  # 核心开关：强制抹除同衣服作弊项
    )
    
    print("\n========== 最终测试报告 (CC 模式) ==========")
    print(f"mAP (平均精度均值) : {mAP:.1%}")
    print(f"Rank-1 (首位命中率): {cmc[0]:.1%}")
    print(f"Rank-5             : {cmc[4]:.1%}")
    print(f"Rank-10            : {cmc[9]:.1%}")
    print("===========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 对齐新工程的唯一主配置文件
    parser.add_argument("--config_file", default="configs/ltcc/vit_ccreid.yml", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    test(cfg)