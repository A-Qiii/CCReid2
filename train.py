import os
import argparse
import torch
import random
import numpy as np
from configs import cfg
from datasets import make_dataloader
from modeling import make_model
from loss import make_loss
from solver import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor import do_train_stage1, extract_text_bank, do_train_stage2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(cfg):
    set_seed(cfg.SOLVER.SEED)

    # 1. 准备数据流
    # (假设你的 make_dataloader 返回以下 7 个对象)
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 动态获取总衣服种类数给网络
    dataset_obj = train_loader.dataset.dataset
    if hasattr(dataset_obj, 'num_train_clothes'):
        num_clothes = dataset_obj.num_train_clothes
    else:
        num_clothes = 1000 # 兜底机制

    cfg.defrost()
    cfg.MODEL.NUM_CLOTHES = num_clothes
    cfg.freeze()

    print(f"数据总线接通 -> 全局分类数(ID): {num_classes}, 探测到全局衣服数(Cloth): {cfg.MODEL.NUM_CLOTHES}")

    # =====================================================================
    # 【STAGE 1】：混合提示学习 (Hybrid Prompting)
    # =====================================================================
    print("\n" + "="*70)
    print("🚀 [STAGE 1 启动]：对比学习构建细粒度语义锚点")
    print("="*70)
    cfg.defrost(); cfg.MODEL.TRAIN_STAGE = 1; cfg.freeze()

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.to(cfg.MODEL.DEVICE)

    loss_func_s1 = make_loss(cfg, num_classes=num_classes)
    optimizer_s1 = make_optimizer(cfg, model)
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=cfg.SOLVER.STAGE1_MAX_EPOCHS)

    do_train_stage1(cfg, model, train_loader, optimizer_s1, scheduler_s1, loss_func_s1)

    # =====================================================================
    # 【BRIDGE】：提取并固化记忆银行 (Text Feature Bank)
    # =====================================================================
    print("\n" + "="*70)
    print("🌉 [阶段交接]：固化 Prompt 并提取全局 Text Bank")
    print("="*70)
    text_bank_id = extract_text_bank(cfg, model, train_loader)

    # =====================================================================
    # 【STAGE 2】：MIPL 视觉特征解耦与微调
    # =====================================================================
    print("\n" + "="*70)
    print("🚀 [STAGE 2 启动]：全局引导与截断正交的联合解耦手术")
    print("="*70)
    cfg.defrost(); cfg.MODEL.TRAIN_STAGE = 2; cfg.freeze()

    # 触发 Stage 2 极其严格的梯度冻结/解冻规则 (锁定 Text 侧，放开 ViT 和 Proj)
    model._freeze_parameters_by_stage()

    loss_func_s2 = make_loss(cfg, num_classes=num_classes)
    optimizer_s2 = make_optimizer(cfg, model)
    scheduler_s2 = WarmupMultiStepLR(optimizer_s2, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                     cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.STAGE2_WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train_stage2(cfg, model, train_loader, val_loader, optimizer_s2, scheduler_s2, loss_func_s2, num_query, text_bank_id)
    
    print("\n🎉 CC-ReID 两阶段联合训练圆满结束！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CC-ReID Hybrid Two-Stage Training")
    parser.add_argument("--config_file", default="configs/prcc/vit_ccreid_prcc.yml", help="path to config file", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    train(cfg)