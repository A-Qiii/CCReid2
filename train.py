import os
import argparse
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
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


def print_config_summary(cfg):
    print("\n" + "=" * 60)
    print("📋 实验配置摘要（请核对超参数是否与预期一致）")
    print("=" * 60)
    print(f"  OUTPUT_DIR             : {cfg.OUTPUT_DIR}")
    print(f"  数据集                 : {cfg.DATASETS.NAMES}")
    print(f"  --- Stage 1 ---")
    print(f"  STAGE1_MAX_EPOCHS      : {cfg.SOLVER.STAGE1_MAX_EPOCHS}")
    print(f"  STAGE1_BASE_LR         : {cfg.SOLVER.STAGE1_BASE_LR}")
    print(f"  --- Stage 2 ---")
    print(f"  STAGE2_MAX_EPOCHS      : {cfg.SOLVER.STAGE2_MAX_EPOCHS}")
    print(f"  STAGE2_BASE_LR         : {cfg.SOLVER.STAGE2_BASE_LR}")
    print(f"  --- 损失权重 ---")
    print(f"  ID_LOSS_WEIGHT         : {cfg.MODEL.ID_LOSS_WEIGHT}")
    print(f"  TRIPLET_LOSS_WEIGHT    : {cfg.MODEL.TRIPLET_LOSS_WEIGHT}")
    print(f"  I2T_ID_WEIGHT          : {cfg.MODEL.I2T_ID_WEIGHT}")
    print(f"  I2T_CLOTH_SC_WEIGHT    : {cfg.MODEL.I2T_CLOTH_SC_WEIGHT}")
    print(f"  I2T_CLOTH_ORTHO_WEIGHT : {cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT}  ← 关键解耦权重")
    print(f"  PROMPT_LENGTH (M)      : {cfg.MODEL.PROMPT_LENGTH}")
    print("=" * 60 + "\n")


def train(cfg):
    set_seed(cfg.SOLVER.SEED)
    print_config_summary(cfg)

    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    dataset_obj = train_loader.dataset.dataset
    num_clothes = getattr(dataset_obj, 'num_train_clothes', 1000)

    cfg.defrost()
    cfg.MODEL.NUM_CLOTHES = num_clothes
    cfg.freeze()
    print(f"数据总线接通 -> ID分类数: {num_classes}, 衣服数: {num_clothes}")

    # =====================================================================
    # STAGE 1：混合提示学习
    # =====================================================================
    print("\n" + "="*70)
    print("🚀 [STAGE 1 启动]")
    print("="*70)

    cfg.defrost(); cfg.MODEL.TRAIN_STAGE = 1; cfg.freeze()
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.to(cfg.MODEL.DEVICE)

    s1_tb_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard", "stage1")
    os.makedirs(s1_tb_dir, exist_ok=True)
    s1_writer = SummaryWriter(log_dir=s1_tb_dir)
    s1_writer.add_text("config/stage1_lr",    str(cfg.SOLVER.STAGE1_BASE_LR),          0)
    s1_writer.add_text("config/ortho_weight", str(cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT),   0)

    # 【修复】显式传入 stage=1
    loss_s1    = make_loss(cfg, num_classes=num_classes, stage=1, tb_writer=s1_writer)
    opt_s1     = make_optimizer(cfg, model)
    sched_s1   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=cfg.SOLVER.STAGE1_MAX_EPOCHS)

    do_train_stage1(cfg, model, train_loader, opt_s1, sched_s1, loss_s1)
    s1_writer.close()

    # =====================================================================
    # BRIDGE：提取 Text Bank
    # =====================================================================
    print("\n" + "="*70)
    print("🌉 [阶段交接]：提取全局 Text Bank")
    print("="*70)
    text_bank_id = extract_text_bank(cfg, model, train_loader)

    # =====================================================================
    # STAGE 2：MIPL 解耦微调
    # =====================================================================
    print("\n" + "="*70)
    print("🚀 [STAGE 2 启动]")
    print("="*70)

    cfg.defrost(); cfg.MODEL.TRAIN_STAGE = 2; cfg.freeze()
    model.switch_stage(2)

    s2_tb_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard", "stage2")
    os.makedirs(s2_tb_dir, exist_ok=True)
    s2_writer = SummaryWriter(log_dir=s2_tb_dir)
    s2_writer.add_text("config/stage2_lr",    str(cfg.SOLVER.STAGE2_BASE_LR),        0)
    s2_writer.add_text("config/ortho_weight", str(cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT), 0)
    s2_writer.add_text("config/sc_weight",    str(cfg.MODEL.I2T_CLOTH_SC_WEIGHT),    0)

    # 【修复】显式传入 stage=2，此时 cfg 里的权重已是命令行覆盖后的值
    loss_s2  = make_loss(cfg, num_classes=num_classes, stage=2, tb_writer=s2_writer)
    opt_s2   = make_optimizer(cfg, model)
    sched_s2 = WarmupMultiStepLR(opt_s2, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                 cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.STAGE2_WARMUP_EPOCHS,
                                 cfg.SOLVER.WARMUP_METHOD)

    do_train_stage2(cfg, model, train_loader, val_loader, opt_s2, sched_s2,
                    loss_s2, num_query, text_bank_id)
    s2_writer.close()

    print(f"\n🎉 训练完成！可视化：python visualize_training.py --log_dir {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/prcc/vit_ccreid_prcc.yml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    train(cfg)