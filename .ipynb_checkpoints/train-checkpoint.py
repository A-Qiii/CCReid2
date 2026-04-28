import os
import argparse
import random
import numpy as np
import torch
from configs import cfg
from datasets import make_dataloader
from modeling import make_model
from loss import make_loss
from processor.processor import do_train
from solver import make_optimizer, build_lr_scheduler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(cfg):
    # 1. 加载数据
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 2. 模型初始化
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.to('cuda')

    # --- 断点续训逻辑：检测 model_40.pth ---
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "model_40.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f">>> 发现断点权重，正在加载第 40 轮模型: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        start_epoch = 40
    else:
        print(">>> 未发现断点权重，将从第 1 轮开始训练。")
    # -----------------------------------

    # 3. 损失函数
    loss_func = make_loss(cfg, num_classes=num_classes)

    # 4. 优化器与调度器
    optimizer = make_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # 5. 启动训练，传入 start_epoch
    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
        start_epoch=start_epoch
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CC-ReID Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    seed = getattr(cfg.SOLVER, 'SEED', 1234)
    set_seed(seed)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    train(cfg)