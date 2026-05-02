import torch

def make_optimizer(cfg, model):
    params = []
    stage = cfg.MODEL.TRAIN_STAGE
    print(f"========== 正在构建 Stage {stage} 优化器 ==========")

    if stage == 1:
        # 阶段一：纯优化提示词 (Prompt Learner)
        base_lr = cfg.SOLVER.STAGE1_BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": base_lr, "weight_decay": weight_decay}]
            
        optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
        print(f"-> [Stage 1] 已启用 Adam 优化器 (LR: {base_lr})")

    elif stage == 2:
        # 阶段二：解耦微调骨干网络与投影器
        base_lr = cfg.SOLVER.STAGE2_BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue

            lr = base_lr
            if "bias" in key:
                lr = base_lr * getattr(cfg.SOLVER, 'BIAS_LR_FACTOR', 1.0)
                weight_decay = getattr(cfg.SOLVER, 'WEIGHT_DECAY_BIAS', 0.0000)

            # 【核心护城河】：给予外科手术刀 10 倍学习率特权！
            if "cloth_proj" in key:
                lr = base_lr * 10.0
                print(f"[高优特权] {key} -> 学习率放大 10 倍 ({lr})")

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        # 推荐使用 AdamW 以强化 ViT 的高维泛化能力
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        print(f"-> [Stage 2] 已启用 AdamW 优化器 (Base LR: {base_lr})")

    print("==================================================")
    return optimizer