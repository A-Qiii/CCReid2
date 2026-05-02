import torch

def make_optimizer(cfg, model):
    params = []
    
    # 打印提示，确认我们的修改生效了
    print("========== 正在构建优化器 ==========")
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
            
        # 默认使用配置文件里的基础学习率 (比如你的 5e-6)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        
        # 偏置项 (bias) 的特殊处理
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * getattr(cfg.SOLVER, 'BIAS_LR_FACTOR', 1.0)
            weight_decay = getattr(cfg.SOLVER, 'WEIGHT_DECAY_BIAS', 0.0000)
            
        # ==========================================================
        # 【核心修改区】：为新初始化的层开通“10倍学习率”绿色通道
        # ==========================================================
        if cfg.SOLVER.LARGE_FC_LR:
            # 如果层名字里包含 classifier, bottleneck 或者我们新加的 cloth_proj
            if "classifier" in key or "bottleneck" in key or "cloth_proj" in key:
                lr = cfg.SOLVER.BASE_LR * 10.0 # 强制放大 10 倍！
                print(f"[高学习率特权] {key} -> 学习率放大 10 倍 ({lr})")
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
    # 根据 Config 选择优化器，强烈建议在 Config 里把名字改成 AdamW
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=getattr(cfg.SOLVER, 'MOMENTUM', 0.9))
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        print("-> 已启用 AdamW 优化器 (更适合 ViT 架构)")
    else:
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        print("-> 已启用普通 Adam 优化器")
        
    print("====================================")
    return optimizer