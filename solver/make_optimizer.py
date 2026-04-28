import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # 使用 getattr 安全获取可选参数，适配 YACS 缺省行为
        bias_lr_factor = getattr(cfg.SOLVER, 'BIAS_LR_FACTOR', 1.0)
        weight_decay_bias = getattr(cfg.SOLVER, 'WEIGHT_DECAY_BIAS', weight_decay)
        large_fc_lr = getattr(cfg.SOLVER, 'LARGE_FC_LR', False)

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * bias_lr_factor
            weight_decay = weight_decay_bias

        if large_fc_lr and "classifier" in key:
            lr = cfg.SOLVER.BASE_LR * 2.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = getattr(cfg.SOLVER, 'OPTIMIZER_NAME', 'Adam')
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(params)

    # 严格对齐 train.py 中的 optimizer = make_optimizer(cfg, model)
    return optimizer
