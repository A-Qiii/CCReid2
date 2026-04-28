from torch.optim.lr_scheduler import MultiStepLR

class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=0.01,
                 warmup_iters=10, warmup_method="linear", last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** len([m for m in self.milestones if m <= self.last_epoch])
                for base_lr in self.base_lrs]

def build_lr_scheduler(cfg, optimizer):
    # 严格对齐 train.py 中的 scheduler = build_lr_scheduler(cfg, optimizer)
    return WarmupMultiStepLR(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=getattr(cfg.SOLVER, 'WARMUP_FACTOR', 0.01),
        warmup_iters=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_method=getattr(cfg.SOLVER, 'WARMUP_METHOD', "linear")
    )
