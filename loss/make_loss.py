import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss

def make_loss(cfg, num_classes):
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = F.cross_entropy

    margin = getattr(cfg.SOLVER, 'MARGIN', 0.3)
    triplet = TripletLoss(margin)

    def loss_func(score, feat_list, target, txt_tuple=None):
        # 1. 基础视觉分类损失
        ID_LOSS = xent(score, target)

        # 2. 三元组度量损失
        global_feat = feat_list[0]
        TRI_LOSS = triplet(global_feat, target)[0]
        
        # 3. 身份图文对齐损失 (InfoNCE)
        score_i2t_id = feat_list[1]
        B = global_feat.size(0)
        target_mask = target.expand(B, B).eq(target.expand(B, B).t()).float()
        target_mask = target_mask / target_mask.sum(dim=1, keepdim=True)
        I2T_ID_LOSS = -(target_mask * F.log_softmax(score_i2t_id, dim=1)).sum(dim=1).mean()

        # 4. 基于 MIPL 的截断正交衣着排斥损失
        t_cloth = feat_list[2]
        v_norm = F.normalize(global_feat, p=2, dim=1)
        t_cloth_norm = F.normalize(t_cloth, p=2, dim=1)
        cos_sim = torch.sum(v_norm * t_cloth_norm, dim=1)
        I2T_CLOTH_LOSS = torch.mean(F.relu(cos_sim))

        # -----------------------------------------------------------
        # 5. 核心修复：重新接入动态权重读取，让 run_search.sh 真正生效
        # -----------------------------------------------------------
        
        # 读取基础视觉权重 (默认通常是 1.0)
        w_base_id = getattr(cfg.MODEL, 'ID_LOSS_WEIGHT', 1.0)
        w_base_tri = getattr(cfg.MODEL, 'TRIPLET_LOSS_WEIGHT', 1.0)
        
        # 读取跨模态权重 (也就是你脚本里正在疯狂搜索的这两个变量)
        w_i2t_id = getattr(cfg.MODEL, 'I2T_ID_WEIGHT', 2.0)
        w_i2t_cloth = getattr(cfg.MODEL, 'I2T_CLOTH_WEIGHT', 0.5)

        # 动态合成总损失
        total_loss = w_base_id * ID_LOSS + \
                     w_base_tri * TRI_LOSS + \
                     w_i2t_id * I2T_ID_LOSS + \
                     w_i2t_cloth * I2T_CLOTH_LOSS
        
        return total_loss, None

    return loss_func