import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss

def make_loss(cfg, num_classes):
    xent = CrossEntropyLabelSmooth(num_classes=num_classes) if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
    triplet = TripletLoss(margin=0.3)

    def loss_func(score, feat_list, target, txt_tuple=None):
        ID_LOSS = xent(score, target)
        global_feat = feat_list[0]
        TRI_LOSS = triplet(global_feat, target)[0]
        
        # 获取 2 个高分辨率文本对齐矩阵
        score_i2t_id = feat_list[1]
        score_i2t_cloth = feat_list[2]
        B = global_feat.size(0)

        # 身份拉近
        target_mask = target.expand(B, B).eq(target.expand(B, B).t()).float()
        target_mask = target_mask / target_mask.sum(dim=1, keepdim=True)
        I2T_ID_LOSS = -(target_mask * F.log_softmax(score_i2t_id, dim=1)).sum(dim=1).mean()

        # 衣着排斥
        I2T_CLOTH_LOSS = F.kl_div(F.log_softmax(score_i2t_cloth, dim=1), 
                                  (torch.ones(B, B) / B).to(score_i2t_cloth.device), 
                                  reduction='batchmean')

        w_id = getattr(cfg.MODEL, 'I2T_ID_WEIGHT', 2.0)
        w_cloth = getattr(cfg.MODEL, 'I2T_CLOTH_WEIGHT', 0.5)

        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                     w_id * I2T_ID_LOSS + w_cloth * I2T_CLOTH_LOSS
        
        return total_loss, None
    return loss_func