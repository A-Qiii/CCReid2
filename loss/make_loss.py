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

    # ==========================================
    # 【新增】：全局步数计数器，用于等距规律打印
    # ==========================================
    global_iter_counter = 0

    def loss_func(score, feat_list, target, txt_tuple=None):
        nonlocal global_iter_counter # 声明使用外部的计数器
        global_iter_counter += 1

        # 1. 基础视觉分类损失
        ID_LOSS = xent(score, target)

        # 2. 三元组度量损失
        global_feat = feat_list[0]
        TRI_LOSS = triplet(global_feat, target)[0]
        
        # 解析特征
        t_id = feat_list[1]
        t_cloth = feat_list[2]
        v_cloth_for_guide = feat_list[3]
        v_cloth_for_ortho = feat_list[4]

        # 3. 身份图文对齐损失：KL 散度软约束
        B = global_feat.size(0)
        v_norm = F.normalize(global_feat, p=2, dim=1)
        t_id_norm = F.normalize(t_id, p=2, dim=1)
        
        sim_vis = torch.matmul(v_norm, v_norm.t())
        sim_text = torch.matmul(t_id_norm, t_id_norm.t())
        
        tau = 0.05
        sim_vis = sim_vis / tau
        sim_text = sim_text / tau
        
        mask = torch.eye(B, dtype=torch.bool).to(global_feat.device)
        sim_vis = sim_vis.masked_fill(mask, -10000.0)
        sim_text = sim_text.masked_fill(mask, -10000.0)
        
        alpha = F.softmax(sim_text, dim=1)
        log_beta = F.log_softmax(sim_vis, dim=1)
        I2T_ID_LOSS = F.kl_div(log_beta, alpha, reduction='batchmean')

        # 4. 彻底切断梯度泄漏的解耦排斥损失
        v_cloth_guide_norm = F.normalize(v_cloth_for_guide, p=2, dim=1)
        t_cloth_norm = F.normalize(t_cloth, p=2, dim=1)
        cos_sim_cloth = torch.sum(v_cloth_guide_norm * t_cloth_norm, dim=1)
        I2T_CLOTH_GUIDE_LOSS = 1.0 - torch.mean(cos_sim_cloth)

        v_cloth_ortho_norm = F.normalize(v_cloth_for_ortho, p=2, dim=1)
        cos_sim_ortho = torch.sum(v_norm * v_cloth_ortho_norm.detach(), dim=1) 
        I2T_CLOTH_ORTHO_LOSS = torch.mean(F.relu(cos_sim_ortho))

        # ========================================================
        # 5. 清爽版体检探针：每 100 个 Batch 精确打印一次
        # ========================================================
        if global_iter_counter % 100 == 0: 
            v_cloth_std = v_cloth_for_guide.std(dim=0).mean().item()
            avg_guide_sim = torch.mean(cos_sim_cloth).item()
            avg_ortho_sim = torch.mean(cos_sim_ortho).item()
            
            print(f"\n[Iter {global_iter_counter}] 深度特征体检探针 ====")
            print(f"-> 投影特征绝对标准差 (跌破0.01即坍塌): {v_cloth_std:.5f}")
            print(f"-> 视觉与文本相似度 (Guide): {avg_guide_sim:.4f}")
            print(f"-> 骨干与投影相似度 (Ortho): {avg_ortho_sim:.4f}")
            print(f"-> ID: {ID_LOSS.item():.3f} | TRI: {TRI_LOSS.item():.3f} | KL: {I2T_ID_LOSS.item():.3f}")
            print(f"========================================\n")

        # 6. 动态合成总损失
        w_base_id = getattr(cfg.MODEL, 'ID_LOSS_WEIGHT', 1.0)
        w_base_tri = getattr(cfg.MODEL, 'TRIPLET_LOSS_WEIGHT', 1.0)
        w_i2t_id = getattr(cfg.MODEL, 'I2T_ID_WEIGHT', 2.0) 
        w_i2t_cloth = getattr(cfg.MODEL, 'I2T_CLOTH_WEIGHT', 0.5)

        total_loss = w_base_id * ID_LOSS + \
                     w_base_tri * TRI_LOSS + \
                     w_i2t_id * I2T_ID_LOSS + \
                     w_i2t_cloth * I2T_CLOTH_GUIDE_LOSS + \
                     w_i2t_cloth * I2T_CLOTH_ORTHO_LOSS
        
        return total_loss, None

    return loss_func