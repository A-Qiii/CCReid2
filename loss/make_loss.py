import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss

def make_loss(cfg, num_classes):
    stage = cfg.MODEL.TRAIN_STAGE
    
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = F.cross_entropy

    triplet = TripletLoss(getattr(cfg.SOLVER, 'MARGIN', 0.3))
    
    global_iter_counter = 0

    # 损失工厂：text_bank 仅在 Stage 2 由外部传入
    def loss_func(score, feat_list, target, text_bank_id=None):
        nonlocal global_iter_counter
        global_iter_counter += 1

        # ========================================================
        # 阶段一：仅优化提示词的 InfoNCE 对比损失
        # ========================================================
        if stage == 1:
            global_feat, t_id, t_cloth = feat_list
            
            # L2 归一化
            v_norm = F.normalize(global_feat, p=2, dim=1)
            t_id_norm = F.normalize(t_id, p=2, dim=1)
            t_cloth_norm = F.normalize(t_cloth, p=2, dim=1)
            
            logit_scale = 100.0 # 经验缩放系数
            
            # ID 对比
            logits_i2t_id = logit_scale * v_norm @ t_id_norm.t()
            logits_t2i_id = logit_scale * t_id_norm @ v_norm.t()
            loss_i2t_id = xent(logits_i2t_id, target)
            loss_t2i_id = xent(logits_t2i_id, target)
            
            # Cloth 对比 (仅为了拉伸空间，目标依然用 identity target，因为同人衣服一致)
            logits_i2t_cloth = logit_scale * v_norm @ t_cloth_norm.t()
            logits_t2i_cloth = logit_scale * t_cloth_norm @ v_norm.t()
            loss_i2t_cloth = xent(logits_i2t_cloth, target)
            loss_t2i_cloth = xent(logits_t2i_cloth, target)
            
            total_loss = loss_i2t_id + loss_t2i_id + loss_i2t_cloth + loss_t2i_cloth
            
            if global_iter_counter % 100 == 0:
                print(f"[Stage 1] InfoNCE Loss: {total_loss.item():.4f}")
                
            return total_loss, None

        # ========================================================
        # 阶段二：MIPL 外科手术解耦与视觉微调
        # ========================================================
        elif stage == 2:
            global_feat, f_img2clo, t_cloth_gt = feat_list
            
            # 1. 基础度量
            L_ce = xent(score, target)
            L_tri = triplet(global_feat, target)[0]
            
            # 2. 全局身份语义牵引 (L_Guide) - 解决“局部挣扎”问题
            if text_bank_id is None:
                raise ValueError("Stage 2 必须传入全局固化的 text_bank_id！")
            
            v_norm = F.normalize(global_feat, p=2, dim=1)
            t_id_bank_norm = F.normalize(text_bank_id, p=2, dim=1)
            
            logits_guide = 100.0 * v_norm @ t_id_bank_norm.t() # [Batch, num_classes]
            L_Guide = xent(logits_guide, target)
            
            # 3. 语义剥离校验 (L_sc) - 仅监督投影矩阵
            f_img2clo_norm = F.normalize(f_img2clo, p=2, dim=1)
            t_cloth_gt_norm = F.normalize(t_cloth_gt, p=2, dim=1)
            # 使用 MSE 保证特征空间的一致性
            L_sc = F.mse_loss(f_img2clo_norm, t_cloth_gt_norm) * 10.0 
            
            # 4. 截断余弦正交切除 (L_de) - 价值千金的防爆盾
            # 【绝对核心：.detach() 保护了 f_img2clo，迫使 ViT 脱衣服】
            cos_sim_ortho = torch.sum(v_norm * f_img2clo_norm.detach(), dim=1)
            L_de = torch.mean(F.relu(cos_sim_ortho))
            
            # 5. 合成总损失
            w_id = cfg.MODEL.ID_LOSS_WEIGHT
            w_tri = cfg.MODEL.TRIPLET_LOSS_WEIGHT
            w_guide = cfg.MODEL.I2T_ID_WEIGHT
            w_sc = cfg.MODEL.I2T_CLOTH_SC_WEIGHT
            w_ortho = cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT
            
            total_loss = w_id * L_ce + w_tri * L_tri + w_guide * L_Guide + w_sc * L_sc + w_ortho * L_de
            
            # 体检探针
            if global_iter_counter % 100 == 0:
                print(f"\n[Stage 2] === MIPL 手术刀体检探针 ===")
                print(f"-> 语义剥离校验 MSE (L_sc): {L_sc.item():.4f}")
                print(f"-> 正交切除强度 (L_de, Max=1): {L_de.item():.4f}")
                print(f"-> 基础[CE:{L_ce.item():.3f} TRI:{L_tri.item():.3f}] | 引导[Guide:{L_Guide.item():.3f}]")
                print(f"=======================================\n")
                
            return total_loss, None

    return loss_func