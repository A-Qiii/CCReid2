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

    # 损失工厂：添加了 target_cloth，保证衣服的对比只与真实的衣服 ID 对齐
    def loss_func(score, feat_list, target, text_bank_id=None, target_cloth=None):
        nonlocal global_iter_counter
        global_iter_counter += 1

        # ========================================================
        # 阶段一：仅优化提示词的 InfoNCE 对比损失 (Batch内多正样本)
        # ========================================================
        if stage == 1:
            global_feat, t_id, t_cloth = feat_list
            
            v_norm = F.normalize(global_feat, p=2, dim=1)
            t_id_norm = F.normalize(t_id, p=2, dim=1)
            t_cloth_norm = F.normalize(t_cloth, p=2, dim=1)
            
            logit_scale = 20.0 
            
            # --- 1. 身份 (ID) 的 Batch 内软标签 InfoNCE ---
            logits_i2t_id = logit_scale * v_norm @ t_id_norm.t()
            logits_t2i_id = logit_scale * t_id_norm @ v_norm.t()
            
            # 构建 [Batch, Batch] 的软标签矩阵
            labels_id = (target.unsqueeze(1) == target.unsqueeze(0)).float().to(target.device)
            labels_id = labels_id / labels_id.sum(dim=1, keepdim=True) # 按行归一化为概率分布
            
            loss_i2t_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_i2t_id, dim=1), dim=1))
            loss_t2i_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_t2i_id, dim=1), dim=1))
            
            # --- 2. 衣着 (Cloth) 的 Batch 内软标签 InfoNCE ---
            logits_i2t_cloth = logit_scale * v_norm @ t_cloth_norm.t()
            logits_t2i_cloth = logit_scale * t_cloth_norm @ v_norm.t()
            
            # 如果有传入精确的衣着标签，则用它；否则兜底用 ID
            cloth_labels_tensor = target_cloth if target_cloth is not None else target
            labels_cloth = (cloth_labels_tensor.unsqueeze(1) == cloth_labels_tensor.unsqueeze(0)).float().to(target.device)
            labels_cloth = labels_cloth / labels_cloth.sum(dim=1, keepdim=True)
            
            loss_i2t_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_i2t_cloth, dim=1), dim=1))
            loss_t2i_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_t2i_cloth, dim=1), dim=1))
            
            total_loss = loss_i2t_id + loss_t2i_id + loss_i2t_cloth + loss_t2i_cloth
            
            if global_iter_counter % 50 == 0:
                # 探针：计算未经放大的纯物理余弦相似度
                sim_matrix = v_norm @ t_id_norm.t()
                pos_mask = labels_id.bool()
                
                # 提取正样本（同 ID）和负样本（不同 ID）的平均相似度
                pos_sim = sim_matrix[pos_mask].mean().item()
                # 负样本提取：取反 mask
                neg_sim = sim_matrix[~pos_mask].mean().item()
                
                print(f"\n[Stage 1 脉搏探针] Iter {global_iter_counter} | Total Loss: {total_loss.item():.4f}")
                print(f"-> ID 损失: {loss_i2t_id.item():.3f} | Cloth 损失: {loss_i2t_cloth.item():.3f}")
                print(f"-> 空间拉扯 (余弦相似度) | 正样本(目标逼近1): {pos_sim:.4f} | 负样本(目标逼近0): {neg_sim:.4f}")
                
                health_status = "🟩 健康生长" if pos_sim > neg_sim + 0.05 else ("🟨 挣扎对齐" if pos_sim > neg_sim else "🟥 严重坍塌")
                print(f"-> 提示词状态: {health_status}\n")
                
            return total_loss, None

        # ========================================================
        # 阶段二：MIPL 外科手术解耦与视觉微调
        # ========================================================
        elif stage == 2:
            global_feat, f_img2clo, t_cloth_gt = feat_list
            
            L_ce = xent(score, target)
            L_tri = triplet(global_feat, target)[0]
            
            if text_bank_id is None:
                raise ValueError("Stage 2 必须传入全局固化的 text_bank_id！")
            
            v_norm = F.normalize(global_feat, p=2, dim=1)
            t_id_bank_norm = F.normalize(text_bank_id, p=2, dim=1)
            
            logits_guide = 100.0 * v_norm @ t_id_bank_norm.t() 
            L_Guide = xent(logits_guide, target)
            
            f_img2clo_norm = F.normalize(f_img2clo, p=2, dim=1)
            t_cloth_gt_norm = F.normalize(t_cloth_gt, p=2, dim=1)
            L_sc = F.mse_loss(f_img2clo_norm, t_cloth_gt_norm) * 10.0 
            
            cos_sim_ortho = torch.sum(v_norm * f_img2clo_norm.detach(), dim=1)
            L_de = torch.mean(F.relu(cos_sim_ortho))
            
            w_id = cfg.MODEL.ID_LOSS_WEIGHT
            w_tri = cfg.MODEL.TRIPLET_LOSS_WEIGHT
            w_guide = cfg.MODEL.I2T_ID_WEIGHT
            w_sc = cfg.MODEL.I2T_CLOTH_SC_WEIGHT
            w_ortho = cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT
            
            total_loss = w_id * L_ce + w_tri * L_tri + w_guide * L_Guide + w_sc * L_sc + w_ortho * L_de
            
            if global_iter_counter % 100 == 0:
                print(f"\n[Stage 2] === MIPL 手术刀体检探针 ===")
                print(f"-> 语义剥离校验 MSE (L_sc): {L_sc.item():.4f}")
                print(f"-> 正交切除强度 (L_de, Max=1): {L_de.item():.4f}")
                print(f"-> 基础[CE:{L_ce.item():.3f} TRI:{L_tri.item():.3f}] | 引导[Guide:{L_Guide.item():.3f}]")
                print(f"=======================================\n")
                
            return total_loss, None

    return loss_func