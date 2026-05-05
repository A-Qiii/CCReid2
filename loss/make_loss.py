import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes, stage, tb_writer=None):
    """
    损失函数工厂。

    【修复说明】原代码在工厂函数调用时捕获 cfg.MODEL.TRAIN_STAGE，
    导致 Stage 2 的 loss_func 闭包里 stage 变量永远是 1，
    Stage 2 分支的代码和权重读取完全无效。
    修复方案：将 stage 作为显式参数传入，在 train.py 中分别为两个阶段
    创建各自的 loss_func，彻底解决闭包捕获问题。

    参数：
        cfg:        配置对象
        num_classes:分类数
        stage:      显式指定当前阶段（1 或 2），不再从 cfg 动态读取
        tb_writer:  可选 TensorBoard SummaryWriter
    """
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = F.cross_entropy

    triplet = TripletLoss(getattr(cfg.SOLVER, 'MARGIN', 0.3))
    cur_iter = [0]

    def loss_func(score, feat_list, target, text_bank_id=None, target_cloth=None):
        cur_iter[0] += 1
        n = cur_iter[0]

        # ============================================================
        # Stage 1：InfoNCE 对比损失
        # ============================================================
        if stage == 1:
            global_feat, t_id, t_cloth = feat_list

            v_norm      = F.normalize(global_feat, p=2, dim=1)
            t_id_norm   = F.normalize(t_id,        p=2, dim=1)
            t_cloth_norm = F.normalize(t_cloth,    p=2, dim=1)

            logit_scale = 20.0

            # 身份 InfoNCE（双向）
            logits_i2t_id = logit_scale * v_norm @ t_id_norm.t()
            logits_t2i_id = logit_scale * t_id_norm @ v_norm.t()
            labels_id = (target.unsqueeze(1) == target.unsqueeze(0)).float().to(target.device)
            labels_id = labels_id / labels_id.sum(dim=1, keepdim=True)
            loss_i2t_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_i2t_id, dim=1), dim=1))
            loss_t2i_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_t2i_id, dim=1), dim=1))

            # 衣着 InfoNCE（双向）
            logits_i2t_cloth = logit_scale * v_norm @ t_cloth_norm.t()
            logits_t2i_cloth = logit_scale * t_cloth_norm @ v_norm.t()
            cloth_labels_tensor = target_cloth if target_cloth is not None else target
            labels_cloth = (cloth_labels_tensor.unsqueeze(1) == cloth_labels_tensor.unsqueeze(0)).float().to(target.device)
            labels_cloth = labels_cloth / labels_cloth.sum(dim=1, keepdim=True)
            loss_i2t_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_i2t_cloth, dim=1), dim=1))
            loss_t2i_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_t2i_cloth, dim=1), dim=1))

            total_loss = loss_i2t_id + loss_t2i_id + loss_i2t_cloth + loss_t2i_cloth

            if n % 50 == 0:
                sim_matrix = v_norm @ t_id_norm.t()
                pos_mask = labels_id.bool()
                pos_sim = sim_matrix[pos_mask].mean().item()
                neg_sim = sim_matrix[~pos_mask].mean().item()
                health = ("🟩 健康" if pos_sim > neg_sim + 0.05
                          else ("🟨 挣扎" if pos_sim > neg_sim else "🟥 坍塌"))
                print(f"\n[Stage 1 探针] Iter {n} | Loss: {total_loss.item():.4f} | "
                      f"pos: {pos_sim:.4f} neg: {neg_sim:.4f} | {health}")
                if tb_writer is not None:
                    tb_writer.add_scalar("Train/Stage1_PosSim", pos_sim, n)
                    tb_writer.add_scalar("Train/Stage1_NegSim", neg_sim, n)

            return total_loss, None

        # ============================================================
        # Stage 2：MIPL 外科手术解耦 + 视觉微调
        # ============================================================
        elif stage == 2:
            global_feat, f_img2clo, t_cloth_gt = feat_list

            L_ce  = xent(score, target)
            L_tri = triplet(global_feat, target)[0]

            if text_bank_id is None:
                raise ValueError("Stage 2 必须传入固化的 text_bank_id！")

            v_norm          = F.normalize(global_feat,  p=2, dim=1)
            t_id_bank_norm  = F.normalize(text_bank_id, p=2, dim=1)
            logits_guide    = 100.0 * v_norm @ t_id_bank_norm.t()
            L_Guide         = xent(logits_guide, target)

            f_img2clo_norm  = F.normalize(f_img2clo,  p=2, dim=1)
            t_cloth_gt_norm = F.normalize(t_cloth_gt, p=2, dim=1)
            L_sc = F.mse_loss(f_img2clo_norm, t_cloth_gt_norm) * 10.0

            cos_sim_ortho = torch.sum(v_norm * f_img2clo_norm.detach(), dim=1)
            L_de = torch.mean(F.relu(cos_sim_ortho))

            # 从 cfg 实时读取权重（确保命令行覆盖生效）
            w_id     = cfg.MODEL.ID_LOSS_WEIGHT
            w_tri    = cfg.MODEL.TRIPLET_LOSS_WEIGHT
            w_guide  = cfg.MODEL.I2T_ID_WEIGHT
            w_sc     = cfg.MODEL.I2T_CLOTH_SC_WEIGHT
            w_ortho  = cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT  # 关键权重

            total_loss = w_id*L_ce + w_tri*L_tri + w_guide*L_Guide + w_sc*L_sc + w_ortho*L_de

            if tb_writer is not None and n % 10 == 0:
                tb_writer.add_scalar("Train/Stage2_TotalLoss", total_loss.item(), n)
                tb_writer.add_scalar("Train/Stage2_L_ce",      L_ce.item(),       n)
                tb_writer.add_scalar("Train/Stage2_L_tri",     L_tri.item(),      n)
                tb_writer.add_scalar("Train/Stage2_L_Guide",   L_Guide.item(),    n)
                tb_writer.add_scalar("Train/Stage2_L_sc",      L_sc.item(),       n)
                tb_writer.add_scalar("Train/Stage2_L_de",      L_de.item(),       n)
                tb_writer.add_scalar("Train/Stage2_BaseAcc",
                                     (score.max(1)[1] == target).float().mean().item(), n)

            if n % 100 == 0:
                print(f"\n[Stage 2] Iter {n} | Total: {total_loss.item():.4f} | "
                      f"L_ce: {L_ce.item():.3f} L_tri: {L_tri.item():.3f} "
                      f"L_Guide: {L_Guide.item():.3f} L_sc: {L_sc.item():.4f} "
                      f"L_de: {L_de.item():.4f} (w={w_ortho})")

            return total_loss, None

    return loss_func