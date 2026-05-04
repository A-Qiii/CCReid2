import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes, tb_writer=None):
    """
    损失函数工厂。
    
    参数：
        cfg: 配置对象
        num_classes: 分类数
        tb_writer: 可选的 TensorBoard SummaryWriter，
                   传入后会在 Stage 2 将各损失分量实时写入，
                   用于 visualize_training.py 的 fig3 / fig6。
    """
    stage = cfg.MODEL.TRAIN_STAGE

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = F.cross_entropy

    triplet = TripletLoss(getattr(cfg.SOLVER, 'MARGIN', 0.3))

    global_iter_counter = [0]  # 用列表包装以便在闭包中修改

    def loss_func(score, feat_list, target, text_bank_id=None, target_cloth=None):
        global_iter_counter[0] += 1
        cur_iter = global_iter_counter[0]

        # ========================================================
        # 阶段一：InfoNCE 对比损失（Batch 内多正样本软标签）
        # ========================================================
        if stage == 1:
            global_feat, t_id, t_cloth = feat_list

            v_norm = F.normalize(global_feat, p=2, dim=1)
            t_id_norm = F.normalize(t_id, p=2, dim=1)
            t_cloth_norm = F.normalize(t_cloth, p=2, dim=1)

            logit_scale = 20.0

            # -- 身份 InfoNCE（双向）--
            logits_i2t_id = logit_scale * v_norm @ t_id_norm.t()
            logits_t2i_id = logit_scale * t_id_norm @ v_norm.t()

            labels_id = (target.unsqueeze(1) == target.unsqueeze(0)).float().to(target.device)
            labels_id = labels_id / labels_id.sum(dim=1, keepdim=True)

            loss_i2t_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_i2t_id, dim=1), dim=1))
            loss_t2i_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_t2i_id, dim=1), dim=1))

            # -- 衣着 InfoNCE（双向）--
            logits_i2t_cloth = logit_scale * v_norm @ t_cloth_norm.t()
            logits_t2i_cloth = logit_scale * t_cloth_norm @ v_norm.t()

            cloth_labels_tensor = target_cloth if target_cloth is not None else target
            labels_cloth = (cloth_labels_tensor.unsqueeze(1) == cloth_labels_tensor.unsqueeze(0)).float().to(target.device)
            labels_cloth = labels_cloth / labels_cloth.sum(dim=1, keepdim=True)

            loss_i2t_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_i2t_cloth, dim=1), dim=1))
            loss_t2i_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_t2i_cloth, dim=1), dim=1))

            total_loss = loss_i2t_id + loss_t2i_id + loss_i2t_cloth + loss_t2i_cloth

            # -- 健康探针（每 50 步打印 + 写 TensorBoard）--
            if cur_iter % 50 == 0:
                sim_matrix = v_norm @ t_id_norm.t()
                pos_mask = labels_id.bool()
                pos_sim = sim_matrix[pos_mask].mean().item()
                neg_sim = sim_matrix[~pos_mask].mean().item()

                print(f"\n[Stage 1 脉搏探针] Iter {cur_iter} | Total Loss: {total_loss.item():.4f}")
                print(f"-> ID 损失: {loss_i2t_id.item():.3f} | Cloth 损失: {loss_i2t_cloth.item():.3f}")
                print(f"-> 余弦相似度 | 正样本(→1): {pos_sim:.4f} | 负样本(→0): {neg_sim:.4f}")

                health = ("🟩 健康" if pos_sim > neg_sim + 0.05
                          else ("🟨 挣扎" if pos_sim > neg_sim else "🟥 坍塌"))
                print(f"-> 提示词状态: {health}\n")

                # 写入 TensorBoard（供 visualize_training.py fig2 使用）
                if tb_writer is not None:
                    tb_writer.add_scalar("Train/Stage1_PosSim", pos_sim, cur_iter)
                    tb_writer.add_scalar("Train/Stage1_NegSim", neg_sim, cur_iter)

            return total_loss, None

        # ========================================================
        # 阶段二：MIPL 外科手术解耦 + 视觉微调
        # ========================================================
        elif stage == 2:
            global_feat, f_img2clo, t_cloth_gt = feat_list
            # 注意：t_cloth_gt 在 make_model.py 中已经 .detach()，此处无需重复

            L_ce = xent(score, target)
            L_tri = triplet(global_feat, target)[0]

            if text_bank_id is None:
                raise ValueError("Stage 2 必须传入固化的 text_bank_id！")

            # L_Guide：图像特征与全局固化身份文本的跨熵引导损失
            v_norm = F.normalize(global_feat, p=2, dim=1)
            t_id_bank_norm = F.normalize(text_bank_id, p=2, dim=1)
            logits_guide = 100.0 * v_norm @ t_id_bank_norm.t()
            L_Guide = xent(logits_guide, target)

            # L_sc：语义剥离一致性损失（衣服投影特征与衣服文本特征的 MSE）
            f_img2clo_norm = F.normalize(f_img2clo, p=2, dim=1)
            t_cloth_gt_norm = F.normalize(t_cloth_gt, p=2, dim=1)
            L_sc = F.mse_loss(f_img2clo_norm, t_cloth_gt_norm) * 10.0

            # L_de：截断余弦正交切除（ReLU 保证夹角>=90度时不再惩罚）
            cos_sim_ortho = torch.sum(v_norm * f_img2clo_norm.detach(), dim=1)
            L_de = torch.mean(F.relu(cos_sim_ortho))

            w_id = cfg.MODEL.ID_LOSS_WEIGHT
            w_tri = cfg.MODEL.TRIPLET_LOSS_WEIGHT
            w_guide = cfg.MODEL.I2T_ID_WEIGHT
            w_sc = cfg.MODEL.I2T_CLOTH_SC_WEIGHT
            w_ortho = cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT

            total_loss = w_id * L_ce + w_tri * L_tri + w_guide * L_Guide + w_sc * L_sc + w_ortho * L_de

            # 写入各分量到 TensorBoard（供 visualize_training.py fig3/fig6 使用）
            if tb_writer is not None and cur_iter % 10 == 0:
                tb_writer.add_scalar("Train/Stage2_L_ce",     L_ce.item(),     cur_iter)
                tb_writer.add_scalar("Train/Stage2_L_tri",    L_tri.item(),    cur_iter)
                tb_writer.add_scalar("Train/Stage2_L_Guide",  L_Guide.item(),  cur_iter)
                tb_writer.add_scalar("Train/Stage2_L_sc",     L_sc.item(),     cur_iter)
                tb_writer.add_scalar("Train/Stage2_L_de",     L_de.item(),     cur_iter)

            if cur_iter % 100 == 0:
                print(f"\n[Stage 2] MIPL 手术刀体检探针 (Iter {cur_iter})")
                print(f"-> L_sc (语义剥离): {L_sc.item():.4f}")
                print(f"-> L_de (正交切除, Max=1): {L_de.item():.4f}")
                print(f"-> L_ce: {L_ce.item():.3f} | L_tri: {L_tri.item():.3f} | L_Guide: {L_Guide.item():.3f}")
                print(f"-> Total: {total_loss.item():.4f}\n")

            return total_loss, None

    return loss_func