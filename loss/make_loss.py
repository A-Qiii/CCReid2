import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes, stage, tb_writer=None):
    """
    损失函数工厂 v3。

    【v3 修复说明】
    诊断发现 L_sc 存在严重的量级 Bug：
      F.mse_loss 对所有元素（包括512个维度）取均值，
      导致实际值 = (2 - 2·cos) / 512 ≈ 0.004，
      而非设计预期的 (2 - 2·cos) ≈ 2.0。
      乘以10后也只有0.04，L_cloth_tri（InfoNCE量级1~4）
      完全主导了Proj_c，使其只在视觉空间聚类而不向文本空间靠拢，
      导致 cloth_align_s_mean 始终约 0.05，门控永久失效。

    修复方案：
      将 L_sc 改为余弦距离损失 (1 - cos)，值域 [0, 2]，
      不依赖特征维度，量级稳定，与其他损失项匹配。

    同步调整：
      - L_cloth_tri 权重默认值从 0.2 降至 0.05，避免主导 Proj_c
      - GATE_TAU 从 0.3 降至 0.05，适配实际 cos 分布范围

    【改进点总结】
    改进①：置信度门控渐进解耦
      w_i = sigmoid(α*(s_i - τ))，s_i = cos(F_img2clo, t_cloth)
      w 截断梯度，仅作控制信号

    改进②：批内 InfoNCE 衣服对比损失
      替代 TripletLoss，支持任意 batch 衣服分布，不会 shape mismatch

    【总损失】
      L = w_id*L_ce + w_tri*L_tri + w_guide*L_Guide
        + w_sc*L_sc + w_ct*L_cloth_tri + w_ortho*L_de_gated
    """
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = F.cross_entropy

    triplet  = TripletLoss(getattr(cfg.SOLVER, 'MARGIN', 0.3))
    cur_iter = [0]

    # ── 门控超参 ────────────────────────────────────────────────────────
    # τ=0.05：诊断显示完整两阶段后 cos(F_img2clo,t_cloth)~0.05，
    #          τ=0.3 会让门控永远不激活，改为 0.05 让门控能及时介入
    GATE_TAU   = getattr(cfg.MODEL, 'ORTHO_GATE_TAU',   0.05)
    GATE_ALPHA = getattr(cfg.MODEL, 'ORTHO_GATE_ALPHA', 5.0)

    def loss_func(score, feat_list, target, text_bank_id=None, target_cloth=None):
        cur_iter[0] += 1
        n = cur_iter[0]

        # ============================================================
        # Stage 1：InfoNCE 对比损失
        # ============================================================
        if stage == 1:
            global_feat, t_id, t_cloth = feat_list

            v_norm       = F.normalize(global_feat, p=2, dim=1)
            t_id_norm    = F.normalize(t_id,        p=2, dim=1)
            t_cloth_norm = F.normalize(t_cloth,     p=2, dim=1)

            logit_scale = 20.0

            # 身份 InfoNCE（双向）
            logits_i2t_id = logit_scale * v_norm    @ t_id_norm.t()
            logits_t2i_id = logit_scale * t_id_norm @ v_norm.t()
            labels_id = (target.unsqueeze(1) == target.unsqueeze(0)).float().to(target.device)
            labels_id = labels_id / labels_id.sum(dim=1, keepdim=True)
            loss_i2t_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_i2t_id, dim=1), dim=1))
            loss_t2i_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_t2i_id, dim=1), dim=1))

            # 衣着 InfoNCE（双向）
            logits_i2t_cloth = logit_scale * v_norm      @ t_cloth_norm.t()
            logits_t2i_cloth = logit_scale * t_cloth_norm @ v_norm.t()
            cloth_labels_tensor = target_cloth if target_cloth is not None else target
            labels_cloth = (cloth_labels_tensor.unsqueeze(1) ==
                            cloth_labels_tensor.unsqueeze(0)).float().to(target.device)
            labels_cloth = labels_cloth / labels_cloth.sum(dim=1, keepdim=True)
            loss_i2t_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_i2t_cloth, dim=1), dim=1))
            loss_t2i_cloth = -torch.mean(torch.sum(labels_cloth * F.log_softmax(logits_t2i_cloth, dim=1), dim=1))

            total_loss = loss_i2t_id + loss_t2i_id + loss_i2t_cloth + loss_t2i_cloth

            if n % 50 == 0:
                sim_matrix = v_norm @ t_id_norm.t()
                pos_mask = labels_id.bool()
                pos_sim  = sim_matrix[pos_mask].mean().item()
                neg_sim  = sim_matrix[~pos_mask].mean().item()
                health = ("🟩 健康" if pos_sim > neg_sim + 0.05
                          else ("🟨 挣扎" if pos_sim > neg_sim else "🟥 坍塌"))
                print(f"\n[Stage 1 探针] Iter {n} | Loss: {total_loss.item():.4f} | "
                      f"pos: {pos_sim:.4f} neg: {neg_sim:.4f} | {health}")
                if tb_writer is not None:
                    tb_writer.add_scalar("Train/Stage1_PosSim", pos_sim, n)
                    tb_writer.add_scalar("Train/Stage1_NegSim", neg_sim, n)

            return total_loss, None

        # ============================================================
        # Stage 2：MIPL 解耦微调
        # ============================================================
        elif stage == 2:
            global_feat, f_img2clo, t_cloth_gt = feat_list

            # ── 基础损失 ──────────────────────────────────────────────
            L_ce  = xent(score, target)
            L_tri = triplet(global_feat, target)[0]

            # ── 身份引导损失 L_Guide ──────────────────────────────────
            if text_bank_id is None:
                raise ValueError("Stage 2 必须传入固化的 text_bank_id！")
            v_norm         = F.normalize(global_feat,  p=2, dim=1)
            t_id_bank_norm = F.normalize(text_bank_id, p=2, dim=1)
            logits_guide   = 100.0 * v_norm @ t_id_bank_norm.t()
            L_Guide        = xent(logits_guide, target)

            # ── 特征归一化 ────────────────────────────────────────────
            f_img2clo_norm  = F.normalize(f_img2clo,  p=2, dim=1)
            t_cloth_gt_norm = F.normalize(t_cloth_gt, p=2, dim=1)

            # ── 监督① L_sc：余弦对齐损失（v3 修复）─────────────────
            #
            # 【修复原因】原版 F.mse_loss 对512维取均值，实际值=(2-2cos)/512
            # 量级比预期小512倍，被 L_cloth_tri 完全压制，Proj_c 无法
            # 向文本空间靠拢，导致 cloth_align_s_mean 始终~0.05。
            #
            # 【新方案】直接用 1-cos 作为损失：
            #   值域 [0, 2]，cos→1 时趋近 0，不依赖特征维度。
            #   乘以 10 后量级约 10（训练初期），与 L_ce、L_Guide 匹配。
            #
            cos_align = (f_img2clo_norm * t_cloth_gt_norm.detach()).sum(dim=1)  # [B]
            L_sc = (1.0 - cos_align).mean() * 10.0

            

            # ── 监督② L_cloth_tri：批内 InfoNCE 衣服对比损失 ─────────
            #
            # 【为什么不用 TripletLoss】
            # hard_example_mining 要求每个标签样本数完全相同，
            # 衣服标签在随机 batch 里无法保证，会导致 shape mismatch。
            #
            # 【方案】软标签 InfoNCE，天然支持任意 batch 分布：
            #   同衣对 → 高相似度（拉近），异衣对 → 低相似度（推远）
            #
            # 【权重】默认 0.05（从 0.2 降低），避免主导 Proj_c 方向，
            # 让 L_sc 的文本锚定优先于视觉内部聚类。
            #
            if target_cloth is not None:
                logits_cloth   = f_img2clo_norm @ f_img2clo_norm.t() / 0.07  # [B,B]
                same_cloth_mat = (target_cloth.unsqueeze(1) ==
                                  target_cloth.unsqueeze(0)).float()           # [B,B]
                same_cloth_mat.fill_diagonal_(0)
                row_sum = same_cloth_mat.sum(dim=1, keepdim=True)
                has_pos = (row_sum.squeeze(1) > 0)
                if has_pos.any():
                    soft_label  = same_cloth_mat[has_pos] / row_sum[has_pos]
                    log_prob    = F.log_softmax(logits_cloth[has_pos], dim=1)
                    L_cloth_tri = -(soft_label * log_prob).sum(dim=1).mean()
                else:
                    L_cloth_tri = torch.tensor(0.0, device=global_feat.device)
            else:
                L_cloth_tri = torch.tensor(0.0, device=global_feat.device)

            # ── 监督③ L_de：置信度门控正交排斥 ──────────────────────
            #
            # s_i = cos(F_img2clo, t_cloth)，衡量 Proj_c 的提取质量
            # w_i = sigmoid(α*(s_i - τ))：
            #   s_i < τ（提取不可信）→ w_i≈0，L_de 自动抑制
            #   s_i > τ（提取可信）  → w_i≈1，L_de 全强度激活
            # w 截断梯度，仅作控制信号
            #
            # 【v3 调整】τ: 0.3→0.05
            # 诊断显示实际 cos~0.004~0.05，τ=0.3 导致门控永久失活。
            # τ=0.05 让门控在 Proj_c 开始对齐后即可逐步介入。
            #
            s_i       = (f_img2clo_norm * t_cloth_gt_norm.detach()).sum(dim=1)  # [B]
            w_i       = torch.sigmoid(GATE_ALPHA * (s_i - GATE_TAU)).detach()   # [B]
            cos_ortho = (v_norm * f_img2clo_norm.detach()).sum(dim=1)            # [B]
            L_de      = (w_i * F.relu(cos_ortho)).mean()

            # ── 损失权重 ──────────────────────────────────────────────
            w_id    = cfg.MODEL.ID_LOSS_WEIGHT
            w_tri   = cfg.MODEL.TRIPLET_LOSS_WEIGHT
            w_guide = cfg.MODEL.I2T_ID_WEIGHT
            w_sc    = cfg.MODEL.I2T_CLOTH_SC_WEIGHT
            w_ct    = getattr(cfg.MODEL, 'I2T_CLOTH_TRI_WEIGHT', 0.05)  # 默认0.05
            w_ortho = cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT

            # ── 总损失 ────────────────────────────────────────────────
            total_loss = (w_id    * L_ce
                        + w_tri   * L_tri
                        + w_guide * L_Guide
                        + w_sc    * L_sc
                        + w_ct    * L_cloth_tri
                        + w_ortho * L_de)

            # ── TensorBoard ───────────────────────────────────────────
            if tb_writer is not None and n % 10 == 0:
                tb_writer.add_scalar("Train/Stage2_TotalLoss",   total_loss.item(),  n)
                tb_writer.add_scalar("Train/Stage2_L_ce",        L_ce.item(),        n)
                tb_writer.add_scalar("Train/Stage2_L_tri",       L_tri.item(),       n)
                tb_writer.add_scalar("Train/Stage2_L_Guide",     L_Guide.item(),     n)
                tb_writer.add_scalar("Train/Stage2_L_sc",        L_sc.item(),        n)
                tb_writer.add_scalar("Train/Stage2_L_cloth_tri", L_cloth_tri.item(), n)
                tb_writer.add_scalar("Train/Stage2_L_de",        L_de.item(),        n)
                tb_writer.add_scalar("ProjC/gate_w_mean",        w_i.mean().item(),  n)
                tb_writer.add_scalar("ProjC/cloth_align_s_mean", s_i.mean().item(),  n)
                tb_writer.add_scalar("Train/Stage2_BaseAcc",
                                     (score.max(1)[1] == target).float().mean().item(), n)

            # ── 同衣/异衣相似度探针（每50iter）────────────────────────
            if tb_writer is not None and n % 50 == 0 and target_cloth is not None:
                same_cloth = (target_cloth.unsqueeze(1) == target_cloth.unsqueeze(0))
                same_cloth.fill_diagonal_(False)
                sim_mat   = f_img2clo_norm @ f_img2clo_norm.t()
                intra_sim = sim_mat[same_cloth].mean().item()  if same_cloth.any()  else 0.0
                inter_sim = sim_mat[~same_cloth].mean().item() if (~same_cloth).any() else 0.0
                tb_writer.add_scalar("ProjC/intra_cloth_sim", intra_sim, n)
                tb_writer.add_scalar("ProjC/inter_cloth_sim", inter_sim, n)

            if n % 100 == 0:
                print(f"\n[Stage 2] Iter {n} | Total: {total_loss.item():.4f} | "
                      f"L_ce:{L_ce.item():.3f} L_tri:{L_tri.item():.3f} "
                      f"L_Guide:{L_Guide.item():.3f} L_sc:{L_sc.item():.4f} "
                      f"L_clt:{L_cloth_tri.item():.4f} L_de:{L_de.item():.4f} "
                      f"| gate_w:{w_i.mean().item():.3f} s:{s_i.mean().item():.3f}")

            return total_loss, None

    return loss_func