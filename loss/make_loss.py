import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes, stage, tb_writer=None):
    """
    损失函数工厂。

    【改进说明 v2】
    在原有 Stage 2 损失基础上新增两项改进：

    改进①：置信度门控渐进解耦（Confidence-Gated Progressive Decoupling）
      - 原问题：L_de 在 Proj_c 未收敛时就开始推开 F_ori，误伤身份特征
      - 方案：用 w_i = sigmoid(α*(s_i - τ)) 为每个样本计算门控值，
              其中 s_i = cos(F_img2clo, t_cloth_gt)，衡量 Proj_c 的提取质量。
              w 截断梯度（detach），仅作控制信号，不参与反向传播。
      - 超参：τ=0.3, α=5.0，可在 cfg 中覆盖：
              MODEL.ORTHO_GATE_TAU / MODEL.ORTHO_GATE_ALPHA

    改进②：衣服三元组损失 L_cloth_tri（破除 Proj_c 模式坍塌）
      - 原问题：仅 L_sc 监督 Proj_c，无法防止 F_img2clo 退化为常数向量
      - 方案：用衣服标签 cloth_label 对 F_img2clo 施加 TripletLoss，
              强迫 F_img2clo 具备"区分不同衣服"的判别性。
      - 权重：I2T_CLOTH_TRI_WEIGHT（默认 0.2）

    【接口不变】
      - loss_func 签名与原版完全一致，新增 target_cloth 参数（Stage 2 中传入）
      - 调用方（processor.py）仅需在 Stage 2 时额外传 target_cloth=cloth_id

    【总损失公式】
      L_total = w_id*L_ce + w_tri*L_tri + w_guide*L_Guide
              + w_sc*L_sc + w_ct*L_cloth_tri + w_ortho*L_de_gated

    参数：
        cfg:        配置对象
        num_classes:分类数（ID 分类头的类别数）
        stage:      显式指定当前阶段（1 或 2），不再从 cfg 动态读取
        tb_writer:  可选 TensorBoard SummaryWriter
    """
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent = F.cross_entropy

    triplet = TripletLoss(getattr(cfg.SOLVER, 'MARGIN', 0.3))
    cur_iter = [0]

    # ── 门控超参（支持 cfg 覆盖，不存在则使用默认值）──────────────────
    GATE_TAU   = getattr(cfg.MODEL, 'ORTHO_GATE_TAU',   0.3)   # 置信度阈值
    GATE_ALPHA = getattr(cfg.MODEL, 'ORTHO_GATE_ALPHA', 5.0)   # sigmoid 温度

    def loss_func(score, feat_list, target, text_bank_id=None, target_cloth=None):
        cur_iter[0] += 1
        n = cur_iter[0]

        # ============================================================
        # Stage 1：InfoNCE 对比损失（不变）
        # ============================================================
        if stage == 1:
            global_feat, t_id, t_cloth = feat_list

            v_norm       = F.normalize(global_feat, p=2, dim=1)
            t_id_norm    = F.normalize(t_id,        p=2, dim=1)
            t_cloth_norm = F.normalize(t_cloth,     p=2, dim=1)

            logit_scale = 20.0

            # 身份 InfoNCE（双向）
            logits_i2t_id  = logit_scale * v_norm      @ t_id_norm.t()
            logits_t2i_id  = logit_scale * t_id_norm   @ v_norm.t()
            labels_id = (target.unsqueeze(1) == target.unsqueeze(0)).float().to(target.device)
            labels_id = labels_id / labels_id.sum(dim=1, keepdim=True)
            loss_i2t_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_i2t_id, dim=1), dim=1))
            loss_t2i_id = -torch.mean(torch.sum(labels_id * F.log_softmax(logits_t2i_id, dim=1), dim=1))

            # 衣着 InfoNCE（双向）
            logits_i2t_cloth = logit_scale * v_norm      @ t_cloth_norm.t()
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
        # Stage 2：MIPL 外科手术解耦 + 视觉微调（含两项改进）
        # ============================================================
        elif stage == 2:
            global_feat, f_img2clo, t_cloth_gt = feat_list

            # ── 基础损失 ──────────────────────────────────────────────
            L_ce  = xent(score, target)
            L_tri = triplet(global_feat, target)[0]

            # ── 图文身份引导损失 L_Guide ──────────────────────────────
            if text_bank_id is None:
                raise ValueError("Stage 2 必须传入固化的 text_bank_id！")

            v_norm         = F.normalize(global_feat,  p=2, dim=1)
            t_id_bank_norm = F.normalize(text_bank_id, p=2, dim=1)
            logits_guide   = 100.0 * v_norm @ t_id_bank_norm.t()
            L_Guide        = xent(logits_guide, target)

            # ── 归一化衣服特征 ────────────────────────────────────────
            f_img2clo_norm  = F.normalize(f_img2clo,  p=2, dim=1)
            t_cloth_gt_norm = F.normalize(t_cloth_gt, p=2, dim=1)

            # ── 监督① L_sc：语义剥离校验（不变）─────────────────────
            L_sc = F.mse_loss(f_img2clo_norm, t_cloth_gt_norm) * 10.0

            # ── 监督②【改进】L_cloth_tri：衣服三元组（破除模式坍塌）──
            # 仅在 target_cloth 可用时计算；否则退化为零，不中断训练
            if target_cloth is not None:
                L_cloth_tri = triplet(f_img2clo, target_cloth)[0]
            else:
                L_cloth_tri = torch.tensor(0.0, device=global_feat.device)

            # ── 解耦损失【改进】：置信度门控截断余弦正交 ─────────────
            #
            # s_i = cos(f_img2clo_i, t_cloth_gt_i)  ∈ [-1, 1]
            #   反映 Proj_c 在样本 i 上的提取质量
            #
            # w_i = sigmoid(α * (s_i - τ))
            #   当 s_i < τ（提取不可信）→ w_i ≈ 0，L_de 自动失效
            #   当 s_i > τ（提取可信）  → w_i ≈ 1，L_de 恢复全强度
            #
            # w 截断梯度：防止 Proj_c 通过让 w 变小来规避惩罚
            #
            s_i = torch.sum(f_img2clo_norm * t_cloth_gt_norm.detach(), dim=1)  # [B]
            w_i = torch.sigmoid(GATE_ALPHA * (s_i - GATE_TAU))                 # [B], ∈(0,1)
            w_i = w_i.detach()                                                  # 截断梯度

            cos_ortho = torch.sum(v_norm * f_img2clo_norm.detach(), dim=1)     # [B]
            L_de = torch.mean(w_i * F.relu(cos_ortho))                         # 加权截断

            # ── 读取损失权重 ──────────────────────────────────────────
            w_id     = cfg.MODEL.ID_LOSS_WEIGHT
            w_tri    = cfg.MODEL.TRIPLET_LOSS_WEIGHT
            w_guide  = cfg.MODEL.I2T_ID_WEIGHT
            w_sc     = cfg.MODEL.I2T_CLOTH_SC_WEIGHT
            w_ct     = getattr(cfg.MODEL, 'I2T_CLOTH_TRI_WEIGHT', 0.2)   # 新增权重
            w_ortho  = cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT

            # ── 总损失 ────────────────────────────────────────────────
            total_loss = (w_id    * L_ce
                        + w_tri   * L_tri
                        + w_guide * L_Guide
                        + w_sc    * L_sc
                        + w_ct    * L_cloth_tri
                        + w_ortho * L_de)

            # ── TensorBoard 监控 ──────────────────────────────────────
            if tb_writer is not None and n % 10 == 0:
                tb_writer.add_scalar("Train/Stage2_TotalLoss",   total_loss.item(),   n)
                tb_writer.add_scalar("Train/Stage2_L_ce",        L_ce.item(),         n)
                tb_writer.add_scalar("Train/Stage2_L_tri",       L_tri.item(),        n)
                tb_writer.add_scalar("Train/Stage2_L_Guide",     L_Guide.item(),      n)
                tb_writer.add_scalar("Train/Stage2_L_sc",        L_sc.item(),         n)
                tb_writer.add_scalar("Train/Stage2_L_cloth_tri", L_cloth_tri.item(),  n)
                tb_writer.add_scalar("Train/Stage2_L_de",        L_de.item(),         n)
                # ── 门控质量监控探针 ──────────────────────────────────
                # w_mean：当前 batch 的平均置信度，应从低到高单调上升
                # s_mean：Proj_c 的平均余弦对齐度，越高说明衣服提取越准
                tb_writer.add_scalar("ProjC/gate_w_mean",        w_i.mean().item(),   n)
                tb_writer.add_scalar("ProjC/cloth_align_s_mean", s_i.mean().item(),   n)
                tb_writer.add_scalar("Train/Stage2_BaseAcc",
                                     (score.max(1)[1] == target).float().mean().item(), n)

            # ── 同衣聚合度 / 异衣分离度探针（每 50 iter）────────────
            if tb_writer is not None and n % 50 == 0 and target_cloth is not None:
                same_cloth = (target_cloth.unsqueeze(1) == target_cloth.unsqueeze(0))  # [B,B]
                same_cloth.fill_diagonal_(False)
                sim_mat = f_img2clo_norm @ f_img2clo_norm.t()                          # [B,B]

                intra_sim = sim_mat[same_cloth].mean().item() if same_cloth.any() else 0.0
                inter_sim = sim_mat[~same_cloth].mean().item() if (~same_cloth).any() else 0.0
                tb_writer.add_scalar("ProjC/intra_cloth_sim", intra_sim, n)
                tb_writer.add_scalar("ProjC/inter_cloth_sim", inter_sim, n)

            if n % 100 == 0:
                print(f"\n[Stage 2] Iter {n} | Total: {total_loss.item():.4f} | "
                      f"L_ce: {L_ce.item():.3f} L_tri: {L_tri.item():.3f} "
                      f"L_Guide: {L_Guide.item():.3f} L_sc: {L_sc.item():.4f} "
                      f"L_cloth_tri: {L_cloth_tri.item():.4f} "
                      f"L_de: {L_de.item():.4f} (w_ortho={w_ortho}) "
                      f"| gate_w={w_i.mean().item():.3f} s_align={s_i.mean().item():.3f}")

            return total_loss, None

    return loss_func