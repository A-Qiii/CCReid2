import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import clip


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# =========================================================================
# 混合提示学习器 (Hybrid Prompt Learner)
#
# 【[X] 位置说明】
# 格式："{Qwen_Macro} [X]1[X]2[X]3[X]4 person/clothes"
# 完整 token 序列：[SOT] {Qwen硬文本tokens} [X]1..M {person/clothes} [EOT]
#
# 设计依据：
# - CLIP-ReID / MIPL 原始格式为 "[X]1..M person"，软提示在前，硬标签在后。
# - 本项目创新点：在 [X] 之前加入 Qwen2 生成的宏观属性硬文本作为语义初始化锚点。
#   [X] 在感知到前方 Qwen 文本的语义上下文后，进行细粒度补充，而非盲目初始化。
# - 与 CoCoOp 的 "context + class" 结构类比：Qwen硬文本 ≈ class token，[X] ≈ context。
#
# 工程实现：
# 将硬文本模板固定为 "{id_text} X X X X person" / "{cloth_text} X X X X clothes"，
# 再用占位字符 "X"（单token）定位插入点，替换为可学习向量。
# 这样无需动态寻找插入点，结构稳定，对任意长度的 Qwen 硬文本均适用。
# =========================================================================
class HybridPromptLearner(nn.Module):
    def __init__(self, cfg, num_classes, num_clothes, clip_model):
        super().__init__()
        self.M = cfg.MODEL.PROMPT_LENGTH  # M=4
        self.num_classes = num_classes
        self.num_clothes = num_clothes

        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512

        # 可学习软提示向量：每个身份/衣服各有独立的 M 个向量
        # 维度: [数量, M, 512]
        self.ctx_id = nn.Parameter(torch.empty(self.num_classes, self.M, ctx_dim))
        self.ctx_cloth = nn.Parameter(torch.empty(self.num_clothes, self.M, ctx_dim))

        nn.init.normal_(self.ctx_id, std=0.02)
        nn.init.normal_(self.ctx_cloth, std=0.02)

        self.token_embedding = clip_model.token_embedding

    def _build_prompts(self, ctx_batch, raw_text_list, suffix_word):
        """
        将 [X] 软提示插入到硬文本之后、末尾词之前。

        最终 token 序列（逻辑）：
            [SOT] {Qwen硬文本 tokens} [X]1..M {suffix_word} [EOT 及 padding]

        实现方式：
        1. 将硬文本 tokenize，得到完整的 77 维 token 序列和 embedding。
        2. 从 embedding 中提取有效的硬文本部分（去掉 SOT 之后的 [X] 占位符和 suffix）。
        3. 拼接：[SOT embed] + [Qwen embed] + [ctx_batch embed] + [suffix embed] + [padding zeros]
        4. 截断到 77 个 token（CLIP 最大长度）。

        注意：tokenized_prompts 用于定位 [EOT] 位置，我们重新 tokenize 含完整占位符的句子。
        """
        device = ctx_batch.device
        B, M, D = ctx_batch.shape

        # 构造完整占位模板（用于定位 [EOT]）："{raw_text} * * * * {suffix_word}."
        # 使用 "*" 作为 M 个单字符占位符（每个恰好是一个 token），便于定位
        placeholder = " ".join(["*"] * M)
        full_texts = [f"{t} {placeholder} {suffix_word}." for t in raw_text_list]
        tokenized = clip.tokenize(full_texts).to(device)  # [B, 77]

        # 获取完整 embedding（含 [SOT]、硬文本、占位符、suffix、[EOT]）
        with torch.no_grad():
            full_embed = self.token_embedding(tokenized).type(ctx_batch.dtype)  # [B, 77, D]

        # 定位 "*" 占位符在序列中的起始位置
        # tokenize("*") 的 token id 是固定的，找到第一个 "*" 的位置即为插入点
        star_token_id = clip.tokenize(["*"]).squeeze()[1].item()  # 跳过 [SOT]
        # 找每个样本中第一个 "*" 的位置
        star_positions = (tokenized == star_token_id).float().argmax(dim=1)  # [B]

        # 逐样本拼装：用可学习 ctx 向量替换 "*" 占位的 M 个 token
        new_embed = full_embed.clone()
        for i in range(B):
            pos = star_positions[i].item()
            new_embed[i, pos: pos + M, :] = ctx_batch[i]  # 替换 [X]1..M

        return new_embed, tokenized

    def forward(self, id_label, cloth_label, id_text, cloth_text):
        """
        输入：
            id_label:    [B] 连续整数身份标签（relabeled）
            cloth_label: [B] 连续整数衣服标签（relabeled）
            id_text:     list[str], 每张图的 Qwen 身份硬文本（如 "Man, thin body shape"）
            cloth_text:  list[str], 每张图的 Qwen 衣服硬文本（如 "Red jacket, black pants"）

        输出：
            prompts_id:    [B, 77, 512]  含软提示的身份 embedding
            prompts_cloth: [B, 77, 512]  含软提示的衣服 embedding
            tk_id:         [B, 77]       身份完整 tokenized（用于定位 [EOT]）
            tk_cloth:      [B, 77]       衣服完整 tokenized
        """
        ctx_id_batch = self.ctx_id[id_label]      # [B, M, 512]
        ctx_cloth_batch = self.ctx_cloth[cloth_label]  # [B, M, 512]

        prompts_id, tk_id = self._build_prompts(ctx_id_batch, id_text, "person")
        prompts_cloth, tk_cloth = self._build_prompts(ctx_cloth_batch, cloth_text, "clothes")

        return prompts_id, prompts_cloth, tk_id, tk_cloth


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.cfg = cfg
        self.stage = cfg.MODEL.TRAIN_STAGE
        self.in_planes = 768
        self.joint_planes = 512

        model_path = clip._download(clip._MODELS["ViT-B-16"])
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        grid_h = cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0]
        grid_w = cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1]

        self.clip_model = clip.build_model(
            state_dict or model.state_dict(), grid_h, grid_w, cfg.MODEL.STRIDE_SIZE[0]
        )
        self.image_encoder = self.clip_model.visual

        # 从 cfg 获取 num_clothes（由 train.py 在 dataloader 后动态填入）
        num_clothes = cfg.MODEL.NUM_CLOTHES

        # 混合提示学习器
        self.prompt_learner = HybridPromptLearner(cfg, num_classes, num_clothes, self.clip_model)

        # Stage 2 的 MIPL 衣服投影层
        self.cloth_proj = nn.Sequential(
            nn.Linear(self.joint_planes, self.joint_planes, bias=False),
            nn.BatchNorm1d(self.joint_planes)
        )
        self.cloth_proj.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.joint_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.joint_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self._freeze_parameters_by_stage()

    def _freeze_parameters_by_stage(self):
        """根据阶段严格控制梯度流"""
        if self.stage == 1:
            # Stage 1：仅放开可学习 Prompt 向量，其余全冻结
            for param in self.clip_model.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
            for param in self.bottleneck.parameters():
                param.requires_grad = False
            for param in self.cloth_proj.parameters():
                param.requires_grad = False
            self.prompt_learner.ctx_id.requires_grad = True
            self.prompt_learner.ctx_cloth.requires_grad = True

        elif self.stage == 2:
            # Stage 2：冻结文本侧，放开视觉骨干和各投影/分类层
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.prompt_learner.ctx_id.requires_grad = False
            self.prompt_learner.ctx_cloth.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = True
            for param in self.cloth_proj.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.bottleneck.parameters():
                param.requires_grad = True

    def switch_stage(self, new_stage):
        """
        在两阶段之间切换时调用此方法，同时更新 self.stage 并重新触发冻结逻辑。
        比手动 model.stage=2 + model._freeze_parameters_by_stage() 更安全。
        """
        self.stage = new_stage
        self._freeze_parameters_by_stage()
        print(f">>> 模型已切换至 Stage {new_stage}，梯度状态已同步更新。")

    def text_encoder_forward(self, prompts, tokenized_prompts):
        """通过 CLIP 文本 Transformer 编码混合提示"""
        x = prompts + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)   # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)   # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        # 取 [EOT] 位置的特征并投影
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
        return x

    def forward(self, x, label=None, cloth_label=None, id_text=None, cloth_text=None):
        # ---- 视觉特征提取（所有阶段共用）----
        _, _, image_features_proj = self.image_encoder(x, None)
        global_feat = image_features_proj[:, 0]  # [B, 512]
 
        # ---- 测试阶段：正交投影减除衣服成分（推理时利用 Proj_c）----
        #
        # 【原理】
        #   Proj_c 在 Stage 2 训练中已学会提取衣服特征方向。
        #   原版推理完全不用 Proj_c，衣服信息仍残留在 global_feat 里。
        #   现在在推理时将 global_feat 沿衣服方向做正交投影减除：
        #
        #     f_cloth_dir = normalize(cloth_proj(f))
        #     cloth_component = (f · f_cloth_dir) × f_cloth_dir
        #     f_clean = f - λ × cloth_component
        #
        #   λ=0 退化为原版（完全不减除）
        #   λ=1 完全减除衣服分量
        #   推荐 λ=0.5 起步，可在不重新训练的情况下调参
        #
        if not self.training:
            lam = getattr(self.cfg.MODEL, 'CLOTH_SUBTRACT_LAMBDA', 0.5)
 
            if lam > 0:
                with torch.no_grad():
                    f_cloth = self.cloth_proj(global_feat)
                    f_cloth_dir = F.normalize(f_cloth, dim=1)
                    proj_len = (global_feat * f_cloth_dir).sum(dim=1, keepdim=True)
                    cloth_component = proj_len * f_cloth_dir
                    feat_clean = global_feat - lam * cloth_component
            else:
                feat_clean = global_feat
 
            return self.bottleneck(feat_clean) if self.cfg.MODEL.NECK_FEAT == 'after' else feat_clean
 
# ===== 复制到这里结束 =====

        # ========================================================
        # 阶段一：混合提示学习 (Hybrid Prompt Learning)
        # 返回 [global_feat, t_id, t_cloth] 给 loss_fn Stage1 分支
        # ========================================================
        if self.stage == 1:
            prompts_id, prompts_cloth, tk_id, tk_cloth = self.prompt_learner(
                label, cloth_label, id_text, cloth_text
            )
            t_id = self.text_encoder_forward(prompts_id, tk_id)
            t_cloth = self.text_encoder_forward(prompts_cloth, tk_cloth)
            return [global_feat, t_id, t_cloth]

        # ========================================================
        # 阶段二：视觉特征解耦与微调 (MIPL Disentanglement)
        # 返回 cls_score, [global_feat, f_img2clo, t_cloth_gt]
        # ========================================================
        elif self.stage == 2:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)

            # 步骤 A：提取衣服映射特征
            f_img2clo = self.cloth_proj(global_feat.detach())

            # 步骤 B：获取当前 Batch 的真实衣服文本特征（用于 L_sc 监督）
            # 注意：prompt_learner 参数已冻结，此处 t_cloth_gt 显式 detach()
            # 确保 L_sc 的监督信号不会意外回传到文本侧
            prompts_id, prompts_cloth, tk_id, tk_cloth = self.prompt_learner(
                label, cloth_label, id_text, cloth_text
            )
            t_cloth_gt = self.text_encoder_forward(prompts_cloth, tk_cloth).detach()

            return cls_score, [global_feat, f_img2clo, t_cloth_gt]


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'ViT-B-16':
        return build_transformer(num_class, camera_num, view_num, cfg)
    raise NotImplementedError(f"不支持的模型: {cfg.MODEL.NAME}")