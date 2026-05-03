import torch
import torch.nn as nn
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
# 将 Qwen 的硬文本与可学习的 M=4 软提示进行拼接
# =========================================================================
class HybridPromptLearner(nn.Module):
    def __init__(self, cfg, num_classes, clip_model):
        super().__init__()
        self.M = cfg.MODEL.PROMPT_LENGTH
        self.num_classes = num_classes
        self.num_clothes = cfg.MODEL.NUM_CLOTHES  # 由 Dataloader 赋值后传入
        
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        
        # 初始化可学习的身份 [X]^p 和 衣服 [X]^c 张量
        # 维度: [数量, M, 512]
        self.ctx_id = nn.Parameter(torch.empty(self.num_classes, self.M, ctx_dim))
        self.ctx_cloth = nn.Parameter(torch.empty(self.num_clothes, self.M, ctx_dim))
        
        nn.init.normal_(self.ctx_id, std=0.02)
        nn.init.normal_(self.ctx_cloth, std=0.02)
        
        self.token_embedding = clip_model.token_embedding

    def forward(self, id_label, cloth_label, id_text, cloth_text):
        # 获取当前 batch 的可学习上下文 [B, M, 512]
        ctx_id_batch = self.ctx_id[id_label]
        ctx_cloth_batch = self.ctx_cloth[cloth_label]
        
        # 文本 token 化并获取原始 embedding [B, 77, 512]
        id_tokens = clip.tokenize(id_text).to(ctx_id_batch.device)
        cloth_tokens = clip.tokenize(cloth_text).to(ctx_cloth_batch.device)
        
        id_embedding = self.token_embedding(id_tokens).type(ctx_id_batch.dtype)
        cloth_embedding = self.token_embedding(cloth_tokens).type(ctx_cloth_batch.dtype)
        
        # 将 [X] 软提示插入到句子开头 (SOT Token 之后)
        # 结构: [SOT] + [X]1..M + [Qwen Macro Text] + [EOT]
        prefix_id = id_embedding[:, :1, :]
        suffix_id = id_embedding[:, 1 + self.M :, :]
        prompts_id = torch.cat([prefix_id, ctx_id_batch, suffix_id], dim=1)
        
        prefix_cloth = cloth_embedding[:, :1, :]
        suffix_cloth = cloth_embedding[:, 1 + self.M :, :]
        prompts_cloth = torch.cat([prefix_cloth, ctx_cloth_batch, suffix_cloth], dim=1)
        
        return prompts_id, prompts_cloth, id_tokens, cloth_tokens

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

        self.clip_model = clip.build_model(state_dict or model.state_dict(), grid_h, grid_w, cfg.MODEL.STRIDE_SIZE[0])
        self.image_encoder = self.clip_model.visual

        # 构建混合提示学习器
        self.prompt_learner = HybridPromptLearner(cfg, num_classes, self.clip_model)
        
        # 构建 Stage 2 的 MIPL 外科手术投影层 (Linear + BN)
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
        """核心：根据阶段严格控制梯度"""
        if self.stage == 1:
            # Stage 1: 冻结 clip_model 全部参数（含 token_embedding/positional_embedding/ln_final/text_projection）
            # 同时冻结 Stage 2 专属层，仅放开可学习 Prompt 向量
            for param in self.clip_model.parameters(): param.requires_grad = False
            for param in self.classifier.parameters(): param.requires_grad = False
            for param in self.bottleneck.parameters(): param.requires_grad = False
            for param in self.cloth_proj.parameters(): param.requires_grad = False
            self.prompt_learner.ctx_id.requires_grad = True
            self.prompt_learner.ctx_cloth.requires_grad = True

        elif self.stage == 2:
            # Stage 2: 冻结 clip_model 全部参数（完整锁死文本侧），放开视觉骨干和各投影/分类层
            for param in self.clip_model.parameters(): param.requires_grad = False
            self.prompt_learner.ctx_id.requires_grad = False
            self.prompt_learner.ctx_cloth.requires_grad = False
            # image_encoder 是 clip_model.visual 的引用，单独解冻视觉参数
            for param in self.image_encoder.parameters(): param.requires_grad = True
            for param in self.cloth_proj.parameters(): param.requires_grad = True
            for param in self.classifier.parameters(): param.requires_grad = True
            for param in self.bottleneck.parameters(): param.requires_grad = True

    def text_encoder_forward(self, prompts, tokenized_prompts):
        x = prompts + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        # 获取 [EOT] 处的特征进行投影
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
        return x

    def forward(self, x, label=None, cloth_label=None, id_text=None, cloth_text=None):
        _, _, image_features_proj = self.image_encoder(x, None)
        global_feat = image_features_proj[:, 0]
        
        if not self.training:
            return self.bottleneck(global_feat) if self.cfg.MODEL.NECK_FEAT == 'after' else global_feat

        # ========================================================
        # 阶段一：混合提示学习 (Prompt Learning)
        # ========================================================
        if self.stage == 1:
            prompts_id, prompts_cloth, tk_id, tk_cloth = self.prompt_learner(label, cloth_label, id_text, cloth_text)
            
            t_id = self.text_encoder_forward(prompts_id, tk_id)
            t_cloth = self.text_encoder_forward(prompts_cloth, tk_cloth)
            
            return [global_feat, t_id, t_cloth]

        # ========================================================
        # 阶段二：视觉特征解耦与微调 (Visual Feature Disentanglement)
        # ========================================================
        elif self.stage == 2:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            
            # 步骤 A：提取当前图片的专属衣服映射特征 F_img2clo
            f_img2clo = self.cloth_proj(global_feat)
            
            # 同样获取当前 Batch 的真实文本特征用于 L_sc 监督
            prompts_id, prompts_cloth, tk_id, tk_cloth = self.prompt_learner(label, cloth_label, id_text, cloth_text)
            t_cloth_gt = self.text_encoder_forward(prompts_cloth, tk_cloth)
            
            return cls_score, [global_feat, f_img2clo, t_cloth_gt]

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'ViT-B-16':
        return build_transformer(num_class, camera_num, view_num, cfg)
    raise NotImplementedError()