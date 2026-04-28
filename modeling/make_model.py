import torch
import torch.nn as nn
from .clip import clip

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
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

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.in_planes = 768
        self.joint_planes = 512

        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_camera = cfg.MODEL.SIE_CAMERA
        self.sie_view = cfg.MODEL.SIE_VIEW
        self.neck_feat = cfg.MODEL.NECK_FEAT

        # 载入 CLIP 权重
        model_path = clip._download(clip._MODELS["ViT-B-16"])
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        grid_h = cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0]
        grid_w = cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1]

        clip_model = clip.build_model(
            state_dict or model.state_dict(),
            grid_h,
            grid_w,
            cfg.MODEL.STRIDE_SIZE[0]
        )

        self.clip_model = clip_model
        self.image_encoder = clip_model.visual

        # 冻结文本流参数
        for name, param in self.clip_model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False

        # SIE 辅助信息嵌入层初始化
        if self.sie_camera and self.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
        elif self.sie_camera:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        elif self.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
        else:
            self.cv_embed = None

        if self.cv_embed is not None:
            torch.nn.init.normal_(self.cv_embed, std=1e-6)

        self.classifier = nn.Linear(self.joint_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.joint_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None, id_text=None, cloth_text=None):
        # 处理辅助信息嵌入 (SIE) 
        cv_embed = None
        if self.sie_camera and self.sie_view and cam_label is not None and view_label is not None:
            cv_embed = self.cv_embed[cam_label * self.view_num + view_label]
        elif self.sie_camera and cam_label is not None:
            cv_embed = self.cv_embed[cam_label]
        elif self.sie_view and view_label is not None:
            cv_embed = self.cv_embed[view_label]

        # 视觉骨干网络提取 
        _, _, image_features_proj = self.image_encoder(x, cv_embed)
        global_feat = image_features_proj[:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            # Tokenize 文本并移动到 GPU
            id_tokens = clip.tokenize(id_text).to(x.device)
            cloth_tokens = clip.tokenize(cloth_text).to(x.device)

            # 提取文本语义特征 
            id_feat = self.clip_model.encode_text(id_tokens)
            cloth_feat = self.clip_model.encode_text(cloth_tokens)

            # L2 归一化至高维超球面 
            img_norm = global_feat / global_feat.norm(dim=-1, keepdim=True)
            id_feat_norm = id_feat / id_feat.norm(dim=-1, keepdim=True)
            cloth_feat_norm = cloth_feat / cloth_feat.norm(dim=-1, keepdim=True)

            # 计算跨模态点积矩阵
            scale = self.clip_model.logit_scale.exp()
            score_i2t_id = scale * img_norm @ id_feat_norm.t()
            score_i2t_cloth = scale * img_norm @ cloth_feat_norm.t()

            cls_score = self.classifier(feat)
            
            # 返回分类得分和用于损失计算的特征包
            return cls_score, [global_feat, score_i2t_id, score_i2t_cloth]
        else:
            # 测试阶段仅返回视觉特征
            return feat if self.neck_feat == 'after' else global_feat

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'ViT-B-16':
        return build_transformer(num_class, camera_num, view_num, cfg)
    raise NotImplementedError("目前配置仅支持 ViT-B-16")