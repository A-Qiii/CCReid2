from yacs.config import CfgNode as CN

cfg = CN()

# --- MODEL: 模型架构与语义四元力场 ---
cfg.MODEL = CN()
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'
cfg.MODEL.METRIC_LOSS_TYPE = 'triplet'
cfg.MODEL.IF_LABELSMOOTH = 'on'
cfg.MODEL.IF_WITH_CENTER = 'no'    # 严格锁死
cfg.MODEL.STRIDE_SIZE = [16, 16]
cfg.MODEL.SIE_CAMERA = False
cfg.MODEL.SIE_VIEW = False
cfg.MODEL.SIE_COE = 1.0

# ==========================================================
# 【两阶段架构核心参数 (Hybrid Prompting & MIPL)】
# ==========================================================
cfg.MODEL.TRAIN_STAGE = 1         # 核心开关：1 为混合提示学习，2 为视觉解耦微调
cfg.MODEL.PROMPT_LENGTH = 4       # M=4，可学习提示词的长度
cfg.MODEL.NUM_CLOTHES = 0         # 占位符，由 dataloader 动态计算后自动填入

# --- 损失权重配置 (Stage 2 使用) ---
cfg.MODEL.ID_LOSS_WEIGHT = 1.0
cfg.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
cfg.MODEL.I2T_ID_WEIGHT = 1.0           # 对应 L_Guide (身份语义锚点牵引)
cfg.MODEL.I2T_CLOTH_SC_WEIGHT = 1.0     # 对应 L_sc (语义剥离校验/监督 cloth_proj)
cfg.MODEL.I2T_CLOTH_ORTHO_WEIGHT = 0.05 # 对应 L_de (截断余弦正交/外科手术排斥)
cfg.MODEL.CLOTH_SUBTRACT_LAMBDA = 0.5   # 测试专用：衣服成分减除强度 λ (0.5 是经验值)

cfg.MODEL.NECK = 'bnneck'
cfg.MODEL.NECK_FEAT = 'before'
cfg.MODEL.COS_LAYER = False
cfg.MODEL.NO_MARGIN = False
cfg.MODEL.DIST_TRAIN = False

# --- INPUT: 图像预处理 ---
cfg.INPUT = CN()
cfg.INPUT.SIZE_TRAIN = [384, 192]
cfg.INPUT.SIZE_TEST = [384, 192]
cfg.INPUT.PROB = 0.5
cfg.INPUT.RE_PROB = 0.7
cfg.INPUT.PADDING = 10
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]

# --- DATALOADER: 数据总线 ---
cfg.DATALOADER = CN()
cfg.DATALOADER.SAMPLER = 'softmax_triplet'
cfg.DATALOADER.NUM_INSTANCE = 4
cfg.DATALOADER.NUM_WORKERS = 8

# --- SOLVER: 动力系统 ---
cfg.SOLVER = CN()

# 【两阶段独立超参数】
cfg.SOLVER.STAGE1_MAX_EPOCHS = 120
cfg.SOLVER.STAGE1_BASE_LR = 3.5e-4

cfg.SOLVER.STAGE2_MAX_EPOCHS = 40
cfg.SOLVER.STAGE2_BASE_LR = 5e-6
cfg.SOLVER.STAGE2_WARMUP_EPOCHS = 10

# 基础设定兼容项 (会被 YML 和阶段代码覆盖)
cfg.SOLVER.MAX_EPOCHS = 60
cfg.SOLVER.CHECKPOINT_PERIOD = 20
cfg.SOLVER.LOG_PERIOD = 50
cfg.SOLVER.EVAL_PERIOD = 60
cfg.SOLVER.BIAS_LR_FACTOR = 2.0
cfg.SOLVER.SEED = 1234
cfg.SOLVER.MARGIN = 0.3
cfg.SOLVER.IMS_PER_BATCH = 64
cfg.SOLVER.OPTIMIZER_NAME = "AdamW" # 推荐ViT使用AdamW
cfg.SOLVER.BASE_LR = 0.00001
cfg.SOLVER.WARMUP_LR_INIT = 0.000001
cfg.SOLVER.WARMUP_EPOCHS = 5
cfg.SOLVER.STEPS = [20, 30]
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_METHOD = 'linear'
cfg.SOLVER.WARMUP_FACTOR = 0.01
cfg.SOLVER.WEIGHT_DECAY = 0.0005
cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
cfg.SOLVER.LARGE_FC_LR = True

# --- TEST: 测试与特征提取 ---
cfg.TEST = CN()
cfg.TEST.EVAL = True
cfg.TEST.EVALUATE_ONLY = False
cfg.TEST.IMS_PER_BATCH = 128
cfg.TEST.RE_RANKING = False
cfg.TEST.WEIGHT = ''
cfg.TEST.NECK_FEAT = 'before'
cfg.TEST.FEAT_NORM = 'yes'

# --- DATASETS: 数据源与文本挂载点 ---
cfg.DATASETS = CN()
cfg.DATASETS.NAMES = ('prcc')
cfg.DATASETS.ROOT_DIR = ''
cfg.DATASETS.LLAVA_JSON_PATH = ''  # 注入新工程的物理通路

cfg.OUTPUT_DIR = ''