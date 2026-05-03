import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .ltcc import LTCC
from .prcc import PRCC
from .bases import ImageDataset
from .sampler import RandomIdentitySampler


def make_dataloader(cfg):
    # 1. 图像预处理流水线
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        T.RandomErasing(p=cfg.INPUT.RE_PROB)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # 2. 数据集路由：注入 JSON 绝对路径
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if cfg.DATASETS.NAMES == 'ltcc':
        dataset = LTCC(root=cfg.DATASETS.ROOT_DIR, llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    elif cfg.DATASETS.NAMES == 'prcc':
        dataset = PRCC(root=cfg.DATASETS.ROOT_DIR, llava_json_path=cfg.DATASETS.LLAVA_JSON_PATH)
    else:
        raise RuntimeError(f"不支持的数据集: {cfg.DATASETS.NAMES}")

    # 3. 封装 Dataloader：注入 llava_dict，彻底切除 STAGE2 冗余
    train_set = ImageDataset(dataset.train, train_transforms, llava_dict=dataset.llava_dict)

    # 直接使用全局 IMS_PER_BATCH 控制采样器
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(
            dataset.train,
            cfg.SOLVER.IMS_PER_BATCH,
            cfg.DATALOADER.NUM_INSTANCE
        ),
        num_workers=num_workers,
        collate_fn=None,
        drop_last=True
    )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, llava_dict=dataset.llava_dict)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=None
    )

    return train_loader, train_loader, val_loader, len(dataset.query), dataset.num_train_pids, dataset.num_train_cams, 0