import torch
import os
import time
import datetime
import numpy as np
from torch.cuda import amp

def do_train_stage1(cfg, model, train_loader, optimizer, scheduler, loss_fn):
    device = cfg.MODEL.DEVICE
    model.train()
    epochs = cfg.SOLVER.STAGE1_MAX_EPOCHS
    log_period = getattr(cfg.SOLVER, 'LOG_PERIOD', 50)
    scaler = amp.GradScaler()

    print(f">>> [Stage 1] 启动混合提示学习循环 (InfoNCE)，共 {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        for n_iter, batch in enumerate(train_loader):
            img = batch[0].to(device)
            target = batch[1].to(device)
            cloth_id = batch[3].to(device)
            id_text = batch[4]
            cloth_text = batch[5]

            with amp.autocast():
                # 阶段一模型仅返回 [global_feat, t_id, t_cloth]
                feat_list = model(x=img, label=target, cloth_label=cloth_id, id_text=id_text, cloth_text=cloth_text)
                loss, _ = loss_fn(None, feat_list, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if n_iter % log_period == 0:
                print(f"Stage 1 Epoch[{epoch}] Iter[{n_iter}/{len(train_loader)}] InfoNCE Loss: {loss.item():.4f}")

        scheduler.step()

        # 保存阶段一固化权重
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == epochs:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(cfg.OUTPUT_DIR, f"stage1_prompt_model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> [Stage 1] Prompt 权重已保存至: {save_path}")

def extract_text_bank(cfg, model, train_loader):
    """【物理桥梁】：提取全局文本锚点矩阵，解决跨 Batch 全集牵引问题"""
    device = cfg.MODEL.DEVICE
    model.eval()
    print(f">>> 正在提取全局文本特征 Bank...")

    # 直接遍历底层 dataset 获取所有全集数据（无视采样器截断）
    dataset_list = train_loader.dataset.dataset.train 
    id_texts = {}
    
    for data_tuple in dataset_list:
        pid = data_tuple[1]
        id_text = data_tuple[4]
        if pid not in id_texts:
            id_texts[pid] = id_text

    num_classes = len(id_texts)
    bank_id = torch.zeros(num_classes, 512).to(device)

    # 冻结状态下提取纯净文本锚点
    with torch.no_grad():
        for pid, text in id_texts.items():
            p_id, _, tk_id, _ = model.prompt_learner(
                torch.tensor([pid]).to(device),
                torch.tensor([0]).to(device), # cloth 占位符，不影响 id
                [text], [""]
            )
            t_id = model.text_encoder_forward(p_id, tk_id)
            bank_id[pid] = t_id.squeeze(0)

    print(f">>> 全局身份文本 Bank 提取完毕，矩阵形状: {bank_id.shape}")
    return bank_id.detach() # 绝对静止的金标准锚点

def do_train_stage2(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, text_bank_id):
    device = cfg.MODEL.DEVICE
    model.train()
    epochs = cfg.SOLVER.STAGE2_MAX_EPOCHS
    log_period = getattr(cfg.SOLVER, 'LOG_PERIOD', 50)
    scaler = amp.GradScaler()

    print(f">>> [Stage 2] 启动 MIPL 视觉特征解耦微调，共 {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        for n_iter, batch in enumerate(train_loader):
            img = batch[0].to(device)
            target = batch[1].to(device)
            cloth_id = batch[3].to(device)
            id_text = batch[4]
            cloth_text = batch[5]

            with amp.autocast():
                # 阶段二模型返回分类 score, 以及用于 MIPL 解耦的特征列表
                cls_score, feat_list = model(x=img, label=target, cloth_label=cloth_id, id_text=id_text, cloth_text=cloth_text)
                
                # 【核心】：将全局 Text Bank 传入 Loss 用于 L_Guide 引导
                loss, _ = loss_fn(cls_score, feat_list, target, text_bank_id=text_bank_id)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if n_iter % log_period == 0:
                acc = (cls_score.max(1)[1] == target).float().mean()
                print(f"Stage 2 Epoch[{epoch}] Iter[{n_iter}/{len(train_loader)}] Total Loss: {loss.item():.4f} | Base Acc: {acc.item():.3f}")

        scheduler.step()

        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == epochs:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(cfg.OUTPUT_DIR, f"stage2_disentangled_model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> [Stage 2] 最终解耦权重已保存至: {save_path}")

def do_inference(cfg, model, val_loader, num_query):
    device = cfg.MODEL.DEVICE
    model.eval()
    feats, pids, camids = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch[0].to(device)
            pid = batch[1]
            camid = batch[2]
            feat = model(x=img) # 测试时仅走图像编码器提取骨干特征
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
    return torch.cat(feats, dim=0), pids, camids