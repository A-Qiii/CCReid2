import torch
import os
import numpy as np
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter


def do_train_stage1(cfg, model, train_loader, optimizer, scheduler, loss_fn):
    device = cfg.MODEL.DEVICE
    model.train()
    epochs = cfg.SOLVER.STAGE1_MAX_EPOCHS
    log_period = getattr(cfg.SOLVER, 'LOG_PERIOD', 50)
    scaler = amp.GradScaler()

    tb_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard", "stage1")
    tb_writer = SummaryWriter(log_dir=tb_dir)

    print(f">>> [Stage 1] 启动混合提示学习循环 (InfoNCE)，共 {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        for n_iter, batch in enumerate(train_loader):
            img       = batch[0].to(device)
            target    = batch[1].to(device)
            cloth_id  = batch[3].to(device)
            id_text   = batch[5]
            cloth_text = batch[6]

            with amp.autocast():
                feat_list = model(x=img, label=target, cloth_label=cloth_id,
                                  id_text=id_text, cloth_text=cloth_text)
                loss, _ = loss_fn(None, feat_list, target, target_cloth=cloth_id)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if n_iter % log_period == 0:
                print(f"Stage 1 Epoch[{epoch}] Iter[{n_iter}/{len(train_loader)}] "
                      f"InfoNCE Loss: {loss.item():.4f}")
                global_step = (epoch - 1) * len(train_loader) + n_iter
                tb_writer.add_scalar("Train/Stage1_Loss", loss.item(), global_step)

        scheduler.step()

        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == epochs:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(cfg.OUTPUT_DIR, f"stage1_prompt_model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> [Stage 1] Prompt 权重已保存至: {save_path}")

    tb_writer.close()


def extract_text_bank(cfg, model, train_loader):
    """
    提取全局文本锚点矩阵（bank_id）。

    遍历底层 dataset 列表而非 DataLoader batch，避免因采样不均导致
    某些 pid 的文本被遗漏或重复覆盖。
    train tuple 格式: (img_path, mapped_pid, camid, cloth_id, id_text, cloth_text)
    """
    device = cfg.MODEL.DEVICE
    model.eval()
    print(f">>> 正在提取全局文本特征 Bank...")

    dataset_obj = train_loader.dataset          # ImageDataset
    raw_dataset_list = dataset_obj.dataset      # PRCC.train / LTCC.train（list of tuples）

    pid_to_text = {}
    for data_tuple in raw_dataset_list:
        mapped_pid = data_tuple[1]
        id_text    = data_tuple[4]
        if mapped_pid not in pid_to_text:
            pid_to_text[mapped_pid] = id_text

    num_classes = len(pid_to_text)
    expected_pids = set(range(num_classes))
    actual_pids   = set(pid_to_text.keys())
    if expected_pids != actual_pids:
        missing = expected_pids - actual_pids
        print(f"[警告] Text Bank 中缺少以下 PID 的文本: {missing}，将使用空字符串兜底。")
        for pid in missing:
            pid_to_text[pid] = "a person"

    bank_id = torch.zeros(num_classes, 512).to(device)

    with torch.no_grad():
        for mapped_pid in range(num_classes):
            text = pid_to_text[mapped_pid]
            # cloth_label 传 0 作为占位（Stage 2 文本 bank 不依赖衣服分支）
            p_id, _, tk_id, _ = model.prompt_learner(
                torch.tensor([mapped_pid]).to(device),
                torch.tensor([0]).to(device),
                [text], [""]
            )
            t_id = model.text_encoder_forward(p_id, tk_id)
            bank_id[mapped_pid] = t_id.squeeze(0)

    print(f">>> 全局身份文本 Bank 提取完毕，矩阵形状: {bank_id.shape}")
    return bank_id.detach()


def do_train_stage2(cfg, model, train_loader, val_loader, optimizer, scheduler,
                    loss_fn, num_query, text_bank_id):
    device = cfg.MODEL.DEVICE
    model.train()
    epochs     = cfg.SOLVER.STAGE2_MAX_EPOCHS
    log_period = getattr(cfg.SOLVER, 'LOG_PERIOD', 50)
    scaler     = amp.GradScaler()

    tb_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard", "stage2")
    tb_writer = SummaryWriter(log_dir=tb_dir)

    print(f">>> [Stage 2] 启动 MIPL 视觉特征解耦微调，共 {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        for n_iter, batch in enumerate(train_loader):
            img        = batch[0].to(device)
            target     = batch[1].to(device)
            cloth_id   = batch[3].to(device)   # ← 衣服标签，已存在于 batch
            id_text    = batch[5]
            cloth_text = batch[6]

            with amp.autocast():
                cls_score, feat_list = model(x=img, label=target, cloth_label=cloth_id,
                                             id_text=id_text, cloth_text=cloth_text)

                # ── 【改动点】新增 target_cloth=cloth_id ───────────────────────
                # make_loss v2 中 Stage 2 分支利用 target_cloth 计算：
                #   1. 置信度门控 w_i（依赖衣服标签索引 t_cloth_gt）
                #   2. L_cloth_tri（衣服三元组，破除 Proj_c 模式坍塌）
                #   3. ProjC/intra_cloth_sim 与 inter_cloth_sim 探针
                # 原版 loss_fn 调用不传 target_cloth，效果完全等价（退化为零）。
                loss, _ = loss_fn(cls_score, feat_list, target,
                                  text_bank_id=text_bank_id,
                                  target_cloth=cloth_id)  # ← 唯一改动

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if n_iter % log_period == 0:
                acc = (cls_score.max(1)[1] == target).float().mean()
                print(f"Stage 2 Epoch[{epoch}] Iter[{n_iter}/{len(train_loader)}] "
                      f"Total Loss: {loss.item():.4f} | Base Acc: {acc.item():.3f}")
                global_step = (epoch - 1) * len(train_loader) + n_iter
                tb_writer.add_scalar("Train/Stage2_TotalLoss", loss.item(), global_step)
                tb_writer.add_scalar("Train/Stage2_BaseAcc",   acc.item(),  global_step)

        scheduler.step()

        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == epochs:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(cfg.OUTPUT_DIR,
                                     f"stage2_disentangled_model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> [Stage 2] 权重已保存至: {save_path}")

    tb_writer.close()


def do_inference(cfg, model, val_loader, num_query):
    device = cfg.MODEL.DEVICE
    model.eval()
    feats, pids, camids = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            img   = batch[0].to(device)
            pid   = batch[1]
            camid = batch[2]
            feat  = model(x=img)
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
    return torch.cat(feats, dim=0), pids, camids