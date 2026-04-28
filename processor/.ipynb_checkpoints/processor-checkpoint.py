import torch
import os
import time
import datetime
import numpy as np
from torch.cuda import amp

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, start_epoch=0):
    print(f">>> 准备从 Epoch[{start_epoch + 1}] 开始训练...", flush=True)
    scaler = amp.GradScaler()
    model.train()
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = getattr(cfg.SOLVER, 'LOG_PERIOD', 50)

    # 关键：如果从断点开始，必须让 scheduler 追赶到之前的进度
    if start_epoch > 0:
        print(f">>> 正在同步学习率调度器进度至第 {start_epoch} 轮...")
        for _ in range(start_epoch):
            scheduler.step()

    start_time = time.time()
    total_iters = (epochs - start_epoch) * len(train_loader)
    global_iter = 0

    # 循环从 start_epoch + 1 开始
    for epoch in range(start_epoch + 1, epochs + 1):
        optimizer.zero_grad()

        for n_iter, (img, vid, target_cam, cloth_id, _, id_text, cloth_text) in enumerate(train_loader):
            global_iter += 1
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)

            with amp.autocast():
                score, feat = model(x=img, label=target, cam_label=target_cam, 
                                    id_text=id_text, cloth_text=cloth_text)
                loss, _ = loss_fn(score, feat, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if n_iter == 0 or (n_iter + 1) % log_period == 0:
                acc = (score.max(1)[1] == target).float().mean() if not isinstance(score, list) else (score[0].max(1)[1] == target).float().mean()
                elapsed = time.time() - start_time
                avg_time_per_iter = elapsed / global_iter
                eta_seconds = int(avg_time_per_iter * (total_iters - global_iter))
                eta_str = str(datetime.timedelta(seconds=eta_seconds))

                print(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.3f}  Acc: {acc.item():.3f}  ETA: {eta_str}", flush=True)

        scheduler.step()

        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == epochs:
            save_path = os.path.join(cfg.OUTPUT_DIR, f"model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f">>> Epoch[{epoch}] 权重已保存至: {save_path}", flush=True)

def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    model.eval()
    feats, pids, camids = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            img, pid, camid = batch[0].to(device), batch[1], batch[2]
            feat = model(x=img)
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
    return torch.cat(feats, dim=0), pids, camids