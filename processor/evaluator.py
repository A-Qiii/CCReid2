import numpy as np

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids=None, g_clothids=None, ltcc_cc_setting=True, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"注: Gallery 样本数较小, 仅为 {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]

        # 核心防作弊逻辑：CC 设定 vs Standard 设定
        if ltcc_cc_setting and q_clothids is not None and g_clothids is not None:
            q_clothid = q_clothids[q_idx]
            # 换衣模式：剔除同摄像头，且剔除同衣服样本
            remove = ((g_pids[order] == q_pid) & (g_camids[order] == q_camid)) | \
                     ((g_pids[order] == q_pid) & (g_clothids[order] == q_clothid))
        else:
            # 标准模式：仅剔除同摄像头样本
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)

        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "致命错误: Query 身份在 Gallery 中完全不存在有效匹配"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP