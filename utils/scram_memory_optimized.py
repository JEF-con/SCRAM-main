import math
import time
import numpy as np
import torch
import hdbscan

# 复用已实现的核心组件与工具，保持算法与隐私流程一致
from .scram import (
    parameters_dict_to_vector_flt,
    parameters_dict_to_vector,
    arithmetic_share,
    models_arithmetic_share,
    gen_sc,
    verify_sc_shares,
    gen_delta,
    gen_delta_ij,
    precompute_y_ciphertexts_for_clients,
    precompute_delta_ciphertexts_for_clients,
    compute_xR_from_cipher_parts,
    filter_benign_clients_by_iqr,
    scram_weighted_aggregate,
    no_defence_balance,
)


def _log_cuda_peak(stage: str):
    """打印当前阶段的 CUDA 显存峰值（若可用）。"""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[SCRAM-MO] {stage} 峰值显存: {peak:.2f} GB")
        except Exception:
            pass


def _reset_cuda_peak():
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _empty_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _ensure_paillier_keys(args):
    """若缺失，则生成 Paillier 公私钥、随机数 R 与缩放因子。"""
    import phe as paillier

    need_generate = (
        not hasattr(args, 'paillier_public_key') or args.paillier_public_key is None or
        not hasattr(args, 'paillier_private_key') or args.paillier_private_key is None or
        not hasattr(args, 'paillier_R') or args.paillier_R is None or
        not hasattr(args, 'paillier_scale_factor') or args.paillier_scale_factor is None
    )
    if need_generate:
        key_length = 256
        args.paillier_public_key, args.paillier_private_key = paillier.generate_paillier_keypair(n_length=key_length)
        args.paillier_R = int(np.random.randint(10, 101))
        args.paillier_scale_factor = int(1000)
        print("[SCRAM-MO] 生成新的 Paillier 密钥对与参数")
    else:
        print("[SCRAM-MO] 复用既有 Paillier 密钥对与参数")


def scram_memory_optimized(local_model, update_params, global_model, args):
    """
    在保留 SCRAM 隐私保护与计算流程的前提下，优化显存使用的实现：
    - 将算术共享、SC 生成/验证、delta/delta_ij 等“共享层”计算尽量在 CPU 执行；
    - 聚类与 IQR 过滤输入改为 CPU 上的明文数组，避免大矩阵驻留 GPU；
    - 预加密与解密全部在 CPU，分块构造分子密文，减少循环中重复加密/解密；
    - 分阶段清理中间变量，并在每阶段结束时主动释放 CUDA 缓存；
    - 保持原算法的安全性（每 (i,j) 使用独立 R），并沿用既有聚类与加权聚合逻辑。

    参数与返回值与原始 `scram` 一致：
    - local_model: List[Dict[str, Tensor]]
    - update_params: List[Dict[str, Tensor]]
    - global_model: Dict[str, Tensor]
    - args: 需包含 `gpu`, `frac`, `num_users`, `malicious` 等字段，以及 Paillier 配置
    返回：新的聚合后全局模型参数字典。
    """
    torch.set_grad_enabled(False)
    _reset_cuda_peak()

    debug = getattr(args, 'debug_scram', False)

    # 阶段 0：向量化（尽量在 CPU 保留副本，用于共享层计算）
    start_t = time.time()
    local_vec_gpu = []
    local_vec_cpu = []
    for param in local_model:
        v = parameters_dict_to_vector_flt(param)
        local_vec_gpu.append(v)
        local_vec_cpu.append(v.detach().cpu())
    n = len(local_vec_cpu)
    m = int(local_vec_cpu[0].numel()) if n > 0 else 0
    if debug:
        print(f"[SCRAM-MO] 向量化完成: n={n}, m={m}")
    _log_cuda_peak("向量化")

    # 阶段 1：算术共享与 SC（CPU）
    # 将共享层计算全部在 CPU；结束后释放 GPU 上的向量副本
    start_t = time.time()
    local_model_share0, local_model_share1 = models_arithmetic_share(local_vec_cpu)

    A0_list, D0_list, A1_list, D1_list = [], [], [], []
    A0_s0, A0_s1, A1_s0, A1_s1 = [], [], [], []
    D0_s0, D0_s1, D1_s0, D1_s1 = [], [], [], []
    for i in range(n):
        A0, D0 = gen_sc(local_vec_cpu[i])
        A1, D1 = gen_sc(local_vec_cpu[i])
        A0_list.append(A0); D0_list.append(D0)
        A1_list.append(A1); D1_list.append(D1)
        s0, s1 = arithmetic_share(A0); A0_s0.append(s0); A0_s1.append(s1)
        s0, s1 = arithmetic_share(A1); A1_s0.append(s0); A1_s1.append(s1)
        s0, s1 = arithmetic_share(D0); D0_s0.append(s0); D0_s1.append(s1)
        s0, s1 = arithmetic_share(D1); D1_s0.append(s0); D1_s1.append(s1)

    # 验证 SC，共享选择一组（与原始实现保持一致）
    A_share0, A_share1, D_share0, D_share1 = [], [], [], []
    verified = 0
    for i in range(n):
        ok = verify_sc_shares(A0_s0[i], A0_s1[i], D0_s0[i], D0_s1[i], A1_s0[i], A1_s1[i], D1_s0[i], D1_s1[i])
        if ok:
            A_share0.append(A0_s0[i]); A_share1.append(A0_s1[i])
            D_share0.append(D0_s0[i]); D_share1.append(D0_s1[i])
            verified += 1
    if debug:
        print(f"[SCRAM-MO] SC 验证通过 {verified}/{n}")
    # 释放 GPU 向量副本（共享层后不再需要）
    try:
        del local_vec_gpu
    except Exception:
        pass
    _empty_cuda_cache()
    _log_cuda_peak("共享与SC")

    # 阶段 2：delta 与 delta_ij（CPU），保持现有隐私公式
    start_t = time.time()
    delta0, delta1 = gen_delta(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1)
    delta_ij0, delta_ij1 = gen_delta_ij(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1, args=args)
    if debug and n > 0:
        print(f"[SCRAM-MO] delta & delta_ij 计算完成 (n={n})")
    _log_cuda_peak("delta/delta_ij")

    # 阶段 3：Paillier 参数确保与密文预计算（CPU）
    _ensure_paillier_keys(args)
    gy10_list, gy11_list = precompute_y_ciphertexts_for_clients(delta0, delta1, args)
    g_d0_list, g_d1_list = precompute_delta_ciphertexts_for_clients(delta0, delta1, args)
    # 预先解密 y_sum（用于构造 yR）
    private_key = args.paillier_private_key
    y_sum_plain = [float(private_key.decrypt(gy10_list[k] + gy11_list[k])) for k in range(n)]

    # 阶段 4：使用隐私路径计算余弦距离矩阵（CPU 列表，避免 GPU 常驻）
    cos_distance = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        j_max = min(i, len(delta_ij0[i]) if i < len(delta_ij0) else 0, len(delta_ij1[i]) if i < len(delta_ij1) else 0)
        for j in range(j_max):
            # 构造 (i,j) 的分子密文并解密得到 xR
            R_ij = int(np.random.randint(10, 101))
            xR_val = compute_xR_from_cipher_parts(i, j, delta_ij0, delta_ij1, g_d0_list, g_d1_list, R_ij, args)
            y1R_val = R_ij * y_sum_plain[i]
            y2R_val = R_ij * y_sum_plain[j]
            denom = math.sqrt(max(y1R_val * y2R_val, 1e-12))
            cos_distance[j][i] = cos_distance[i][j] = 1.0 - (xR_val / denom)
        cos_distance[i][i] = 0.0

    if debug:
        print("[SCRAM-MO] 余弦距离矩阵构造完成（隐私路径）")
    _log_cuda_peak("隐私余弦矩阵")

    # 阶段 5：HDBSCAN 聚类（CPU，使用预计算距离矩阵，保持隐私）
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    try:
        clusterer = hdbscan.HDBSCAN(
            metric='precomputed',
            min_cluster_size=num_clients // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        ).fit(np.array(cos_distance, dtype=np.float64))
        labels = clusterer.labels_
    except Exception as e:
        print("[SCRAM-MO] HDBSCAN 聚类失败，原因:", e)
        labels = np.zeros(n, dtype=int)
    if debug:
        print("[SCRAM-MO] HDBSCAN 聚类结果:", labels.tolist())
    _log_cuda_peak("聚类")

    # 阶段 6：最大簇选择（CPU）
    benign_client = []
    if labels.max() < 0:
        benign_client = list(range(n))
    else:
        max_num, max_idx = 0, 0
        for cid in range(labels.max() + 1):
            count = int(np.sum(labels == cid))
            if count > max_num:
                max_num, max_idx = count, cid
        for i in range(n):
            if labels[i] == max_idx:
                benign_client.append(i)

    # 统计比例（与原始逻辑保持一致）
    for idx in benign_client:
        if idx < num_malicious_clients:
            args.wrong_mal += 1
        else:
            args.right_ben += 1
    args.turn += 1
    mal_denom = num_malicious_clients * args.turn
    ben_denom = num_benign_clients * args.turn
    mal_ratio = (args.wrong_mal / mal_denom) if mal_denom > 0 else 0.0
    ben_ratio = (args.right_ben / ben_denom) if ben_denom > 0 else 0.0
    print('[SCRAM-MO] 恶意被选比例:', mal_ratio)
    print('[SCRAM-MO] 良性被选比例:', ben_ratio)
    _log_cuda_peak("最大簇选择")

    # 阶段 7：IQR 过滤（CPU，使用 delta0/1 之和的数值）
    filtered_benign_client = filter_benign_clients_by_iqr(
        benign_client,
        delta0,
        delta1,
        args,
        whisker_k=1.5,
    )
    print("[SCRAM-MO] IQR 过滤后的良性客户端索引:", filtered_benign_client)

    # 阶段 8：加权聚合（CPU 计算权重，聚合时只在需要的张量上用 GPU）
    candidates = filtered_benign_client
    if len(candidates) == 0:
        print("[SCRAM-MO] IQR 过滤后无良性客户端，回退为均匀平均聚合。")
        agg_model = no_defence_balance(update_params, global_model)
        _log_cuda_peak("聚合(均匀)")
        return agg_model

    # 权重基于隐私距离矩阵；传入 scram_weighted_aggregate 复用既有聚合实现
    agg_model = scram_weighted_aggregate(update_params, global_model, candidates, cos_list=cos_distance)
    _log_cuda_peak("聚合(加权)")

    # 阶段 9：清理（主动释放中间大对象）
    try:
        del local_model_share0, local_model_share1
        del A0_list, D0_list, A1_list, D1_list
        del A0_s0, A0_s1, A1_s0, A1_s1
        del D0_s0, D0_s1, D1_s0, D1_s1
        del A_share0, A_share1, D_share0, D_share1
        del delta0, delta1, delta_ij0, delta_ij1
        del gy10_list, gy11_list, g_d0_list, g_d1_list, y_sum_plain
        del cos_distance, labels, benign_client, filtered_benign_client
        del local_vec_cpu
    except Exception:
        pass
    _empty_cuda_cache()
    _log_cuda_peak("清理")

    return agg_model


__all__ = [
    'scram_memory_optimized',
]