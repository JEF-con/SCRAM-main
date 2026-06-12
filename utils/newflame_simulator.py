#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟 newFlame 的计时与结果保存（CPU真实计算版，无缩放、无采样）。

功能：
- 提供两种模式：
  1) CPU上执行真实张量/矩阵运算并计时（按传入 n、P 规模直接运行）；
  2) 理论复杂度生成耗时（不做缩放，仅用于对比或快速验证）；
- 支持多次调用，将结果以与 newFlame 相同的 CSV/JSON 结构追加保存；

输出目录与文件名：
- 目录：timing_analysis
- 文件：newflame_timing_{training_session_id}.csv / .json
"""

import os
import json
import csv
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def _fmt6(x: float) -> str:
    return f"{x:.6f}"


def _fmt2(x: float) -> str:
    return f"{x:.2f}"


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _append_json(json_filename: str, timing_data: dict) -> None:
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(timing_data)
        except (json.JSONDecodeError, FileNotFoundError):
            existing = [timing_data]
    else:
        existing = [timing_data]
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _append_csv(csv_filename: str,
                session_id: str,
                call_num: int,
                timestamp: str,
                sections: dict,
                total_time: float) -> None:
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['训练会话ID', '调用次数', '时间戳', '操作名称', '用时(秒)', '占比(%)'])

        def write_row(name: str, t: float):
            w.writerow([session_id, call_num, timestamp, name, _fmt6(t), _fmt2((t / total_time * 100) if total_time > 0 else 0.0)])

        # 顶层与嵌套步骤（名称需与 defense.py 保持一致）
        write_row('模型拆分成算术共享操作', sections['arithmetic_share'])
        write_row('张量A和D生成及算术共享操作', sections['tensor_generation'])
        write_row('  - A和D验证过程', sections['ad_verification'])
        write_row('张量alpha、beta、gamma生成及算术共享操作', sections['alpha_beta_gamma'])
        write_row('  - alpha、beta、gamma验证过程', sections['abg_verification'])
        write_row('zeta相关操作', sections['zeta_operations'])
        write_row('deta相关操作', sections['deta_operations'])
        write_row('同态加密操作', sections['homomorphic_operations'])
        write_row('聚类算法', sections['clustering_algorithm'])
        write_row('  - 余弦相似度矩阵计算', sections['cosine_similarity'])
        write_row('  - HDBSCAN聚类算法', sections['hdbscan_clustering'])
        write_row('  - 客户端分类和选择', sections['client_classification'])
        w.writerow([session_id, call_num, timestamp, '总体用时', _fmt6(total_time), '100.00'])
        w.writerow([])


def _gen_times(n: int, P: int, randomness: float) -> dict:
    """按复杂度生成各步骤耗时（秒）。n 为客户端数，P 为参数规模。"""
    def jitter(x: float) -> float:
        return x * (1.0 + random.uniform(-randomness, randomness))

    arithmetic_share = jitter(3.0e-7 * P)
    tensor_generation = jitter(1.2e-6 * n * P)
    ad_verification = jitter(0.98 * tensor_generation)
    alpha_beta_gamma = jitter(1.0e-6 * n * P)
    abg_verification = jitter(0.98 * alpha_beta_gamma)
    zeta_operations = jitter(5.0e-7 * n * P)
    deta_operations = jitter(5.0e-6 * (n ** 2) * P)

    homomorphic_operations = jitter(2.0e-3 * max(n, 1))

    cosine_similarity = jitter(2.0e-6 * (n ** 2) * P)
    hdbscan_clustering = jitter(1.0e-4 * (n ** 2))
    client_classification = jitter(5.0e-5 * n)
    clustering_algorithm = cosine_similarity + hdbscan_clustering + client_classification

    sections = {
        'arithmetic_share': max(arithmetic_share, 0.0),
        'tensor_generation': max(tensor_generation, 0.0),
        'ad_verification': max(ad_verification, 0.0),
        'alpha_beta_gamma': max(alpha_beta_gamma, 0.0),
        'abg_verification': max(abg_verification, 0.0),
        'zeta_operations': max(zeta_operations, 0.0),
        'deta_operations': max(deta_operations, 0.0),
        'homomorphic_operations': max(homomorphic_operations, 0.0),
        'cosine_similarity': max(cosine_similarity, 0.0),
        'hdbscan_clustering': max(hdbscan_clustering, 0.0),
        'client_classification': max(client_classification, 0.0),
        'clustering_algorithm': max(clustering_algorithm, 0.0),
    }
    return sections


def _measure(func):
    """测量函数执行的用时（秒）。"""
    start = time.time()
    result = func()
    return time.time() - start, result


def _make_dummy_params(num_clients: int, param_size: int):
    """构造CPU上的伪造模型/更新参数，形状与规模受P控制。"""
    def make_vec():
        return torch.rand(param_size, dtype=torch.float64, device='cpu')

    local_model = []
    update_params = []
    global_model = {}
    for _ in range(num_clients):
        m = {'param': make_vec()}
        u = {'param': make_vec()}
        local_model.append(m)
        update_params.append(u)
    global_model['param'] = make_vec()
    return local_model, update_params, global_model


def _simulate_cpu_once(num_clients: int,
                       param_size: int,
                       randomness: float) -> dict:
    """
    在CPU上执行真实计算以计时，直接按传入的 n、P 规模运行，不做缩放和采样。
    返回各步骤的耗时字典（秒）。
    """
    sample_n = num_clients
    sample_p = param_size
    local_model, update_params, global_model = _make_dummy_params(sample_n, sample_p)

    # 算术共享（O(n·P)）
    def arithmetic_share_step():
        update_params0 = []
        update_params1 = []
        for user_params in update_params:
            val = user_params['param']
            share1 = torch.rand_like(val, device='cpu', dtype=torch.float64)
            share2 = val - share1
            update_params0.append({'param': share1})
            update_params1.append({'param': share2})
        return update_params0, update_params1
    t_arith, (update_params0, update_params1) = _measure(arithmetic_share_step)

    # A/D 生成（O(n·P)）
    def tensor_gen_step():
        A0, A1, D0, D1 = [], [], [], []
        for user_params in update_params:
            A_tensor = torch.rand_like(user_params['param'])
            D_tensor = torch.pow(A_tensor, 2)
            A0.append({'param': A_tensor})
            D0.append({'param': D_tensor})

            A_tensor2 = torch.rand_like(user_params['param'])
            D_tensor2 = torch.pow(A_tensor2, 2)
            A1.append({'param': A_tensor2})
            D1.append({'param': D_tensor2})
        return A0, A1, D0, D1
    t_tensor, (A0, A1, D0, D1) = _measure(tensor_gen_step)

    # A/D 验证（报告但不计入总时长）
    def ad_verify_step():
        for idx in range(sample_n):
            A1_cpu = A0[idx]['param']
            A2_cpu = A1[idx]['param']
            D1_cpu = D0[idx]['param']
            D2_cpu = D1[idx]['param']
            t = 3
            e = t * A1_cpu - A2_cpu
            _ = t**2 * D1_cpu - D2_cpu - 2 * t * e * A1_cpu + torch.pow(e, 2)
        return None
    t_ad_verify, _ = _measure(ad_verify_step)

    # α/β/γ 生成（O(n·P)）
    def abg_step():
        alpha, beta, gamma = [], [], []
        for user_params in update_params:
            a = torch.rand_like(user_params['param'])
            b = torch.rand_like(user_params['param'])
            g = torch.rand_like(user_params['param'])
            alpha.append({'param': a})
            beta.append({'param': b})
            gamma.append({'param': g})
        return alpha, beta, gamma
    t_abg, (alpha, beta, gamma) = _measure(abg_step)

    def abg_verify_step():
        for i in range(sample_n):
            _ = torch.sum((alpha[i]['param'] + beta[i]['param']) > gamma[i]['param'])
        return None
    t_abg_verify, _ = _measure(abg_verify_step)

    # zeta 操作（O(n·P)）
    def zeta_step():
        e0, e1, e = [], [], []
        Zeta0, Zeta1 = [], []
        for i in range(sample_n):
            e0_user = update_params0[i]['param'] - A0[i]['param']
            e1_user = update_params1[i]['param'] - A1[i]['param']
            e_user = e0_user + e1_user
            Z0 = D0[i]['param'] + 2 * e_user * update_params0[i]['param'] - torch.pow(e_user, 2)
            Z1 = D1[i]['param'] + 2 * e_user * update_params1[i]['param']
            e0.append({'param': e0_user}); e1.append({'param': e1_user}); e.append({'param': e_user})
            Zeta0.append({'param': Z0}); Zeta1.append({'param': Z1})
        return e0, e1, e, Zeta0, Zeta1
    t_zeta, (e0, e1, e, Zeta0, Zeta1) = _measure(zeta_step)

    # deta 操作（O(n^2·P)）
    def deta_step():
        for i in range(sample_n):
            vi = update_params0[i]['param']
            for j in range(i + 1, sample_n):
                vj = update_params1[j]['param']
                tmp = vi + vj
                tmp = tmp * tmp
                _ = tmp - (Zeta0[i]['param'] + Zeta1[j]['param'])
        return None
    t_deta, _ = _measure(deta_step)

    # 同态操作近似（O(n^2)）
    def he_step():
        acc = 0.0
        for i in range(sample_n):
            for j in range(i + 1, sample_n):
                acc += float(torch.mean(update_params0[i]['param'])) / (float(torch.mean(update_params1[j]['param'])) + 1e-9)
        return acc
    t_he, _ = _measure(he_step)

    # 余弦相似度矩阵（O(n^2·P)）：严格 1 - cos(ai, bj)
    def cosine_step():
        vecs = [u['param'].numpy() for u in update_params]
        n = len(vecs)
        cos_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            ai = vecs[i]
            ni = np.linalg.norm(ai) + 1e-9
            for j in range(n):
                bj = vecs[j]
                nj = np.linalg.norm(bj) + 1e-9
                cosv = float(np.dot(ai, bj)) / (ni * nj)
                cos_mat[i][j] = 1.0 - cosv
        return cos_mat
    t_cos, cos_list = _measure(cosine_step)

    # HDBSCAN聚类
    def cluster_step():
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, sample_n // 2 + 1),
                min_samples=1,
                allow_single_cluster=True,
                metric='precomputed'
            ).fit(cos_list)
            labels = clusterer.labels_
        except Exception:
            labels = np.zeros(len(cos_list), dtype=int)
        return labels
    t_hdb, cluster_labels = _measure(cluster_step)

    # 客户端分类与L2范数（CPU）
    def classify_step(labels_input=cluster_labels):
        labels_arr = np.array(labels_input)
        benign = []
        if labels_arr.max() < 0:
            benign = list(range(sample_n))
        else:
            max_idx = 0
            max_cnt = 0
            for c in range(labels_arr.max() + 1):
                cnt = np.sum(labels_arr == c)
                if cnt > max_cnt:
                    max_cnt = cnt; max_idx = c
            benign = [i for i in range(len(labels_arr)) if labels_arr[i] == max_idx]
        norms = []
        for i in range(sample_n):
            norms.append(float(torch.norm(update_params[i]['param'], p=2)))
        return benign, norms
    t_class, _ = _measure(classify_step)

    sections = {
        'arithmetic_share': t_arith,
        'tensor_generation': t_tensor,
        'ad_verification': t_ad_verify,
        'alpha_beta_gamma': t_abg,
        'abg_verification': t_abg_verify,
        'zeta_operations': t_zeta,
        'deta_operations': t_deta,
        'homomorphic_operations': t_he,
        'cosine_similarity': t_cos,
        'hdbscan_clustering': t_hdb,
        'client_classification': t_class,
    }
    sections['clustering_algorithm'] = sections['cosine_similarity'] + sections['hdbscan_clustering'] + sections['client_classification']
    return sections


def simulate_newflame(num_clients: int,
                      param_size: int,
                      num_calls: int = 1,
                      session_id: Optional[str] = None,
                      randomness: float = 0.0,
                      mode: str = "cpu",  # "cpu" 真实计时；"theory" 理论生成（不缩放）
                      timing_dir: str = "timing_analysis") -> str:
    """
    模拟调用 newFlame，生成计时并保存到 CSV/JSON。

    Args:
        num_clients: 客户端数量 n
        param_size: 每客户端参数元素总数 P
        num_calls: 连续调用次数（每次都会追加到同一会话文件）
        session_id: 自定义会话ID；为空则使用当前时间戳
        randomness: 理论模式的随机扰动幅度（0~1）；CPU模式忽略
        mode: "cpu" 在CPU上执行真实计算计时；"theory" 使用理论复杂度生成耗时（不缩放）
        timing_dir: 输出目录（与 newFlame 一致为 timing_analysis）

    Returns:
        返回生成的会话ID字符串
    """
    if num_clients <= 0:
        raise ValueError("num_clients 必须为正整数")
    if param_size <= 0:
        raise ValueError("param_size 必须为正整数")
    if num_calls <= 0:
        raise ValueError("num_calls 必须为正整数")

    _ensure_dir(timing_dir)
    if not session_id:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_filename = os.path.join(timing_dir, f"newflame_timing_{session_id}.json")
    csv_filename = os.path.join(timing_dir, f"newflame_timing_{session_id}.csv")

    for call_idx in range(1, num_calls + 1):
        if mode == "cpu":
            sections = _simulate_cpu_once(num_clients, param_size, randomness)
        else:
            sections = _gen_times(num_clients, param_size, randomness)

        sections['clustering_algorithm'] = (
            sections['cosine_similarity']
            + sections['hdbscan_clustering']
            + sections['client_classification']
        )

        total_time = (
            sections['arithmetic_share']
            + sections['tensor_generation']
            + sections['alpha_beta_gamma']
            + sections['zeta_operations']
            + sections['deta_operations']
            + sections['homomorphic_operations']
            + sections['clustering_algorithm']
        )

        timestamp = datetime.now().isoformat()
        timing_data = {
            "training_session_id": session_id,
            "call_number": call_idx,
            "timestamp": timestamp,
            "num_clients": num_clients,
            "param_size": param_size,
            "mode": mode,
            "total_time": total_time,
            "operations": {
                "arithmetic_share": {
                    "time": sections['arithmetic_share'],
                    "percentage": (sections['arithmetic_share'] / total_time * 100) if total_time > 0 else 0.0
                },
                "tensor_generation": {
                    "time": sections['tensor_generation'],
                    "percentage": (sections['tensor_generation'] / total_time * 100) if total_time > 0 else 0.0,
                    "ad_verification": {
                        "time": sections['ad_verification'],
                        "percentage": (sections['ad_verification'] / total_time * 100) if total_time > 0 else 0.0
                    }
                },
                "alpha_beta_gamma": {
                    "time": sections['alpha_beta_gamma'],
                    "percentage": (sections['alpha_beta_gamma'] / total_time * 100) if total_time > 0 else 0.0,
                    "abg_verification": {
                        "time": sections['abg_verification'],
                        "percentage": (sections['abg_verification'] / total_time * 100) if total_time > 0 else 0.0
                    }
                },
                "zeta_operations": {
                    "time": sections['zeta_operations'],
                    "percentage": (sections['zeta_operations'] / total_time * 100) if total_time > 0 else 0.0
                },
                "deta_operations": {
                    "time": sections['deta_operations'],
                    "percentage": (sections['deta_operations'] / total_time * 100) if total_time > 0 else 0.0
                },
                "homomorphic_operations": {
                    "time": sections['homomorphic_operations'],
                    "percentage": (sections['homomorphic_operations'] / total_time * 100) if total_time > 0 else 0.0
                },
                "clustering_algorithm": {
                    "time": sections['clustering_algorithm'],
                    "percentage": (sections['clustering_algorithm'] / total_time * 100) if total_time > 0 else 0.0,
                    "cosine_similarity": {
                        "time": sections['cosine_similarity'],
                        "percentage": (sections['cosine_similarity'] / total_time * 100) if total_time > 0 else 0.0
                    },
                    "hdbscan_clustering": {
                        "time": sections['hdbscan_clustering'],
                        "percentage": (sections['hdbscan_clustering'] / total_time * 100) if total_time > 0 else 0.0
                    },
                    "client_classification": {
                        "time": sections['client_classification'],
                        "percentage": (sections['client_classification'] / total_time * 100) if total_time > 0 else 0.0
                    }
                }
            }
        }

        _append_json(json_filename, timing_data)
        _append_csv(csv_filename, session_id, call_idx, timestamp, sections, total_time)

    print(f"模拟计时已保存到:\n JSON: {json_filename}\n CSV: {csv_filename}")
    return session_id


if __name__ == "__main__":
    # 演示：直接按参数运行（不缩放不采样）。为保证示例可快速完成，选择适中规模。
    simulate_newflame(num_clients=50, param_size=1500000, num_calls=2, mode="cpu")