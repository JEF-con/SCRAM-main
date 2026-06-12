# 定义算术共享函数
from threading import local
from turtle import window_height
import numpy as np
import torch
import copy
import time
import hdbscan
import random
import math
from concurrent.futures import ThreadPoolExecutor
import phe as paillier
import os, sys
import json
import pickle
from typing import Dict
# 确保可以导入上级目录中的模块（如 yuanbao/utils.py）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))



def parameters_dict_to_vector(param_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """将参数字典转换为向量"""
    vectors = []
    for key, param in param_dict.items():
        if 'num_batches_tracked' not in key:
            vectors.append(param.view(-1))
    return torch.cat(vectors) if vectors else torch.tensor([])

def parameters_dict_to_vector_flt(param_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """将参数字典转换为向量（浮点版本）"""
    return parameters_dict_to_vector(param_dict).float()

def no_defence_balance(update_params, global_model):
    """无防御平衡聚合"""
    if not update_params:
        return global_model
    
    aggregated_params = {}
    for key in global_model.keys():
        if 'num_batches_tracked' in key:
            continue
        
        param_sum = torch.zeros_like(global_model[key])
        for params in update_params:
            param_sum += params[key]
        
        aggregated_params[key] = param_sum / len(update_params)
    
    return aggregated_params



def weighted_defence_balance(update_params, global_model, weights):
    """将带权重的更新量加到全局模型上。

    - update_params: List[Dict[str, Tensor]]，每个元素为客户端的参数更新量 (w_local - w_glob)
    - global_model: Dict[str, Tensor]，当前全局参数
    - weights: List[float]，与 update_params 等长的权重

    返回：新的全局模型参数 (global + 加权更新)
    """
    if not update_params:
        return global_model

    # 权重长度不匹配时回退为均匀权重
    if (weights is None) or (len(weights) != len(update_params)):
        weights = [1.0 / len(update_params)] * len(update_params)

    # 归一化并做鲁棒性处理
    total_weight = float(sum(float(w) for w in weights))
    if (not math.isfinite(total_weight)) or (total_weight <= 0):
        weights = [1.0 / len(update_params)] * len(update_params)
        total_weight = 1.0

    aggregated_params = copy.deepcopy(global_model)
    for key in aggregated_params.keys():
        # BN 的 num_batches_tracked 不参与数值聚合
        if 'num_batches_tracked' in key:
            continue

        # 计算带权更新并加到当前全局
        upd_sum = torch.zeros_like(aggregated_params[key])
        for params, w in zip(update_params, weights):
            upd_sum += params[key] * float(w)
        aggregated_params[key] = aggregated_params[key] + (upd_sum / total_weight)

    return aggregated_params


def scram_weighted_aggregate(update_params, global_model, candidates, cos_list=None, eps: float = 1e-8):
    """
    SCRAM 可选的加权聚合函数：仅对候选（良性）客户端进行加权更新并累加到全局。

    通过是否调用本函数来决定是否使用“权重版”聚合；若不调用则保持你当前的无权平均逻辑。

    参数:
    - update_params: List[Dict[str, Tensor]]，每个客户端的参数更新量 (w_local - w_glob)
    - global_model: Dict[str, Tensor]，当前全局参数
    - candidates: List[int]，参与聚合的客户端索引（通常是 IQR 过滤后的良性子集）
    - cos_list: Optional[List[List[float]]]，基于原始向量计算得到的余弦距离矩阵；
                 若为 None 或数值异常，将回退为均匀权重。
    - eps: float，避免除零的微小常数

    返回:
    - Dict[str, Tensor] 新的全局参数（global + 加权平均更新）

    权重构造：对每个候选 i，计算其与其它候选的距离和 S_i，然后令权重 w_i =  (S_i + eps)，最后归一化。
    若 cos_list 缺失或权重总和非有限/非正，则回退为均匀权重。
    """
    # 候选为空时回退为无防御平均（或直接返回原逻辑）
    if candidates is None or len(candidates) == 0:
        print("[SCRAM] 候选集合为空，回退为不加权平均聚合（全体更新）。")
        return no_defence_balance(update_params, global_model)

    # 提取候选子集的更新
    subset_updates = [update_params[idx] for idx in candidates]

    # 若未提供距离矩阵，直接均匀权重
    if cos_list is None:
        weights = [1.0 / len(candidates)] * len(candidates)
        return weighted_defence_balance(subset_updates, global_model, weights)

    # 基于候选子集间的距离构造逆距离权重
    raw_weights = []
    for idx in candidates:
        s = 0.0
        for jdx in candidates:
            if jdx == idx:
                continue
            try:
                dij = float(cos_list[idx][jdx])
            except Exception:
                dij = 0.0
            if not math.isfinite(dij):
                dij = 0.0
            s += max(dij, 0.0)
        raw_weights.append(1.0 / (s + eps))

    total = float(sum(raw_weights))
    if (not math.isfinite(total)) or (total <= 0):
        weights = [1.0 / len(candidates)] * len(candidates)
    else:
        weights = [rw / total for rw in raw_weights]

    print("[SCRAM] 加权聚合候选:", candidates)
    print("[SCRAM] 归一化权重:", weights, "sum=", sum(weights))

    # 执行带权更新的聚合
    return weighted_defence_balance(subset_updates, global_model, weights)


def compute_similarity_sum_weights(cos_list, candidates, eps: float = 1e-8):
    """
    基于余弦相似度矩阵为候选（良性）客户端计算权重：
    对每个候选 i，取其与其它候选 j 的余弦相似度之和作为原始权重，
    然后进行归一化，使权重和为 1。

    注意：为避免出现负权重或数值不稳定，这里将相似度的贡献裁剪为非负值 max(sim, 0)。
    若总权重非有限或非正，则回退为均匀权重。

    参数:
    - cos_list: List[List[float]]，各向量之间的余弦相似度矩阵
    - candidates: List[int]，IQR 过滤得到的良性客户端索引
    - eps: float，避免除零的微小常数

    返回:
    - List[float]，与 candidates 对齐的归一化权重列表
    """
    if candidates is None or len(candidates) == 0:
        return []

    raw_weights = []
    for idx in candidates:
        s = 0.0
        for jdx in candidates:
            if jdx == idx:
                continue
            try:
                sim = float(cos_list[idx][jdx])
            except Exception:
                sim = 0.0
            if not math.isfinite(sim):
                sim = 0.0
            # 使用非负贡献，避免负权重影响稳定性
            s += max(sim, 0.0)
        raw_weights.append(s)

    total = float(sum(raw_weights))
    if (not math.isfinite(total)) or (total <= eps):
        weights = [1.0 / len(candidates)] * len(candidates)
    else:
        weights = [rw / total for rw in raw_weights]

    return weights


def scram_weighted_cluster_aggregate(update_params, global_model, candidates, cos_list, eps: float = 1e-8):
    """
    使用“余弦相似度求和权重”执行加权聚类/聚合：
    - 权重 w_i = sum_j max(cos(i, j), 0)（仅在 IQR 过滤的候选集合内求和）
    - 归一化后，对候选更新执行带权平均，并累加到全局参数上。

    参数:
    - update_params: List[Dict[str, Tensor]]，每个客户端的参数更新量 (w_local - w_glob)
    - global_model: Dict[str, Tensor]，当前全局参数
    - candidates: List[int]，参与聚合的客户端索引（IQR 过滤后的良性子集）
    - cos_list: List[List[float]]，余弦相似度矩阵
    - eps: float，避免除零的微小常数

    返回:
    - Dict[str, Tensor] 新的全局参数（global + 加权平均更新）
    """
    if candidates is None or len(candidates) == 0:
        print("[SCRAM] 候选集合为空，回退为不加权平均聚合（全体更新）。")
        return no_defence_balance(update_params, global_model)

    subset_updates = [update_params[idx] for idx in candidates]
    weights = compute_similarity_sum_weights(cos_list, candidates, eps=eps)

    print("[SCRAM] 加权聚类候选:", candidates)
    print("[SCRAM] 余弦相似度求和归一化权重:", weights, "sum=", sum(weights))

    return weighted_defence_balance(subset_updates, global_model, weights)



# 轻量调试输出辅助
def _tensor_stats(t: torch.Tensor):
    try:
        return {
            'shape': list(t.shape),
            'dtype': str(t.dtype),
            'device': str(t.device),
            # 'min': float(t.min().item()),
            # 'max': float(t.max().item()),
            # 'mean': float(t.float().mean().item())
            'tensor': t.tolist()
        }
    except Exception:
        return {'shape': list(getattr(t, 'shape', [])), 'dtype': str(getattr(t, 'dtype', 'unknown'))}

def _safe_print(label: str, payload: dict):
    try:
        print(f"[SCRAM DEBUG] {label}: {json.dumps(payload)}")
    except Exception:
        print(f"[SCRAM DEBUG] {label}: {payload}")

def arithmetic_share(tensor):
    """
    将张量拆分成两个算术共享
    Args:
        tensor: 输入张量
    Returns:
        share1, share2: 两个共享，满足 tensor = share1 + share2
    """
    min_float = 0.0
    max_float = 1000.0
    # 根据计算策略选择精度
    dtype = torch.float64
    dev = tensor.device
    share1 = torch.rand_like(tensor, device=dev, dtype=dtype) * (max_float - min_float) + min_float
    share2 = tensor.to(dtype) - share1
    return share1, share2

def models_arithmetic_share(models):
    """
    将模型参数拆分成两个算术共享
    Args:
        models: 输入模型参数
    Returns:
        share0, share1: 两个共享，满足 models = share0 + share1
    """
    share0 = []
    share1 = []
    for model in models:
        s0, s1 = arithmetic_share(model)
        share0.append(s0)
        share1.append(s1)
    return share0, share1

def gen_sc(tensor):
    """
    生成SC
    Args:
        tensor: 输入张量
    Returns:
        sc: 随机数
    """
    min_A = 1
    max_A = 1000
    # 设备与类型基于输入张量，无需全局变量
    dev = tensor.device
    A = torch.randint_like(tensor, low=int(min_A), high=int(max_A), device=dev, dtype=torch.long)
    D = A * A
    return A,D


def verify_sc_shares(Ai_share0, Ai_share1, Di_share0, Di_share1,Aj_share0, Aj_share1, Dj_share0, Dj_share1):
    """
    验证SC共享是否正确
    Args:
        Ai_share0, Ai_share1, Di_share0, Di_share1, Aj_share0, Aj_share1, Dj_share0, Dj_share1: 八个共享列表
    Returns:
        bool: 是否验证通过
    """
    t = torch.randint_like(Ai_share0, low=0, high=51, device=Ai_share0.device, dtype=torch.long) * 2 + 1  # 生成0-100之间的随机奇数，形状与Ai_share0相同
    e0 = t*Ai_share0 - Aj_share0
    e1 = t*Ai_share1 - Aj_share1
    e = e0+e1
    psi0 = t*t*Di_share0 - Dj_share0 - 2*t*e*Ai_share0 + e*e
    psi1= t*t*Di_share1 - Dj_share1 - 2*t*e*Ai_share1
    psi = psi0 + psi1
    if not torch.allclose(psi, torch.zeros_like(psi), atol=1e-4):
        return False
    return True

def gen_delta(local_model_share0 , local_model_share1,A_share0, A_share1,D_share0, D_share1):
    """
    计算模型更新的差异向量
    Args:
        local_model_share0, local_model_share1: 本地模型参数共享
        A_share0, A_share1: SC共享
    Returns:
        delta
    """
    n = len(local_model_share0)
    print(n)
    # 验证是否对于每个元素都有 A_i^2 == D_i
    for i in range(len(A_share0)):
        A_i = A_share0[i] + A_share1[i]
        D_i = D_share0[i] + D_share1[i]
        if not torch.allclose(A_i * A_i, D_i, atol=1e-4):
            raise ValueError(f"SC 验证失败：A[{i}]^2 != D[{i}]")


    
    delta0 = []
    delta1 = []
    for i in range(n):
        e0 = local_model_share0[i] - A_share0[i]
        e1 = local_model_share1[i] - A_share1[i]
        e = e0 + e1
        # print('='*60)
        # print('e:',e)

        # print('D_share0[i]:',D_share0[i])
        # print('local_model_share0[i]:',local_model_share0[i])
        # print('2*e*local_model_share0[i]:',2*e*local_model_share0[i])
        # print('e*e:',e*e)

        delta_i0 = D_share0[i] + 2*e*local_model_share0[i] - e*e
        delta_i1 = D_share1[i] + 2*e*local_model_share1[i] 
        # print('delta_i0:',delta_i0)
        # print('delta_i1:',delta_i1)
        delta0.append(delta_i0.sum())
        delta1.append(delta_i1.sum())


    return delta0,delta1

def gen_delta_ij(local_model_share0 , local_model_share1, A_share0, A_share1, D_share0, D_share1, args=None):
    """
    在“不可还原”约束下，以分块 + 向量化方式计算所有 (i,j) 的向量距离平方对应的算术共享。

    说明：
    - 不重构明文向量；仅在块内按原始算法对 e0/e1/E0/E1 做 share 级别的运算与求和。
    - 使用客户端维度与参数维度的双重分块以降低内存占用与 Python 循环开销。
    返回：
    - delta_ij0, delta_ij1：三角列表结构，第 i 行包含 j ∈ [0, i) 的条目。
    """
    n = len(local_model_share0)
    if n == 0:
        return [], []

    # 将各份共享堆叠为矩阵，但不做 shares 合并，不还原明文
    LM0 = torch.stack(local_model_share0)  # [n, m]
    LM1 = torch.stack(local_model_share1)  # [n, m]
    A0 = torch.stack(A_share0)             # [n, m]
    A1 = torch.stack(A_share1)             # [n, m]
    D0v = torch.stack(D_share0)            # [n, m]
    D1v = torch.stack(D_share1)            # [n, m]

    device = LM0.device
    m = LM0.shape[1]

    # 结果为三角列表结构
    delta_ij0 = [[] for _ in range(n)]
    delta_ij1 = [[] for _ in range(n)]

    # 分块大小（可按机器内存/显存调整，可从 args 注入）
    # 默认：CPU 较小块；GPU 较大块
    dev_type = device.type
    default_client_block = 16 if dev_type == 'cuda' else 16
    # default_param_block = 100000 if dev_type == 'cuda' else 100000
    default_param_block = 10000 if dev_type == 'cuda' else 100000
    
    client_block = getattr(args, 'delta_ij_client_block', default_client_block) if args is not None else default_client_block
    param_block = getattr(args, 'delta_ij_param_block', default_param_block) if args is not None else default_param_block

    # Determine computation device: use args.gpu/args.device if provided, else follow tensor's device
    compute_device = getattr(args, 'gpu', getattr(args, 'device', device)) if args is not None else device

    # 运行时打印块参数来源与设备，便于调优与确认
    src_client = 'args' if (args is not None and hasattr(args, 'delta_ij_client_block')) else 'default'
    src_param = 'args' if (args is not None and hasattr(args, 'delta_ij_param_block')) else 'default'
    try:
        print(f"[SCRAM DEBUG] delta_ij on data_device={device}, compute_device={compute_device}, client_block={client_block} ({src_client}), param_block={param_block} ({src_param})")
    except Exception:
        pass

    with torch.no_grad():
        # 遍历客户端块 (Rows i)
        for i_start in range(0, n, client_block):
            i_end = min(i_start + client_block, n)
            bi = i_end - i_start
            
            # 遍历客户端块 (Cols j)
            # 只需要遍历到 i_end，因为是下三角
            for j_start in range(0, i_end, client_block):
                j_end = min(j_start + client_block, n)
                bj = j_end - j_start
                
                # 累积矩阵块的结果 [bi, bj]
                acc0 = torch.zeros((bi, bj), device=compute_device, dtype=torch.float64)
                acc1 = torch.zeros((bi, bj), device=compute_device, dtype=torch.float64)
                
                for k_start in range(0, m, param_block):
                    k_end = min(k_start + param_block, m)
                    
                    # 取 I 块数据 [bi, bm] -> [bi, 1, bm]
                    LM0_IK = LM0[i_start:i_end, k_start:k_end].unsqueeze(1).to(compute_device)
                    LM1_IK = LM1[i_start:i_end, k_start:k_end].unsqueeze(1).to(compute_device)
                    A0_IK = A0[i_start:i_end, k_start:k_end].unsqueeze(1).to(compute_device)
                    A1_IK = A1[i_start:i_end, k_start:k_end].unsqueeze(1).to(compute_device)
                    D0_IK = D0v[i_start:i_end, k_start:k_end].unsqueeze(1).to(compute_device)
                    D1_IK = D1v[i_start:i_end, k_start:k_end].unsqueeze(1).to(compute_device)
                    
                    # 取 J 块数据 [bj, bm] -> [1, bj, bm]
                    LM0_JK = LM0[j_start:j_end, k_start:k_end].unsqueeze(0).to(compute_device)
                    LM1_JK = LM1[j_start:j_end, k_start:k_end].unsqueeze(0).to(compute_device)
                    A0_JK = A0[j_start:j_end, k_start:k_end].unsqueeze(0).to(compute_device)
                    A1_JK = A1[j_start:j_end, k_start:k_end].unsqueeze(0).to(compute_device)
                    D0_JK = D0v[j_start:j_end, k_start:k_end].unsqueeze(0).to(compute_device)
                    D1_JK = D1v[j_start:j_end, k_start:k_end].unsqueeze(0).to(compute_device)
                    
                    # 广播计算 [bi, bj, bm]
                    # e0 = LM0_I - LM0_J - A0_I
                    e0 = LM0_IK - LM0_JK - A0_IK
                    e1 = LM1_IK - LM1_JK - A1_IK
                    e = e0 + e1
                    
                    E0 = LM0_IK - LM0_JK - A0_JK
                    E1 = LM1_IK - LM1_JK - A1_JK
                    E = E0 + E1
                    
                    eE = e + E
                    
                    # val0_k = 0.5 * (D0_I + D0_J) + (LM0_I - LM0_J) * eE - 0.5 * (e * e)
                    val0_k = 0.5 * (D0_IK + D0_JK) + (LM0_IK - LM0_JK) * eE - 0.5 * (e * e)
                    val1_k = 0.5 * (D1_IK + D1_JK) + (LM1_IK - LM1_JK) * eE - 0.5 * (E * E)
                    
                    # 沿参数维度求和 [bi, bj]
                    acc0 += val0_k.sum(dim=-1)
                    acc1 += val1_k.sum(dim=-1)
                    
                    # 清理显存
                    try:
                        del LM0_IK, LM1_IK, A0_IK, A1_IK, D0_IK, D1_IK
                        del LM0_JK, LM1_JK, A0_JK, A1_JK, D0_JK, D1_JK
                        del e0, e1, e, E0, E1, E, eE, val0_k, val1_k
                    except Exception:
                        pass
                    if compute_device.type == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                            
                # 将结果分配到 delta_ij 列表
                # acc0 是 [bi, bj] 矩阵
                # 对应 delta_ij0[i_start + r]
                
                is_diagonal_block = (i_start == j_start)
                
                for r in range(bi):
                    ii = i_start + r
                    
                    if is_diagonal_block:
                        # 对角块：只取 col < row 的部分 (即 c < r)
                        # acc0[r, :r]
                        valid_len = r
                        row_vals0 = acc0[r, :valid_len]
                        row_vals1 = acc1[r, :valid_len]
                    else:
                        # 非对角块 (j < i)：取全行
                        row_vals0 = acc0[r]
                        row_vals1 = acc1[r]
                    
                    # 转移到 CPU 并添加到列表
                    delta_ij0[ii].extend(row_vals0.detach().cpu().tolist())
                    delta_ij1[ii].extend(row_vals1.detach().cpu().tolist())
                
                try:
                    del acc0, acc1
                except Exception:
                    pass
                if compute_device.type == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    print("finish delta_ij")
    return delta_ij0, delta_ij1

def div(x_share0,x_share1,y1_share0,y1_share1,y2_share0,y2_share1,args):
    # 使用全局密钥对和随机数
    public_key = args.paillier_public_key
    private_key = args.paillier_private_key
    R = random.randint(10, 100)
    scale_factor = args.paillier_scale_factor

    # 兼容张量/标量，稳定转换为整数
    def _to_int(val):
        if isinstance(val, torch.Tensor):
            return int((val * scale_factor).item())
        return int(val * scale_factor)

    x0_int = _to_int(x_share0)
    x1_int = _to_int(x_share1)
    y10_int = _to_int(y1_share0)
    y11_int = _to_int(y1_share1)
    y20_int = _to_int(y2_share0)
    y21_int = _to_int(y2_share1)

    gx0 = public_key.encrypt(x0_int)
    gy10 = public_key.encrypt(y10_int)
    gy20 = public_key.encrypt(y20_int)

    # 在不用到 x0 和 y10、y20 的情况下利用 gx0、gy0 计算密文
    gx0R = gx0 * R
    gy10R = gy10 * R
    gy20R = gy20 * R

    gx1R = public_key.encrypt(R*x1_int)
    gy11R = public_key.encrypt(R*y11_int)
    gy21R = public_key.encrypt(R*y21_int)

    gxR = gx0R + gx1R
    gy1R = gy10R + gy11R
    gy2R = gy20R + gy21R

    xR = private_key.decrypt(gxR)
    y1R = private_key.decrypt(gy1R)
    y2R = private_key.decrypt(gy2R)

    # 将解密得到的整数转换为张量以进行 sqrt，并匹配设备与精度
    dtype = torch.float64
    device = getattr(args, 'gpu', torch.device('cpu'))
    xR_t = torch.tensor(float(xR), device=device, dtype=dtype)
    y1R_t = torch.tensor(float(y1R), device=device, dtype=dtype)
    y2R_t = torch.tensor(float(y2R), device=device, dtype=dtype)
    denom = torch.sqrt(torch.clamp(y1R_t * y2R_t, min=1e-12))
    result = 1.0 - (xR_t / denom)
    return float(result.item())


def _to_int_scaled(val, scale_factor: int) -> int:
    """辅助：将张量或数值按缩放因子转为整数，并对 NaN/Inf 做稳健清洗。

    行为：
    - 对输入转换为 Python float；若为非有限（NaN/Inf），回退为 0.0 并计数。
    - 使用 round 进行缩放转换，避免由于浮点误差导致整数截断偏差。
    """
    # 转为 Python float
    if isinstance(val, torch.Tensor):
        try:
            return int((val * scale_factor).item())
        except Exception:
            return int((val.detach().cpu() * scale_factor).item())
    return int(val * scale_factor)


def precompute_y_ciphertexts_for_clients(delta0, delta1, args):
    """预加密每个客户端的 y10 和 y11（两份共享），以便在不同随机 R 下快速得到 y1R 的密文。

    返回：
    - gy10_list, gy11_list：长度为 n 的列表，元素为加密后的整数（Paillier EncryptedNumber）
    """
    public_key = args.paillier_public_key
    scale_factor = int(args.paillier_scale_factor)

    n = len(delta0)
    gy10_list = [None] * n
    gy11_list = [None] * n
    for k in range(n):
        y10_int = _to_int_scaled(delta0[k], scale_factor)
        y11_int = _to_int_scaled(delta1[k], scale_factor)
        gy10_list[k] = public_key.encrypt(y10_int)
        gy11_list[k] = public_key.encrypt(y11_int)
    return gy10_list, gy11_list


def precompute_delta_ciphertexts_for_clients(delta0, delta1, args):
    """预加密每个客户端的 delta0[i], delta1[i]，用于构造分子密文。"""
    public_key = args.paillier_public_key
    scale_factor = int(args.paillier_scale_factor)

    n = len(delta0)
    g_d0_list = [None] * n
    g_d1_list = [None] * n
    for i in range(n):
        d0_int = _to_int_scaled(delta0[i], scale_factor)
        d1_int = _to_int_scaled(delta1[i], scale_factor)
        g_d0_list[i] = public_key.encrypt(d0_int)
        g_d1_list[i] = public_key.encrypt(d1_int)
    return g_d0_list, g_d1_list


def compute_xR_from_cipher_parts(i, j, delta_ij0, delta_ij1, g_d0_list, g_d1_list, R: int, args) -> float:
    """使用预加密的 per-client 组件与本次 (i,j) 的 delta_ij 构造分子密文并解密得到 xR。"""
    public_key = args.paillier_public_key
    private_key = args.paillier_private_key
    scale_factor = int(args.paillier_scale_factor)

    aij0_int = _to_int_scaled(delta_ij0[i][j], scale_factor)
    aij1_int = _to_int_scaled(delta_ij1[i][j], scale_factor)
    g_aij0 = public_key.encrypt(aij0_int)
    g_aij1 = public_key.encrypt(aij1_int)

    g_x0_nohalf = g_d0_list[i] + g_d0_list[j] + (g_aij0 * -1)
    g_x1_nohalf = g_d1_list[i] + g_d1_list[j] + (g_aij1 * -1)
    g_x_nohalf_sum = g_x0_nohalf + g_x1_nohalf
    g_xR = g_x_nohalf_sum * R
    xR_int = private_key.decrypt(g_xR)
    return 0.5 * float(xR_int)


def div_with_precomputed_yR(x_share0, x_share1, y1R_val: float, y2R_val: float, R: int, args) -> float:
    """使用已缓存的 yR（两个端点的合并共享乘以 R 的解密值），仅对 x 进行一次加密/解密。

    保留同态加密的使用：
    - 对 x_share0 执行加密并乘以 R；对 R*x_share1 执行加密；两者相加得到 gxR，解密为 xR。
    - 使用预先解密的 y1R_val, y2R_val 计算分母 sqrt(y1R_val * y2R_val)。
    返回：
    - 1 - xR / sqrt(y1R * y2R)
    """
    public_key = args.paillier_public_key
    private_key = args.paillier_private_key
    scale_factor = int(args.paillier_scale_factor)

    x0_int = _to_int_scaled(x_share0, scale_factor)
    x1_int = _to_int_scaled(x_share1, scale_factor)

    gx0 = public_key.encrypt(x0_int)
    gx0R = gx0 * R
    gx1R = public_key.encrypt(R * x1_int)
    gxR = gx0R + gx1R
    xR = float(private_key.decrypt(gxR))

    # 使用纯 Python 计算，避免 GPU/CPU 往返与张量构造开销
    denom = math.sqrt(max(y1R_val * y2R_val, 1e-12))
    return 1.0 - (xR / denom)


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)



def scram(local_model, update_params, global_model, args):
    # print("into scram")
    debug = getattr(args, 'debug_scram', False)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-4).to(args.gpu)
    cos_list=[]
    local_model_vector = []
    update_model_vector = []
    if debug:
        _safe_print('start', {'num_clients': len(local_model), 'num_updates': len(update_params)})
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        flt_para = parameters_dict_to_vector_flt(param)
        local_model_vector.append(flt_para)

    for param in update_params:
        update_model_vector.append(parameters_dict_to_vector_flt(param))
    m = len(local_model_vector[0])
    n = len(local_model_vector)
    print(m)
    if debug:
        _safe_print('vectors_built', {'m': m, 'n': n, 'sample_vec_stats': _tensor_stats(local_model_vector[0])})

    # 计算本地模型的算术共享
    
    start_time = time.time()
    local_model_share0 , local_model_share1 = models_arithmetic_share(local_model_vector)
    elapsed = time.time() - start_time
    print(f"[SCRAM DEBUG] models_arithmetic_share 用时: {elapsed:.6f} 秒")


    if debug:
        _safe_print('model_shares', {
            'share0_len': len(local_model_share0),
            'share1_len': len(local_model_share1),
            'sample_share0': _tensor_stats(local_model_share0[0]),
            'sample_share1': _tensor_stats(local_model_share1[0])
        })

    # for i in range(n):
    #     print('local_model_vector[i]:',local_model_vector[i])
    #     print('local_model_share0[i]:',local_model_share0[i])
    #     print('local_model_share1[i]:',local_model_share1[i])
    #     print('local_model_share0[i] + local_model_share1[i]:',local_model_share0[i] + local_model_share1[i])
    # 计算每个客户端的SC，再拆分成算术共享
    start_sc = time.time()  # 记录SC生成与共享拆分开始时间
    A0_list = []
    D0_list = []
    A1_list = []
    D1_list = []
    A0_list_share0 = []
    A0_list_share1 = []
    D0_list_share0 = []
    D0_list_share1 = []
    A1_list_share0 = []
    A1_list_share1 = []
    D1_list_share0 = []
    D1_list_share1 = []
    for i in range(n):
        A0 , D0 =gen_sc(local_model_vector[i])
        A1 , D1 =gen_sc(local_model_vector[i])
        A0_list.append(A0)
        D0_list.append(D0)
        A1_list.append(A1)
        D1_list.append(D1)
        A0_share0,A0_share1 = arithmetic_share(A0)
        A1_share0,A1_share1 = arithmetic_share(A1)
        A0_list_share0.append(A0_share0)
        A0_list_share1.append(A0_share1)
        A1_list_share0.append(A1_share0)
        A1_list_share1.append(A1_share1)
        D0_share0,D0_share1 = arithmetic_share(D0)
        D1_share0,D1_share1 = arithmetic_share(D1)
        D0_list_share0.append(D0_share0)
        D0_list_share1.append(D0_share1)
        D1_list_share0.append(D1_share0)
        D1_list_share1.append(D1_share1)
    elapsed_sc = time.time() - start_sc
    print(f"[SCRAM DEBUG] SC生成与共享拆分 用时: {elapsed_sc:.6f} 秒")
    if debug and n > 0:
        _safe_print('sc_generated', {
            'count': n,
            'A0_sample': _tensor_stats(A0_list[0]),
            'D0_sample': _tensor_stats(D0_list[0])
        })

    # 验证SC 并得到最后的SC共享，为了方便只验证了一组SC，实际上应该生成两组SC，一组用来计算L2范数平方，一组用来计算向量之间距离的平方
    A_share0 = []
    A_share1 = []
    D_share0 = []
    D_share1 = []
    verified_count = 0
    start_verify = time.time()  # 记录SC验证开始时间
    for i in range(n):
        if verify_sc_shares(A0_list_share0[i], A0_list_share1[i], D0_list_share0[i], D0_list_share1[i],A1_list_share0[i], A1_list_share1[i], D1_list_share0[i], D1_list_share1[i]):
            A_share0.append(A0_list_share0[i])
            A_share1.append(A0_list_share1[i])
            D_share0.append(D0_list_share0[i])
            D_share1.append(D0_list_share1[i])
            verified_count += 1
    elapsed_verify = time.time() - start_verify
    print(f"[SCRAM DEBUG] SC验证 用时: {elapsed_verify:.6f} 秒")
    if debug:
        _safe_print('sc_verified', {'verified': verified_count, 'expected': n})

    # TODO：在用时的SC的产生和验证的时候要乘以2
    # 计算delta 即向量的l2范数平方的算术共享
    start_delta = time.time()  # 记录delta计算开始时间
    delta0,delta1 = gen_delta(local_model_share0 , local_model_share1,A_share0, A_share1,D_share0, D_share1)
    elapsed_delta = time.time() - start_delta
    print(f"[SCRAM DEBUG] delta计算 用时: {elapsed_delta:.6f} 秒")
    if debug and len(delta0) > 0:
        _safe_print('delta', {'len': len(delta0), 'sample0': float(delta0[0].item()), 'sample1': float(delta1[0].item())})
    # for i in range(n):
    #     print('delta0[i]:',delta0[i])
    #     print('delta1[i]:',delta1[i])
    #     print('delta0[i] + delta1[i]:',delta0[i] + delta1[i])
    #     print('local_model_vector[i] L2 norm squared:', torch.norm(local_model_vector[i], p=2).item() ** 2)




    # 计算delta_ij 即向量之间的距离平方,为了计算方便服用了前面SC，真正的方案中需要用新的SC
    start_delta_ij = time.time()  # 记录delta_ij计算开始时间
    delta_ij0,delta_ij1 = gen_delta_ij(local_model_share0 , local_model_share1, A_share0, A_share1, D_share0, D_share1, args=args)
    elapsed_delta_ij = time.time() - start_delta_ij
    print(f"[SCRAM DEBUG] delta_ij计算 用时: {elapsed_delta_ij:.6f} 秒")
    if debug:
        _safe_print('delta_ij', {'len': len(delta_ij0), 'len_row0': len(delta_ij0[0]) if len(delta_ij0)>0 else 0})
    # 额外校验：每行 delta_ij 应有 i 个条目（j∈[0,i)）。若不满足，记录告警，避免后续越界。
    try:
        row_lens0 = [len(r) for r in delta_ij0]
        row_lens1 = [len(r) for r in delta_ij1]
        for i_idx in range(n):
            if row_lens0[i_idx] != i_idx or row_lens1[i_idx] != i_idx:
                print(f"[SCRAM WARN] delta_ij 行 {i_idx} 长度异常: 0={row_lens0[i_idx]}, 1={row_lens1[i_idx]}, 期望={i_idx}")
    except Exception:
        pass


    start_gen_keys = time.time()  # 记录全局密钥生成开始时间
    # 全局密钥生成 - 确保整个训练过程中只生成一次（若缺失或为None则重新生成）
    need_generate_keys = (
        not hasattr(args, 'paillier_public_key') or args.paillier_public_key is None or
        not hasattr(args, 'paillier_private_key') or args.paillier_private_key is None or
        not hasattr(args, 'paillier_R') or args.paillier_R is None or
        not hasattr(args, 'paillier_scale_factor') or args.paillier_scale_factor is None
    )
    if need_generate_keys:
        key_length = 256
        args.paillier_public_key, args.paillier_private_key = paillier.generate_paillier_keypair(n_length=key_length)
        args.paillier_R = random.randint(10, 100)
        # 设置缩放因子（适中，避免溢出）
        args.paillier_scale_factor = 1000
    else:
        print("复用已生成的Paillier密钥对和随机数R")
    if debug:
        _safe_print('paillier_keys', {'generated': bool(need_generate_keys), 'R': int(args.paillier_R), 'scale': int(args.paillier_scale_factor)})
    elapsed_gen_keys = time.time() - start_gen_keys
    print(f"[SCRAM DEBUG] 全局密钥生成 用时: {elapsed_gen_keys:.6f} 秒")
    


    start_cos = time.time()  # 记录余弦相似度计算开始时间
    # 使用全局密钥对和随机数
    public_key = args.paillier_public_key
    private_key = args.paillier_private_key
    # 预加密每个客户端的 y10/y11（两份共享），以便在循环内对当前 R 生成 yR 密文
    gy10_list, gy11_list = precompute_y_ciphertexts_for_clients(delta0, delta1, args)
    # 分子组件预加密（每个客户端的 delta0/1），用于快速组合 x 的密文
    g_d0_list, g_d1_list = precompute_delta_ciphertexts_for_clients(delta0, delta1, args)
    # 预先解密每个客户端的 y_sum = y10 + y11（缩放域的整数），后续按 R 做纯乘法得到 yR
    y_sum_plain = []
    private_key = args.paillier_private_key
    for k in range(n):
        y_sum_plain.append(float(private_key.decrypt(gy10_list[k] + gy11_list[k])))

    # 初始化固定大小的矩阵，避免行长度不一致导致索引越界
    cos_distance = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        # 安全索引：仅计算可用的 j 范围，避免越界
        j_max = min(i, len(delta_ij0[i]) if i < len(delta_ij0) else 0, len(delta_ij1[i]) if i < len(delta_ij1) else 0)
        for j in range(j_max):
            # 将分子 x 的两份共享转为 Python 浮点数，避免跨设备张量混算
            x0_val = 0.5 * (float(delta0[i].item()) + float(delta0[j].item()) - float(delta_ij0[i][j].item()))
            x1_val = 0.5 * (float(delta1[i].item()) + float(delta1[j].item()) - float(delta_ij1[i][j].item()))
            # 为每次 (i,j) 使用一个新的随机 R，保持算法安全性
            R_ij = random.randint(10, 100)
            # 使用预先解密的 y_sum_plain，在本次 R 下得到 yR（缩放域的整数）
            y1R_val = R_ij * y_sum_plain[i]
            y2R_val = R_ij * y_sum_plain[j]

            # 构造分子密文并解密得到 xR（避免在循环内重复加密原子组件）
            xR_val = compute_xR_from_cipher_parts(i, j, delta_ij0, delta_ij1, g_d0_list, g_d1_list, R_ij, args)
            # 直接用 xR_val（已为 R*x 的数值）与 y1R_val, y2R_val 组合计算余弦距离
            cos_distance[j][i] = cos_distance[i][j] = 1.0 - (
                xR_val / math.sqrt(max(y1R_val * y2R_val, 1e-12))
            )
        # 未计算到的 j（若存在）置 0，保证矩阵完整
        for j in range(j_max, i):
            cos_distance[j][i] = cos_distance[i][j] = 0.0
        # 对角置 0（自身距离为 0）
        cos_distance[i][i] = 0.0
    
     
    if debug and n > 1:
        _safe_print('cos_distance_div', {'rows': len(cos_distance), 'row1_len': len(cos_distance[1]) if len(cos_distance)>1 else 0})
    elapsed_cos = time.time() - start_cos
    print(f"[SCRAM DEBUG] 余弦相似度计算 用时: {elapsed_cos:.6f} 秒")
    print("余弦相似度",cos_distance)

    # print('cos_distance:',type(cos_distance))
    # for i in range(len(cos_distance)):
    #     print(cos_distance[i])



    
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            # 利用公式定义准确计算余弦距离：1 - 余弦相似度
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug:
        _safe_print('cosine_matrix', {'rows': len(cos_list), 'cols': len(cos_list[0]) if len(cos_list)>0 else 0, 'sample_row0': [float(x) for x in (cos_list[0][:3] if len(cos_list)>0 else [])]})
    
    # print('cos_list:',type(cos_list))
    # for i in range(len(cos_list)):
    #     print(cos_list[i])
    
    
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    
    print("HDBSCAN 聚类结果:",clusterer.labels_)

    if debug:
        labels = clusterer.labels_
        vals, counts = np.unique(labels, return_counts=True)
        _safe_print('clusters', {'labels': labels.tolist(), 'unique': vals.tolist(), 'counts': counts.tolist()})
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print("HDBSCAN 正常客户端索引:",benign_client)
    if debug:
        if norm_list.size > 0:
            _safe_print('norm_list_stats', {'len': int(norm_list.size), 'min': float(norm_list.min()), 'max': float(norm_list.max()), 'median': float(np.median(norm_list))})
        else:
            _safe_print('norm_list_stats', {'len': 0})

    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn += 1
    mal_denom = num_malicious_clients * args.turn
    ben_denom = num_benign_clients * args.turn
    mal_ratio = (args.wrong_mal / mal_denom) if mal_denom > 0 else 0.0
    ben_ratio = (args.right_ben / ben_denom) if ben_denom > 0 else 0.0
    print('proportion of malicious are selected:', mal_ratio)
    print('proportion of benign are selected:', ben_ratio)
    

    # 使用IQR算法对模长进行筛除
    filtered_benign_client = filter_benign_clients_by_iqr(
        benign_client,
        delta0, delta1,
        args,
        whisker_k=1.5
    )
    print("IQR 过滤后的良性客户端索引:",filtered_benign_client)
    print("恶意客户端下标",args.attackers_chosen[-1])

    # 仅在 IQR 过滤出的良性客户端上进行加权聚合
    candidates = filtered_benign_client
    if len(candidates) == 0:
        print("[SCRAM] IQR 过滤后无良性客户端，回退为均匀平均聚合。")
        agg_model = no_defence_balance(update_params, global_model)
        return agg_model

    # 改为不加权的平均聚合（在全局参数上累加平均更新）
    print("[SCRAM] 候选良性客户端:", candidates)
    subset_updates = [update_params[idx] for idx in candidates]
    # print("[SCRAM] 使用不加权平均聚合，候选数=", len(candidates))
    # agg_model = weighted_defence_balance(subset_updates, global_model, None)
    print("[SCRAM] 使用加权平均聚合，候选数=", len(candidates))
    agg_model = scram_weighted_aggregate(update_params,global_model,candidates,cos_list)
    return agg_model


def scram_without_privacy(local_model, update_params, global_model, args):
    """
    纯明文版本的 SCRAM：省略算术共享、SC 生成与同态加密等隐私计算，
    直接使用明文向量计算余弦距离与 L2 范数，并按 scram 函数的流程执行
    HDBSCAN 聚类、最大簇选取、IQR 过滤和最终的聚合步骤。

    入参:
    - local_model: List[Dict[str, Tensor]] 本地模型参数字典列表
    - update_params: List[Dict[str, Tensor]] 各客户端更新量 (w_local - w_glob)
    - global_model: Dict[str, Tensor] 当前全局模型参数
    - args: 运行参数（需包含 gpu, frac, num_users, malicious 等字段）

    返回:
    - Dict[str, Tensor] 聚合后的全局参数（在当前全局上累加聚合更新）
    """
    debug = getattr(args, 'debug_scram', False)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-4).to(args.gpu)

    # 将模型参数转换为向量（与 scram 中一致）
    local_model_vector = []
    for param in local_model:
        try:
            flt_para = parameters_dict_to_vector_flt(param)
        except Exception:
            # 兜底：若 FP16/设备相关版本不可用，使用通用版本
            flt_para = parameters_dict_to_vector(param)
        local_model_vector.append(flt_para)

    if len(local_model_vector) == 0:
        print("[SCRAM/plain] 无本地模型向量，直接回退到不加权平均聚合。")
        return no_defence_balance(update_params, global_model)

    m = len(local_model_vector[0])
    n = len(local_model_vector)
    if debug:
        print(f"[SCRAM/plain] 向量维度 m={m}, 客户端数 n={n}")
    # 诊断：输入与向量健康性
    try:
        print(
            f"[SCRAM/plain DIAG] m={m}, n={n}, frac={getattr(args, 'frac', None)}, num_users={getattr(args, 'num_users', None)}, malicious={getattr(args, 'malicious', None)}"
        )
        zero_norm_vec = 0
        nonfinite_vec = 0
        for vec in local_model_vector:
            try:
                nv = float(torch.norm(vec, p=2).item())
                if nv == 0.0:
                    zero_norm_vec += 1
                if vec.dtype.is_floating_point:
                    # float 类型才可 isfinite
                    if not bool(torch.isfinite(vec).all().item()):
                        nonfinite_vec += 1
            except Exception:
                nonfinite_vec += 1
        print(f"[SCRAM/plain DIAG] 本地向量零范数数={zero_norm_vec}, 非有限向量数={nonfinite_vec}")
    except Exception as e:
        print(f"[SCRAM/plain DIAG] 向量统计失败: {e}")

    # 直接用明文计算余弦距离矩阵：d(i,j) = 1 - cos(vec_i, vec_j)
    cos_list = []
    for i in range(n):
        cos_i = []
        for j in range(n):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            try:
                cos_i.append(float(cos_ij.item()))
            except Exception:
                cos_i.append(float(cos_ij))
        cos_list.append(cos_i)

    if debug:
        print("[SCRAM/plain] 余弦距离矩阵已计算，示例前3项:", cos_list[0][:3] if len(cos_list) > 0 else [])
    # 诊断：余弦矩阵健康性
    try:
        mat = np.array(cos_list, dtype=np.float64)
        shape = tuple(mat.shape)
        nan_count = int(np.isnan(mat).sum())
        inf_count = int(np.isinf(mat).sum())
        min_val = float(np.nanmin(mat)) if mat.size > 0 else 0.0
        max_val = float(np.nanmax(mat)) if mat.size > 0 else 0.0
        diag_nonzero = int(np.sum(np.abs(np.diag(mat)) > 1e-6)) if mat.size > 0 else 0
        sym_ok = bool(np.allclose(mat, mat.T, atol=1e-6, rtol=1e-6)) if mat.size > 0 else True
        print(
            f"[SCRAM/plain DIAG] 余弦矩阵 shape={shape}, min={min_val:.6f}, max={max_val:.6f}, NaN={nan_count}, Inf={inf_count}, 对称性={sym_ok}, 非零对角={diag_nonzero}"
        )
    except Exception as e:
        print(f"[SCRAM/plain DIAG] 余弦矩阵统计失败: {e}")

    # HDBSCAN 聚类（与 scram 的参数保持一致）
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    print(f"[SCRAM/plain DIAG] HDBSCAN 输入 行数={len(cos_list)}, 列数={len(cos_list[0]) if len(cos_list) > 0 else 0}, min_cluster_size={num_clients // 2 + 1}")
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=num_clients // 2 + 1,
            min_samples=1,
            allow_single_cluster=True
        ).fit(cos_list)
        labels = clusterer.labels_
    except Exception as e:
        print("[SCRAM/plain] HDBSCAN 聚类失败，原因:", e)
        labels = np.zeros(n, dtype=int)

    print("[SCRAM/plain] HDBSCAN 聚类结果:", labels)
    try:
        vals, counts = np.unique(labels, return_counts=True)
        print(f"[SCRAM/plain DIAG] 聚类标签分布: {list(zip(vals.tolist(), counts.tolist()))}")
    except Exception:
        pass

    # 选择最大簇作为初步良性集合（若全为噪声则选全部）
    benign_client = []
    if labels.max() < 0:
        benign_client = list(range(n))
    else:
        max_num_in_cluster = 0
        max_cluster_index = 0
        for index_cluster in range(labels.max() + 1):
            count = len(labels[labels == index_cluster])
            if count > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = count
        for i in range(n):
            if labels[i] == max_cluster_index:
                benign_client.append(i)
    print(f"[SCRAM/plain DIAG] 初筛良性数量={len(benign_client)}")

    # 统计并打印选择比例（复刻原 scram 的打印逻辑）
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal += 1
        else:
            args.right_ben += 1
    args.turn += 1
    mal_denom = num_malicious_clients * args.turn
    ben_denom = num_benign_clients * args.turn
    mal_ratio = (args.wrong_mal / mal_denom) if mal_denom > 0 else 0.0
    ben_ratio = (args.right_ben / ben_denom) if ben_denom > 0 else 0.0
    print('[SCRAM/plain] 恶意被选比例:', mal_ratio)
    print('[SCRAM/plain] 良性被选比例:', ben_ratio)

    # 使用明文 L2 范数进行 IQR 过滤（不依赖 delta0/1）
    norms = []
    invalid_update = 0
    zero_update_norm = 0
    for i in range(n):
        try:
            v = parameters_dict_to_vector(update_params[i])
            nv = float(torch.norm(v, p=2).item())
            norms.append(nv)
            if nv == 0.0:
                zero_update_norm += 1
            if v.dtype.is_floating_point:
                try:
                    if not bool(torch.isfinite(v).all().item()):
                        invalid_update += 1
                except Exception:
                    invalid_update += 1
        except Exception:
            norms.append(0.0)
            invalid_update += 1
    try:
        print(f"[SCRAM/plain DIAG] 更新零范数数={zero_update_norm}, 非有限更新数={invalid_update}")
    except Exception:
        pass

    arr = np.array([norms[idx] for idx in benign_client], dtype=np.float64)
    filtered_benign_client = benign_client[:]
    print(f"[SCRAM/plain DIAG] 良性集合范数样本量={arr.size}")
    if arr.size > 0 and np.all(np.isfinite(arr)):
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        print(f"[SCRAM/plain DIAG] IQR统计: q1={q1:.6f}, q3={q3:.6f}, iqr={iqr:.6f}")
        if np.isfinite(iqr) and iqr > 0:
            # Use args.mul instead of hardcoded 1.5
            mul_val = getattr(args, 'mul', 1.5)
            upper = q3 + mul_val * iqr
            print(f"[SCRAM/plain DIAG] IQR上界={upper:.6f} (mul={mul_val})")
            tmp = []
            for idx, val in zip(benign_client, arr.tolist()):
                if np.isfinite(val) and val <= upper:
                    tmp.append(idx)
            if len(tmp) == 0:
                # 兜底：保留最接近中位数的一个
                median = float(np.median(arr))
                best = int(np.argmin(np.abs(arr - median)))
                tmp = [benign_client[best]]
                print(f"[SCRAM/plain DIAG] IQR全剔除，回退保留中位数邻近客户端 idx={benign_client[best]}")
            print(f"[SCRAM/plain DIAG] IQR过滤前={len(benign_client)}, 过滤后={len(tmp)}")
            filtered_benign_client = tmp
        else:
            print("[SCRAM/plain DIAG] IQR<=0 或非有限，跳过过滤。")
    else:
        print("[SCRAM/plain DIAG] 范数数组为空或存在非有限值，跳过IQR。")

    print("[SCRAM/plain] IQR 过滤后的良性客户端索引:", filtered_benign_client)
    if hasattr(args, 'attackers_chosen') and isinstance(args.attackers_chosen, (list, tuple)) and len(args.attackers_chosen) > 0:
        print("[SCRAM/plain] 恶意客户端下标", args.attackers_chosen[-1])

    # 聚合：与 scram 保持一致，使用加权版（基于候选之间的相似度/距离）
    candidates = filtered_benign_client
    if len(candidates) == 0:
        print("[SCRAM/plain] IQR 过滤后无良性客户端，回退为均匀平均聚合。")
        return no_defence_balance(update_params, global_model)

    print("[SCRAM/plain] 候选良性客户端:", candidates)
    agg_model = scram_weighted_aggregate(update_params, global_model, candidates, cos_list)
    return agg_model





def run_scram_from_flame_stream(stream_path, start_epoch=None, end_epoch=None, device=None, dry_run=False):
    """
    从由 append_flame_epoch 生成的流式文件中按顺序读取
    (epoch, w_locals, w_updates, w_glob, args)，并将参数传入 scram。
    以流式迭代的方式逐条处理，避免一次性加载造成显存/内存压力。
    参数:
    - stream_path: 数据文件路径，如 logs/.../flame_io_stream.pkl
    - start_epoch, end_epoch: 可选的轮次范围过滤
    - device: 可选的运行设备覆盖，如 'cuda' 或 'cpu'；若不指定则使用记录中的 args.gpu
    - dry_run: 为 True 时仅打印信息，不执行 scram
    返回:
    - 生成器，逐条返回 (epoch, aggregated_global_model 或 None[dry_run])
    """
    try:
        with open(stream_path, 'rb') as f:
            while True:
                try:
                    payload = pickle.load(f)
                except EOFError:
                    break
                except Exception as e:
                    print(f"[run_scram_from_flame_stream] 读取失败: {e}")
                    break

                epoch = payload.get('epoch')
                w_locals = payload.get('w_locals') or []
                w_updates = payload.get('w_updates') or []
                w_glob = payload.get('w_glob') or {}
                args = payload.get('args')

                if start_epoch is not None and epoch is not None and epoch < start_epoch:
                    continue
                if end_epoch is not None and epoch is not None and epoch > end_epoch:
                    continue

                # 设备设置：优先使用调用方指定的 device，否则用记录中的 args.gpu，兜底为 CPU
                target_device = torch.device(device) if device is not None else getattr(args, 'gpu', torch.device('cpu'))
                try:
                    args.gpu = target_device
                except Exception:
                    pass

                def _to_device_dict(d, dev):
                    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
                def _to_device_list_of_dicts(lst, dev):
                    return [_to_device_dict(sd, dev) for sd in lst]

                w_locals_dev = _to_device_list_of_dicts(w_locals, target_device)
                w_updates_dev = _to_device_list_of_dicts(w_updates, target_device)
                w_glob_dev = _to_device_dict(w_glob, target_device)

                if dry_run:
                    print(f"[run_scram_from_flame_stream] epoch={epoch} clients={len(w_locals_dev)} (dry_run)")
                    yield (epoch, None)
                    continue

                try:
                    aggregated_global = scram(w_locals_dev, w_updates_dev, w_glob_dev, args)
                except Exception as e:
                    print(f"[run_scram_from_flame_stream] scram执行失败: {e}")
                    aggregated_global = None
                yield (epoch, aggregated_global)
    except FileNotFoundError:
        print(f"[run_scram_from_flame_stream] 文件未找到: {stream_path}")
    except Exception as e:
        print(f"[run_scram_from_flame_stream] 打开文件失败: {e}")

# 指定路径便捷包装
DEFAULT_FLAME_STREAM_PATH = '/home/jfl/code/FLAME-main/logs/save/my_experiments/dba/avg/2.0/fashion_mnist__iid1cnn_avg_2025-11-20_21-33-43_dba_0.1malicious_1.0poisondata/flame_io_stream.pkl'

def run_scram_from_default_flame_stream(start_epoch=None, end_epoch=None, device=None, dry_run=False):
    """
    便捷函数：使用用户给定的固定路径读取，并逐轮调用 scram。
    返回生成器，逐条产出 (epoch, aggregated_global_model)。
    """
    return run_scram_from_flame_stream(DEFAULT_FLAME_STREAM_PATH, start_epoch, end_epoch, device, dry_run)


def filter_benign_clients_by_iqr(benign_client, delta0 , delta1 , args, whisker_k: float = 1.5):
    """
    使用 IQR 算法在良性客户端集合中进一步剔除 L2 范数异常的客户端。

    入参：
    - benign_client: HDBSCAN 初筛得到的良性客户端索引列表
    - update_params: 所有客户端的更新参数列表（字典构成），与 benign_client 的索引对应
    - whisker_k: IQR 胡须系数，默认 1.5

    返回：
    - 过滤后的良性客户端索引列表（顺序保持输入顺序）
    """
    debug = getattr(args, 'debug_scram', True) if args is not None else True
    if benign_client is None or len(benign_client) == 0:
        if debug:
            _safe_print('iqr_filter', {'count_in': 0, 'count_out': 0})
        return []

    norms = []
    for idx in benign_client:
        # 使用 delta0[idx] + delta1[idx] 的数值（标量），避免将 CUDA 张量直接传入 numpy
        val = delta0[idx] + delta1[idx]
        try:
            norms.append(float(val.item()))
        except Exception:
            # 兜底：若 .item() 不可用（极少数情况），先转 CPU 再取值
            norms.append(float(val.detach().cpu().item()))

    arr = np.array(norms, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        if debug:
            _safe_print('iqr_filter', {'count_in': int(len(benign_client)), 'count_out': int(len(benign_client)), 'reason': 'empty_or_nonfinite'})
        return benign_client[:]

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr <= 0:
        # IQR 无法有效区分，则直接返回原集合
        if debug:
            _safe_print('iqr_filter', {'count_in': int(len(benign_client)), 'count_out': int(len(benign_client)), 'q1': q1, 'q3': q3, 'iqr': float(iqr), 'reason': 'iqr_nonpositive'})
        return benign_client[:]

    # lower = q1 - whisker_k * iqr
    upper = q3 + whisker_k * iqr

    filtered = [idx for idx, val in zip(benign_client, arr.tolist()) if (np.isfinite(val) and  val <= upper)]


    if len(filtered) == 0:
        # 兜底：若全部被剔除，则保留最接近中位数的一个客户端，避免空集导致后续聚合失败
        median = float(np.median(arr))
        best = int(np.argmin(np.abs(arr - median)))
        filtered = [benign_client[best]]
    # if debug:
    #     _safe_print('iqr_filter', {
    #         'count_in': int(len(benign_client)),
    #         'count_out': int(len(filtered)),
    #         'q1': q1,
    #         'q3': q3,
    #         'iqr': float(iqr),
    #         'lower': float(lower),
    #         'upper': float(upper),
    #     })

    return filtered

# =========================
# 轻量测试辅助与示例测试用例
# =========================

# 说明：以下测试函数旨在对本文件中的核心方法进行基本功能验证，
# 包括算术共享、SC生成与验证、参数字典向量化等。测试尽量使用CPU，
# 并提供最小化依赖环境（如伪造的 args 与 computation_strategy）。


class _TestArgs:
    """为测试构造的最小化参数对象"""
    def __init__(self, device: str = 'cpu'):
        # 设备：默认CPU，若可用且需要可设为 'cuda'
        self.gpu = torch.device(device)
        # 供 scram 使用的必要属性（最小化设置）
        self.frac = 1.0
        self.num_users = 3
        self.malicious = 0.0
        self.wrong_mal = 0
        self.right_ben = 0
        self.turn = 0
        self.noise = 0.0
        # Paillier 相关占位（按需在测试中初始化）
        self.paillier_public_key = None
        self.paillier_private_key = None
        self.paillier_R = None
        self.paillier_scale_factor = None


def _set_test_globals(use_fp16: bool = False, device: str = 'cpu'):
    """设置测试所需的全局变量（args 与 computation_strategy）。"""
    global args, computation_strategy
    args = _TestArgs(device=device)
    computation_strategy = {'use_fp16': use_fp16}


def test_arithmetic_share_basic(shape=(4, 5), use_fp16=False, device='cpu'):
    """测试 arithmetic_share：验证两份共享求和是否还原原始张量。"""
    _set_test_globals(use_fp16=use_fp16, device=device)
    x = torch.randn(shape, device=args.gpu, dtype=torch.float32)
    s1, s2 = arithmetic_share(x)
    restored = s1 + s2
    # arithmetic_share 内部将原始 x 转为所选 dtype，因此比较时也转型
    x_cast = x.to(s1.dtype)
    passed = torch.allclose(restored, x_cast, atol=1e-3, rtol=1e-3)
    return {
        'test': 'arithmetic_share_basic',
        'shape': shape,
        'use_fp16': use_fp16,
        'device': device,
        'passed': bool(passed),
        'max_abs_err': float((restored - x_cast).abs().max().item())
    }


def test_models_arithmetic_share_basic(num_tensors=3, tensor_size=10, device='cpu'):
    """测试 models_arithmetic_share：检查返回结构与还原正确性。

    注意：当前实现将 (share1, share2) 元组分别追加到两个列表中，
    使得返回的两个列表元素均为元组，后续使用可能产生不符合预期的类型。
    本测试会检测此结构并尝试验证每个模型的还原正确性。"""
    _set_test_globals(use_fp16=False, device=device)
    models = [torch.randn(tensor_size, device=args.gpu) for _ in range(num_tensors)]
    s0_list, s1_list = models_arithmetic_share(models)

    # 结构校验：期望每个元素是张量，但当前实现返回的是元组
    structure_ok = all(isinstance(s, torch.Tensor) for s in s0_list) and \
                   all(isinstance(s, torch.Tensor) for s in s1_list)

    # 若结构不是张量，尝试按元组解释并验证还原（用于暴露当前实现问题）
    restored_ok = True
    if not structure_ok:
        try:
            for idx, model in enumerate(models):
                # s0_list[idx] 与 s1_list[idx] 都是 (share_a, share_b)，尝试任选一份进行还原检查
                s0_a, s0_b = s0_list[idx]
                restored = s0_a + s0_b
                if not torch.allclose(restored, model.to(restored.dtype), atol=1e-3, rtol=1e-3):
                    restored_ok = False
                    break
        except Exception:
            restored_ok = False

    return {
        'test': 'models_arithmetic_share_basic',
        'num_tensors': num_tensors,
        'tensor_size': tensor_size,
        'device': device,
        'structure_is_tensor_lists': structure_ok,
        'restored_ok_under_tuple_structure': restored_ok
    }


def test_gen_sc_and_verify(shape=(8,), device='cpu'):
    """测试 SC 生成与共享验证流程。"""
    _set_test_globals(use_fp16=False, device=device)
    base = torch.randint(low=1, high=100, size=shape, device=args.gpu)
    A_i, D_i = gen_sc(base)
    A_j, D_j = gen_sc(base)

    Ai_s0, Ai_s1 = arithmetic_share(A_i)
    Di_s0, Di_s1 = arithmetic_share(D_i)
    Aj_s0, Aj_s1 = arithmetic_share(A_j)
    Dj_s0, Dj_s1 = arithmetic_share(D_j)

    ok = verify_sc_shares(Ai_s0, Ai_s1, Di_s0, Di_s1, Aj_s0, Aj_s1, Dj_s0, Dj_s1)
    return {
        'test': 'gen_sc_and_verify',
        'shape': shape,
        'device': device,
        'passed': bool(ok)
    }


def test_parameters_dict_to_vector_functions(device='cpu'):
    """测试参数字典向量化（含 CPU 版本）。"""
    _set_test_globals(use_fp16=False, device=device)

    net = {
        'layer1.weight': torch.randn(4, device=args.gpu),
        'layer1.bias': torch.randn(4, device=args.gpu),
        'bn.num_batches_tracked': torch.tensor(5, device=args.gpu)
    }

def test_gen_delta_integer_simple(device='cpu'):
    """使用整数张量测试 gen_delta，便于手动核对。

    构造三个简洁的本地模型向量（整数），使用 A=0, D=0 的共享，
    预期 delta0[i]+delta1[i] 等于原向量的 L2 范数平方。
    """
    _set_test_globals(use_fp16=False, device=device)

    local_model_vector = [
        torch.tensor([1, 2, 3], dtype=torch.int64, device=args.gpu),
        torch.tensor([0, -1, 2], dtype=torch.int64, device=args.gpu),
        torch.tensor([2, 0, -2], dtype=torch.int64, device=args.gpu),
    ]

    # # 生成模型共享
    # s0, s1 = models_arithmetic_share(local_model_vector)

    # # 构造 A=0, D=0 的共享（简化验证）
    # A0, A1, D0, D1 = [], [], [], []
    # for vec in local_model_vector:
    #     zero = torch.zeros_like(vec, dtype=torch.float64, device=args.gpu)
    #     a0, a1 = arithmetic_share(zero)
    #     d0, d1 = arithmetic_share(zero)
    #     A0.append(a0); A1.append(a1)
    #     D0.append(d0); D1.append(d1)

    s0 = torch.tensor([[1, 2, 3]], dtype=torch.int64, device=args.gpu)
    s1 = torch.tensor([[4, 5, 6]], dtype=torch.int64, device=args.gpu)
    s = s0 + s1
    A0 = torch.tensor([[1,2,3]], dtype=torch.int64, device=args.gpu)
    A1 = torch.tensor([[3,2,1]], dtype=torch.int64, device=args.gpu)
    A = A0 + A1
    D = A * A
    D0 = torch.tensor([[1, 4, 9]], dtype=torch.int64, device=args.gpu)
    D1 = D - D0


    delta0, delta1 = gen_delta(s0, s1, A0, A1, D0, D1)
    print(delta0 + delta1)
    print(s.to(torch.float64) ** 2)


    expected = s.to(torch.float64) ** 2
    got = delta0 + delta1
    # for i, vec in enumerate(local_model_vector):
    #     exp = float((vec.to(torch.float64) ** 2).sum().item())
    #     got_val = float((delta0[i] + delta1[i]).item())
    #     expected.append(exp)
    #     got.append(got_val)

    return {
        'test': 'gen_delta_integer_simple',
        'device': device,
        # 'vectors': [v.tolist() for v in local_model_vector],
        'expected_l2_sq': expected,
        'got_sum_delta': got
        # 'passed': all(abs(e - g) < 1e-4 for e, g in zip(expected, got))
    }

def test_gen_delta_ij_integer_simple(device='cpu'):
    """使用整数张量测试 gen_delta_ij，便于手动核对。

    计算每对 (i,j) 的 (xi-xj)^2 期望值，与 delta_ij0+delta_ij1 比较。
    """
    _set_test_globals(use_fp16=False, device=device)

    # local_model_vector = [
    #     torch.tensor([1, 2, 3], dtype=torch.int64, device=args.gpu),
    #     torch.tensor([0, -1, 2], dtype=torch.int64, device=args.gpu),
    #     torch.tensor([2, 0, -2], dtype=torch.int64, device=args.gpu),
    # ]

    # s0, s1 = models_arithmetic_share(local_model_vector)

    s0 = torch.tensor([[-1,1,1,100,100,100,100],[2,-2,2,100,100,100,100],[3,3,-3,100,100,100,100]], dtype=torch.int64, device=args.gpu)
    s1 = torch.tensor([[1,-1,0,100,100,100,100],[-2,3,-2,100,100,100,100],[-2,-3,3,100,100,100,100]], dtype=torch.int64, device=args.gpu)
    s = s0 + s1

    # A/D 共享在当前实现的 gen_delta_ij 中未直接使用，但保持签名一致
    A0 = torch.tensor([[1,2,3,100,100,100,100],[4,5,6,100,100,100,100],[7,1,10,100,100,100,100]], dtype=torch.int64, device=args.gpu)
    A1 = torch.tensor([[3,2,1,100,100,100,100],[6,5,4,100,100,100,100],[4,1,1,100,100,100,100]], dtype=torch.int64, device=args.gpu)
    A = A0 + A1
    D = A * A
    D0 = torch.tensor([[1, 4, 9,100,100,100,100],[16,15,16,100,100,100,100],[10,21,3,100,100,100,100]], dtype=torch.int64, device=args.gpu)
    D1 = D - D0

    delta_ij0, delta_ij1 = gen_delta_ij(s0, s1, A0, A1, D0, D1)
    print(delta_ij0 + delta_ij1)
    for i in range(len(s)):
        for j in range(i):
            diff = (s[i] - s[j]).to(torch.float64)
            exp = float((diff ** 2).sum().item())
            got_val = float((delta_ij0[i][j] + delta_ij1[i][j]).item())
            
            print(i,j,exp, got_val)
    # expected = []
    # got = []
    # pairs = []
    # for i in range(len(local_model_vector)):
    #     for j in range(i):
    #         diff = (local_model_vector[i] - local_model_vector[j]).to(torch.float64)
    #         exp = float((diff ** 2).sum().item())
    #         got_val = float((delta_ij0[i][j] + delta_ij1[i][j]).item())
    #         expected.append(exp)
    #         got.append(got_val)
    #         pairs.append([i, j])

    return {
        'test': 'gen_delta_ij_integer_simple',
        'device': device,
        # 'pairs': pairs,
        # 'expected_pair_l2_sq': expected,
        # 'got_sum_delta_ij': got,
        # 'passed': all(abs(e - g) < 1e-4 for e, g in zip(expected, got))
    }



def test_div_with_paillier_small_values(device='cpu'):
    """对 div 函数进行小规模烟雾测试。

    注意：div 当前将 Paillier 解密得到的整数与 torch.sqrt 混用，
    在多数环境下会产生类型不匹配问题。本测试主要用于暴露潜在异常。
    """
    _set_test_globals(use_fp16=False, device=device)

    # 初始化必要的 Paillier 参数
    key_length = 256
    pub, priv = paillier.generate_paillier_keypair(n_length=key_length)
    args.paillier_public_key = pub
    args.paillier_private_key = priv
    args.paillier_R = random.randint(10, 100)
    args.paillier_scale_factor = 1000

    # 构造微小共享值
    x0 = torch.tensor(0.1, device=args.gpu)
    x1 = torch.tensor(0.2, device=args.gpu)
    y10 = torch.tensor(0.3, device=args.gpu)
    y11 = torch.tensor(0.4, device=args.gpu)
    y20 = torch.tensor(0.5, device=args.gpu)
    y21 = torch.tensor(0.6, device=args.gpu)

    error = None
    result = None
    try:
        result = div(x0, x1, y10, y11, y20, y21, args)
    except Exception as e:
        error = str(e)

    return {
        'test': 'div_with_paillier_small_values',
        'device': device,
        'succeeded': error is None,
        'error': error,
        'result_type': type(result).__name__ if result is not None else None
    }


def run_all_basic_tests(device='cpu'):
    """运行本文件的基础测试用例并打印结果。"""
    results = []
    # results.append(test_arithmetic_share_basic(shape=(4, 5), use_fp16=False, device=device))
    # results.append(test_models_arithmetic_share_basic(num_tensors=3, tensor_size=10, device=device))
    # results.append(test_gen_sc_and_verify(shape=(8,), device=device))
    # results.append(test_parameters_dict_to_vector_functions(device=device))
    # # 新增：整数输入验证 gen_delta / gen_delta_ij
    # results.append(test_gen_delta_integer_simple(device=device))
    # results.append(test_gen_delta_ij_integer_simple(device=device))
    # results.append(test_div_with_paillier_small_values(device=device))

    # 端到端：构造最小 scram() 输入，验证流程可运行
    try:
        # 构造三个“客户端”参数字典
        _set_test_globals(use_fp16=False, device=device)
        local_model = []
        update_params = []
        for _ in range(10):
            d = {
                'layer.weight': torch.randn(4, device=args.gpu),
                'layer.bias': torch.randn(4, device=args.gpu),
                'bn.num_batches_tracked': torch.tensor(1, device=args.gpu)
            }
            local_model.append(d)
            # update 使用与 local 相同结构
            update_params.append({k: v.clone() for k, v in d.items()})

        # 初始全局模型（与结构一致）
        global_model = {k: torch.zeros_like(v) for k, v in local_model[0].items()}

        # 跑 scram 流程
        _ = scram(local_model, update_params, global_model, args)
        results.append({'test': 'scram_e2e_minimal', 'passed': True})
    except Exception as e:
        results.append({'test': 'scram_e2e_minimal', 'passed': False, 'error': str(e)})

    # print("\n==== defence.py 基础测试结果 ====")
    # for r in results:
    #     if r is None:
    #         print("- [None result]")
    #     else:
    #         print(f"- {r['test']}: {json.dumps(r, ensure_ascii=False)}")
    return results


if __name__ == '__main__':
    # 直接运行本文件时，执行基础测试（默认CPU）
    # import json  # 局部导入，避免顶层污染
    # run_all_basic_tests(device='cpu')
    print("*"*50)
    count = 0
    # 默认在第 0 张 GPU 上运行（避免占用第 1 张）；若仅一张可见 GPU，则使用该 GPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            target_device = torch.device('cuda:0')
        else:
            target_device = torch.device('cuda')
    else:
        target_device = torch.device('cpu')
    print(f"[scram_runner] 使用设备: {target_device} (cuda_count={torch.cuda.device_count()})")
    for epoch, aggregated_global in run_scram_from_default_flame_stream(device=target_device, dry_run=False):
        count += 1
        keys_count = len(aggregated_global) if isinstance(aggregated_global, dict) else 0
        print(f"[scram_runner] 已处理 epoch={epoch}, 聚合参数数目={keys_count}")
    if count == 0:
        print("[scram_runner] 未读取到任何条目，请检查文件路径或轮次范围。")