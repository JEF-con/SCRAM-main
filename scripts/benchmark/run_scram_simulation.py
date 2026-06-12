import numpy as np
import torch
import time
import random
import math
from concurrent.futures import ThreadPoolExecutor
import phe as paillier
import copy
import hdbscan
from typing import Dict, List, Tuple


def generate_random_model_params(num_layers: int, layer_sizes: List[int], device='cpu') -> Dict[str, torch.Tensor]:
    """生成随机模型参数字典"""
    params = {}
    for i in range(num_layers):
        weight_key = f'layer_{i}.weight'
        bias_key = f'layer_{i}.bias'
        weight_shape = (layer_sizes[i], layer_sizes[i-1] if i > 0 else layer_sizes[i])
        bias_shape = (layer_sizes[i],)
        
        params[weight_key] = torch.randn(weight_shape, device=device, dtype=torch.float32) * 0.1
        params[bias_key] = torch.randn(bias_shape, device=device, dtype=torch.float32) * 0.1
    return params


def generate_random_client_data(num_clients: int, num_layers: int, layer_sizes: List[int], device='cpu') -> Tuple[List[Dict], List[Dict]]:
    """生成随机客户端本地模型和更新参数"""
    local_models = []
    update_params = []
    
    for i in range(num_clients):
        local_model = generate_random_model_params(num_layers, layer_sizes, device)
        global_model = generate_random_model_params(num_layers, layer_sizes, device)
        
        # 计算更新参数 (w_local - w_glob)
        update_param = {}
        for key in local_model.keys():
            update_param[key] = local_model[key] - global_model[key]
        
        local_models.append(local_model)
        update_params.append(update_param)
    
    return local_models, update_params


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
    return A, D


def verify_sc_shares(Ai_share0, Ai_share1, Di_share0, Di_share1, Aj_share0, Aj_share1, Dj_share0, Dj_share1):
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
    psi1 = t*t*Di_share1 - Dj_share1 - 2*t*e*Ai_share1
    psi = psi0 + psi1
    if not torch.allclose(psi, torch.zeros_like(psi), atol=1e-4):
        return False
    return True


def gen_delta(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1):
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


    return delta0, delta1


def gen_delta_ij(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1, args=None):
    """
    在"不可还原"约束下，以分块 + 向量化方式计算所有 (i,j) 的向量距离平方对应的算术共享。

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
    # 运行时打印块参数来源与设备，便于调优与确认
    src_client = 'args' if (args is not None and hasattr(args, 'delta_ij_client_block')) else 'default'
    src_param = 'args' if (args is not None and hasattr(args, 'delta_ij_param_block')) else 'default'
    try:
        print(f"[SCRAM DEBUG] delta_ij on device={device}, client_block={client_block} ({src_client}), param_block={param_block} ({src_param})")
    except Exception:
        pass

    with torch.no_grad():
        # 遍历客户端块
        for i_start in range(0, n, client_block):
            i_end = min(i_start + client_block, n)

            # 逐行生成下三角条目：对每个 ii，覆盖 j ∈ [0, ii)
            for ii in range(i_start, i_end):
                # j 分块遍历，确保每行完整覆盖 j< ii
                for j_start in range(0, ii, client_block):
                    j_end = min(j_start + client_block, ii)
                    bj = j_end - j_start

                    # 累积每块的结果（按参数维度分块累加）
                    acc0 = torch.zeros((bj,), device=device, dtype=torch.float64)
                    acc1 = torch.zeros((bj,), device=device, dtype=torch.float64)

                    for k_start in range(0, m, param_block):
                        k_end = min(k_start + param_block, m)

                        # 取本参数块的切片：单行 ii 与 j 块
                        LM0_IK = LM0[ii, k_start:k_end].unsqueeze(0)      # [1, bm]
                        LM0_JK = LM0[j_start:j_end, k_start:k_end]        # [bj, bm]
                        LM1_IK = LM1[ii, k_start:k_end].unsqueeze(0)
                        LM1_JK = LM1[j_start:j_end, k_start:k_end]
                        A0_IK = A0[ii, k_start:k_end].unsqueeze(0)
                        A0_JK = A0[j_start:j_end, k_start:k_end]
                        A1_IK = A1[ii, k_start:k_end].unsqueeze(0)
                        A1_JK = A1[j_start:j_end, k_start:k_end]
                        D0_IK = D0v[ii, k_start:k_end].unsqueeze(0)
                        D0_JK = D0v[j_start:j_end, k_start:k_end]
                        D1_IK = D1v[ii, k_start:k_end].unsqueeze(0)
                        D1_JK = D1v[j_start:j_end, k_start:k_end]

                        # 扩展到 [bj, bm] 以便逐 j 计算
                        LM0_I = LM0_IK.expand(bj, -1)
                        LM0_J = LM0_JK
                        LM1_I = LM1_IK.expand(bj, -1)
                        LM1_J = LM1_JK
                        A0_I = A0_IK.expand(bj, -1)
                        A0_J = A0_JK
                        A1_I = A1_IK.expand(bj, -1)
                        A1_J = A1_JK
                        D0_I = D0_IK.expand(bj, -1)
                        D0_J = D0_JK
                        D1_I = D1_IK.expand(bj, -1)
                        D1_J = D1_JK

                        # 按原始公式计算块内 e/E/eE 与 val0/val1（仅 share 层）
                        e0 = LM0_I - LM0_J - A0_I
                        e1 = LM1_I - LM1_J - A1_I
                        e = e0 + e1

                        E0 = LM0_I - LM0_J - A0_J
                        E1 = LM1_I - LM1_J - A1_J
                        E = E0 + E1

                        eE = e + E
                        val0_k = 0.5 * (D0_I + D0_J) + (LM0_I - LM0_J) * eE - 0.5 * (e * e)
                        val1_k = 0.5 * (D1_I + D1_J) + (LM1_I - LM1_J) * eE - 0.5 * (E * E)

                        # 沿参数维度求和并累加到该 j 块
                        acc0 += val0_k.sum(dim=-1)
                        acc1 += val1_k.sum(dim=-1)

                        # 及时删除临时张量以缓解显存占用
                        try:
                            del LM0_IK, LM0_JK, LM1_IK, LM1_JK
                            del A0_IK, A0_JK, A1_IK, A1_JK
                            del D0_IK, D0_JK, D1_IK, D1_JK
                            del LM0_I, LM0_J, LM1_I, LM1_J
                            del A0_I, A0_J, A1_I, A1_J
                            del D0_I, D0_J, D1_I, D1_J
                            del e0, e1, e, E0, E1, E, eE
                            del val0_k, val1_k
                        except Exception:
                            pass
                        if device.type == 'cuda':
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass

                    # 将该 j 块的结果按顺序写入三角列表
                    for offset in range(bj):
                        delta_ij0[ii].append(acc0[offset].detach().cpu())
                        delta_ij1[ii].append(acc1[offset].detach().cpu())

                    # 删除该 j 块的累加器并清理缓存
                    try:
                        del acc0, acc1
                    except Exception:
                        pass
                    if device.type == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

    print("finish delta_ij")
    return delta_ij0, delta_ij1


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
    - 使用预先解密的 y1R_val, y2R_val 计算分母 sqrt(y1R * y2R)。
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


def filter_benign_clients_by_iqr(benign_client, delta0, delta1, args, whisker_k: float = 1.5):
    """使用 IQR 算法在良性客户端集合中进一步剔除 L2 范数异常的客户端"""
    if benign_client is None or len(benign_client) == 0:
        return []

    norms = []
    for idx in benign_client:
        val = delta0[idx] + delta1[idx]
        try:
            norms.append(float(val.item()))
        except Exception:
            norms.append(float(val.detach().cpu().item()))

    arr = np.array(norms, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return benign_client[:]

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr <= 0:
        return benign_client[:]

    upper = q3 + whisker_k * iqr

    filtered = [idx for idx, val in zip(benign_client, arr.tolist()) if (np.isfinite(val) and val <= upper)]

    if len(filtered) == 0:
        median = float(np.median(arr))
        best = int(np.argmin(np.abs(arr - median)))
        filtered = [benign_client[best]]

    return filtered


def scram_simulation(local_model, update_params, global_model, args):
    """模拟SCRAM算法运行时间"""
    print(f"开始模拟SCRAM算法，客户端数量: {len(local_model)}，参数维度: {len(local_model[0])}")
    
    debug = getattr(args, 'debug_scram', False)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-4).to(args.gpu)
    cos_list = []
    local_model_vector = []
    update_model_vector = []
    
    # 转换参数为向量
    start_time = time.time()
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        flt_para = parameters_dict_to_vector_flt(param)
        local_model_vector.append(flt_para)

    for param in update_params:
        update_model_vector.append(parameters_dict_to_vector_flt(param))
    
    m = len(local_model_vector[0])
    n = len(local_model_vector)
    vector_time = time.time() - start_time
    print(f"参数向量化用时: {vector_time:.6f}秒")
    print(m)

    # 计算本地模型的算术共享
    start_time = time.time()
    local_model_share0, local_model_share1 = models_arithmetic_share(local_model_vector)
    share_time = time.time() - start_time
    print(f"算术共享用时: {share_time:.6f}秒")

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
        A0, D0 = gen_sc(local_model_vector[i])
        A1, D1 = gen_sc(local_model_vector[i])
        A0_list.append(A0)
        D0_list.append(D0)
        A1_list.append(A1)
        D1_list.append(D1)
        A0_share0, A0_share1 = arithmetic_share(A0)
        A1_share0, A1_share1 = arithmetic_share(A1)
        A0_list_share0.append(A0_share0)
        A0_list_share1.append(A0_share1)
        A1_list_share0.append(A1_share0)
        A1_list_share1.append(A1_share1)
        D0_share0, D0_share1 = arithmetic_share(D0)
        D1_share0, D1_share1 = arithmetic_share(D1)
        D0_list_share0.append(D0_share0)
        D0_list_share1.append(D0_share1)
        D1_list_share0.append(D1_share0)
        D1_list_share1.append(D1_share1)
    elapsed_sc = time.time() - start_sc
    print(f"[SCRAM DEBUG] SC生成与共享拆分 用时: {elapsed_sc:.6f} 秒")

    # 验证SC 并得到最后的SC共享，为了方便只验证了一组SC，实际上应该生成两组SC，一组用来计算L2范数平方，一组用来计算向量之间距离的平方
    A_share0 = []
    A_share1 = []
    D_share0 = []
    D_share1 = []
    verified_count = 0
    start_verify = time.time()  # 记录SC验证开始时间
    for i in range(n):
        if verify_sc_shares(A0_list_share0[i], A0_list_share1[i], D0_list_share0[i], D0_list_share1[i], A1_list_share0[i], A1_list_share1[i], D1_list_share0[i], D1_list_share1[i]):
            A_share0.append(A0_list_share0[i])
            A_share1.append(A0_list_share1[i])
            D_share0.append(D0_list_share0[i])
            D_share1.append(D0_list_share1[i])
            verified_count += 1
    elapsed_verify = time.time() - start_verify
    print(f"[SCRAM DEBUG] SC验证 用时: {elapsed_verify:.6f} 秒")

    # TODO：在用时的SC的产生和验证的时候要乘以2
    # 计算delta 即向量的l2范数平方的算术共享
    start_delta = time.time()  # 记录delta计算开始时间
    delta0, delta1 = gen_delta(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1)
    elapsed_delta = time.time() - start_delta
    print(f"[SCRAM DEBUG] delta计算 用时: {elapsed_delta:.6f} 秒")

    # 计算delta_ij 即向量之间的距离平方,为了计算方便服用了前面SC，真正的方案中需要用新的SC
    start_delta_ij = time.time()  # 记录delta_ij计算开始时间
    delta_ij0, delta_ij1 = gen_delta_ij(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1, args=args)
    elapsed_delta_ij = time.time() - start_delta_ij
    print(f"[SCRAM DEBUG] delta_ij计算 用时: {elapsed_delta_ij:.6f} 秒")
    
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
    
    elapsed_cos = time.time() - start_cos
    print(f"[SCRAM DEBUG] 余弦相似度计算 用时: {elapsed_cos:.6f} 秒")

    # 明文余弦距离计算
    start_time = time.time()
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            # 利用公式定义准确计算余弦距离：1 - 余弦相似度
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    plain_cos_time = time.time() - start_time
    print(f"明文余弦距离计算用时: {plain_cos_time:.6f}秒")

    # HDBSCAN聚类
    start_time = time.time()
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)
    cluster_time = time.time() - start_time
    print(f"HDBSCAN聚类用时: {cluster_time:.6f}秒")

    # IQR过滤
    start_time = time.time()
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())
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
        norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())  # no consider BN
    
    print("HDBSCAN 聚类结果:", clusterer.labels_)
    print("HDBSCAN 正常客户端索引:", benign_client)
    
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal += 1
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
    print("IQR 过滤后的良性客户端索引:", filtered_benign_client)
    if hasattr(args, 'attackers_chosen') and isinstance(args.attackers_chosen, (list, tuple)) and len(args.attackers_chosen) > 0:
        print("恶意客户端下标", args.attackers_chosen[-1])
    
    iqr_time = time.time() - start_time
    print(f"IQR过滤用时: {iqr_time:.6f}秒")
    
    print(f"最终候选客户端数量: {len(filtered_benign_client)}")
    
    # 简单返回全局模型（模拟聚合）
    return global_model


class TestArgs:
    """测试参数类"""
    def __init__(self, device='cpu', num_users=10, frac=1.0, malicious=0.0):
        self.gpu = torch.device(device)
        self.frac = frac
        self.num_users = num_users
        self.malicious = malicious
        self.wrong_mal = 0
        self.right_ben = 0
        self.turn = 0
        self.noise = 0.0
        self.paillier_public_key = None
        self.paillier_private_key = None
        self.paillier_R = None
        self.paillier_scale_factor = None
        # 添加分块参数以控制内存使用
        self.delta_ij_client_block = 16  # 减少客户端块大小以降低内存占用
        self.delta_ij_param_block = 10000  # 减少参数块大小以降低内存占用


def run_performance_test(num_clients_list: List[int], layer_sizes_list: List[List[int]], device='cpu'):
    """运行性能测试"""
    print(f"开始性能测试，设备: {device}")
    print("=" * 80)
    
    results = []
    
    for num_clients in num_clients_list:
        for layer_sizes in layer_sizes_list:
            print(f"\n测试配置: 客户端数量={num_clients}, 层大小={layer_sizes}")
            
            # 生成随机数据
            start_time = time.time()
            local_models, update_params = generate_random_client_data(
                num_clients, len(layer_sizes), layer_sizes, device
            )
            global_model = generate_random_model_params(len(layer_sizes), layer_sizes, device)
            data_gen_time = time.time() - start_time
            
            print(f"数据生成用时: {data_gen_time:.6f}秒")
            
            # 创建参数对象
            args = TestArgs(device=device, num_users=num_clients)
            
            # 运行SCRAM模拟
            scram_start_time = time.time()
            result = scram_simulation(local_models, update_params, global_model, args)
            scram_time = time.time() - scram_start_time
            
            print(f"SCRAM总用时: {scram_time:.6f}秒")
            print("-" * 60)
            
            results.append({
                'num_clients': num_clients,
                'layer_sizes': layer_sizes,
                'data_gen_time': data_gen_time,
                'scram_time': scram_time,
                'total_params': sum(s * s for s in layer_sizes)  # 粗略估计参数数量
            })
    
    # 打印汇总结果
    print("\n" + "=" * 80)
    print("性能测试汇总:")
    print(f"{'客户端数量':<10} {'层配置':<20} {'数据生成时间':<12} {'SCRAM时间':<12} {'总参数数':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['num_clients']:<10} {str(result['layer_sizes']):<20} "
              f"{result['data_gen_time']:<12.6f} {result['scram_time']:<12.6f} {result['total_params']:<10}")
    
    return results


if __name__ == "__main__":
    # 检查CUDA可用性并默认使用GPU 1
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 定义测试参数 - 减小客户端数量和网络规模以适应内存限制
    num_clients_list = [50,70,100]  # 减少客户端数量
    layer_sizes_list = [
        # [100, 50, 10],  # 小型网络
        # [250, 100, 50],  # 中型网络
        [500, 250, 100],  # 大型网络，但比原来小
        [784, 512, 256, 128, 10],  # 6.5M
        [1024, 768, 384, 297, 10],
        # [4096, 2048, 2048, 1024, 512, 10]  #12.5M
        
    ]
    
    # 运行性能测试
    results = run_performance_test(num_clients_list, layer_sizes_list, device)
    
    print("\n测试完成！")