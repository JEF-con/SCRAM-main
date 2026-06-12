#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机批量调用 newFlame 的独立脚本与函数（不依赖 newflame_simulator）。

功能：
- 随机生成 local_model / update_params / global_model（不通过训练/ML流程），
  结构与 utils.defense.newFlame 期望一致；
- 允许指定用户数量、参数规模、调用次数等；
- 复用 newFlame 内部的计时与 CSV/JSON 保存逻辑（timing_analysis/）。

用法（命令行）：
- python -m utils.newflame_random_runner --num_clients 30 --param_size 5000 --num_calls 2 \
    --malicious_frac 0.0 --noise 0.0 --session_id test_session_01

备注：
- 若环境支持 CUDA，将在 GPU 上生成张量；否则退回 CPU。
"""

import os
import sys
import argparse
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from utils.defense import newFlame


def _make_random_param_dict(param_size: int, device: torch.device, dtype=torch.float64):
    """生成与 newFlame 兼容的参数字典：仅包含一个权重张量。

    Args:
        param_size: 权重张量的总元素个数
        device: 目标设备（cuda/cpu）
        dtype: 张量数据类型

    Returns:
        dict: {"layer.weight": Tensor[param_size]}
    """
    return {
        "layer.weight": torch.randn(param_size, device=device, dtype=dtype)
    }


def _make_args(num_clients: int,
               malicious_frac: float,
               noise: float,
               session_id: str | None):
    """构造与 newFlame 兼容的 args 对象。

    关键字段：
    - gpu: 设备对象
    - frac / num_users / malicious / mul / noise
    - training_session_id：用于 newFlame 的计时文件命名
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = SimpleNamespace()
    args.gpu = device
    args.frac = 1.0
    args.num_users = int(num_clients)
    args.malicious = float(malicious_frac)
    args.mul = 2.0
    args.noise = float(noise)
    # 让 newFlame 在同一 args 上复用密钥与计数器
    args.training_session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return args


def run_newflame_random(num_clients: int,
                        param_size: int,
                        num_calls: int,
                        malicious_frac: float = 0.0,
                        noise: float = 0.0,
                        session_id: str | None = None,
                        seed: int | None = None):
    """随机生成参数并多次调用 newFlame。

    Args:
        num_clients: 用户数量（local_model / update_params 的长度）
        param_size: 每个参数字典中权重张量的元素数量
        num_calls: 调用 newFlame 的次数
        malicious_frac: 恶意客户端比例（0.0-1.0）
        noise: 聚合后加噪标准差缩放系数（参考 newFlame）
        session_id: 指定会话ID（用于输出文件命名）；默认自动生成
        seed: 随机种子（固定重现实验）

    Returns:
        dict: 运行信息，包括输出文件路径与会话ID
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 构造 args（复用同一对象确保 Paillier 密钥只生成一次）
    args = _make_args(num_clients=num_clients,
                      malicious_frac=malicious_frac,
                      noise=noise,
                      session_id=session_id)

    device = args.gpu
    dtype = torch.float64

    # 初始化 global_model（全零），确保键与 update_params 对齐
    global_model = _make_random_param_dict(param_size, device=device, dtype=dtype)
    for k in list(global_model.keys()):
        global_model[k] = torch.zeros_like(global_model[k], device=device, dtype=dtype)

    # 预创建一次固定的 local_model / update_params（每轮调用可重新生成或复用）
    def make_batch():
        local_model = []
        update_params = []
        for _ in range(num_clients):
            local_model.append(_make_random_param_dict(param_size, device=device, dtype=dtype))
            update_params.append(_make_random_param_dict(param_size, device=device, dtype=dtype))
        return local_model, update_params

    # 多次调用 newFlame
    t0 = time.time()
    for call_idx in range(num_calls):
        # 每次调用都生成新的随机输入
        local_model, update_params = make_batch()
        newFlame(local_model=local_model,
                 update_params=update_params,
                 global_model=global_model,
                 args=args)
    total_elapsed = time.time() - t0

    timing_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "timing_analysis")
    json_path = os.path.join(timing_dir, f"newflame_timing_{args.training_session_id}.json")
    csv_path = os.path.join(timing_dir, f"newflame_timing_{args.training_session_id}.csv")

    return {
        "session_id": args.training_session_id,
        "json": json_path,
        "csv": csv_path,
        "calls": num_calls,
        "num_clients": num_clients,
        "param_size": param_size,
        "elapsed": total_elapsed,
        "device": str(device),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="随机批量调用 newFlame 并保存计时统计（CSV/JSON）")
    p.add_argument("--num_clients", type=int, required=True, help="用户数量")
    p.add_argument("--param_size", type=int, required=True, help="每个权重张量的元素数量")
    p.add_argument("--num_calls", type=int, required=True, help="调用次数")
    p.add_argument("--malicious_frac", type=float, default=0.0, help="恶意客户端比例 [0,1]")
    p.add_argument("--noise", type=float, default=0.0, help="聚合加噪幅度系数")
    p.add_argument("--session_id", type=str, default=None, help="训练会话ID（输出文件名前缀）")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    return p.parse_args()


if __name__ == "__main__":
    cli = _parse_args()
    info = run_newflame_random(
        num_clients=cli.num_clients,
        param_size=cli.param_size,
        num_calls=cli.num_calls,
        malicious_frac=cli.malicious_frac,
        noise=cli.noise,
        session_id=cli.session_id,
        seed=cli.seed,
    )
    print("\n=== newFlame 随机批量调用完成 ===")
    print(f"会话ID: {info['session_id']}")
    print(f"调用次数: {info['calls']}")
    print(f"用户数量: {info['num_clients']}")
    print(f"参数规模: {info['param_size']}")
    print(f"设备: {info['device']}")
    print(f"总耗时: {info['elapsed']:.6f} 秒")
    print(f"JSON: {info['json']}")
    print(f"CSV:  {info['csv']}")