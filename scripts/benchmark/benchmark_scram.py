
import time
import torch
import numpy as np
import argparse
import sys
import os
import random
import math
import hdbscan
import phe as paillier
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

try:
    from utils.scram import (
        models_arithmetic_share,
        gen_sc,
        arithmetic_share,
        verify_sc_shares,
        gen_delta,
        gen_delta_ij,
        precompute_y_ciphertexts_for_clients,
        precompute_delta_ciphertexts_for_clients,
        compute_xR_from_cipher_parts
    )
except ImportError:
    print("Error: Could not import functions from utils.scram.")
    print(f"Repo root assumed at: {repo_root}")
    sys.exit(1)

class BenchmarkArgs:
    def __init__(self, device_str, client_block, param_block):
        self.gpu = torch.device(device_str)
        self.device = self.gpu
        self.paillier_public_key = None
        self.paillier_private_key = None
        self.paillier_R = None
        self.paillier_scale_factor = None
        self.delta_ij_client_block = client_block
        self.delta_ij_param_block = param_block
        # Dummy values required by some functions if they access args directly
        self.frac = 1.0
        self.num_users = 0
        self.malicious = 0.0
        self.wrong_mal = 0
        self.right_ben = 0
        self.turn = 0
        self.debug_scram = False

def generate_synthetic_data(num_clients, model_size, device):
    # Store data on CPU to avoid OOM
    print(f"Generating data for {num_clients} clients, model size {model_size} on CPU (Storage Strategy)...")
    print("  Note: Data is kept on CPU to prevent GPU Out-Of-Memory. Computations will use GPU.")
    local_model_vectors = []
    for i in range(num_clients):
        # Generate random vector simulating a flattened model
        vec = torch.randn(model_size, device='cpu', dtype=torch.float32)
        local_model_vectors.append(vec)
    return local_model_vectors

def run_benchmark(num_clients, model_size, device_str='cpu', run_crypto=True, client_block=None, param_block=None):
    # Ensure we use GPU 0 if cuda is specified
    if device_str == 'cuda':
         device_str = 'cuda:1'
            
    device = torch.device(device_str)
    
    # Configure block sizes (heuristic based on device if not provided)
    if client_block is None:
        client_block = 16
    if param_block is None:
        param_block = 1000000 if device.type == 'cuda' else 100000
    
    if device.type == 'cuda' and param_block < 100000:
        print(f"Warning: param_block ({param_block}) is small for GPU. Performance may be poor due to loop overhead.")
        print("Suggestion: Use --param-block 500000 or larger.")
    
    print(f"Configuration: Client Block={client_block}, Param Block={param_block}")
    
    args = BenchmarkArgs(device_str, client_block, param_block)
    args.num_users = num_clients
    
    times = {}
    print(f"\n--- Starting Benchmark: Clients={num_clients}, ModelSize={model_size}, Device={device_str} ---")

    # ---------------------------------------------------------
    # 0. Data Generation
    # ---------------------------------------------------------
    t0 = time.time()
    # Generate on CPU to save GPU memory
    local_model_vectors = generate_synthetic_data(num_clients, model_size, device)
    if device.type == 'cuda': torch.cuda.synchronize()
    times['0. Data Generation'] = time.time() - t0
    print(f"Data Generation: {times['0. Data Generation']:.4f}s")

    # ---------------------------------------------------------
    # 1. Models Arithmetic Share
    # ---------------------------------------------------------
    t0 = time.time()
    # Perform arithmetic share on CPU (it's fast enough and saves GPU mem)
    # If needed, we could chunk this to GPU, but for now CPU is safer for storage
    local_model_share0, local_model_share1 = models_arithmetic_share(local_model_vectors)
    if device.type == 'cuda': torch.cuda.synchronize()
    times['1. Arithmetic Share'] = time.time() - t0
    print(f"Arithmetic Share: {times['1. Arithmetic Share']:.4f}s")

    # ---------------------------------------------------------
    # 2. Secure Component (SC) Generation & Verification
    # ---------------------------------------------------------
    t0 = time.time()
    
    A0_list_share0 = []
    A0_list_share1 = []
    D0_list_share0 = []
    D0_list_share1 = []
    A1_list_share0 = []
    A1_list_share1 = []
    D1_list_share0 = []
    D1_list_share1 = []
    
    for i in range(num_clients):
        # Move single client vector to GPU for SC generation
        vec_gpu = local_model_vectors[i].to(device)
        
        # Generate two sets of SCs (as per scram.py logic)
        A0, D0 = gen_sc(vec_gpu)
        A1, D1 = gen_sc(vec_gpu)
        
        # Share them
        s0, s1 = arithmetic_share(A0)
        A0_list_share0.append(s0.cpu()); A0_list_share1.append(s1.cpu())
        
        s0, s1 = arithmetic_share(A1)
        A1_list_share0.append(s0.cpu()); A1_list_share1.append(s1.cpu())
        
        s0, s1 = arithmetic_share(D0)
        D0_list_share0.append(s0.cpu()); D0_list_share1.append(s1.cpu())
        
        s0, s1 = arithmetic_share(D1)
        D1_list_share0.append(s0.cpu()); D1_list_share1.append(s1.cpu())
        
        # Cleanup GPU memory
        del vec_gpu, A0, D0, A1, D1, s0, s1
        if device.type == 'cuda': torch.cuda.empty_cache()

    # Verification Step (CPU is fine, or move to GPU in chunks if needed)
    A_share0 = []
    A_share1 = []
    D_share0 = []
    D_share1 = []
    
    for i in range(num_clients):
        # Move shares to GPU for verification computation
        verify_args = [
            A0_list_share0[i], A0_list_share1[i], D0_list_share0[i], D0_list_share1[i],
            A1_list_share0[i], A1_list_share1[i], D1_list_share0[i], D1_list_share1[i]
        ]
        # Move to device if needed (verify_sc_shares handles device based on input)
        # However, verify_sc_shares might expect inputs on same device.
        # Let's move them to device for verification, then discard
        verify_args_dev = [t.to(device) for t in verify_args]
        
        if verify_sc_shares(*verify_args_dev):
            A_share0.append(A0_list_share0[i]) # Keep on CPU
            A_share1.append(A0_list_share1[i])
            D_share0.append(D0_list_share0[i])
            D_share1.append(D0_list_share1[i])
        else:
            print(f"Warning: SC Verification failed for client {i}")
        
        del verify_args_dev
        if device.type == 'cuda': torch.cuda.empty_cache()

    if device.type == 'cuda': torch.cuda.synchronize()
    times['2. SC Gen & Verify'] = time.time() - t0
    print(f"SC Gen & Verify: {times['2. SC Gen & Verify']:.4f}s")

    # ---------------------------------------------------------
    # 3. Delta Calculation (L2 Norm Squared)
    # ---------------------------------------------------------
    t0 = time.time()
    # gen_delta computes row-wise. We can do it on GPU row by row.
    # But gen_delta in utils/scram.py loops over all clients.
    # To optimize memory, we should probably run gen_delta with CPU inputs and let it run on CPU?
    # Or modify gen_delta. For now, let's run it on CPU (it's O(N), not heavy).
    # Actually, gen_delta is lightweight O(N*M). CPU is fine.
    delta0, delta1 = gen_delta(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1)
    if device.type == 'cuda': torch.cuda.synchronize()
    times['3. Delta Calculation'] = time.time() - t0
    print(f"Delta Calculation: {times['3. Delta Calculation']:.4f}s")

    # ---------------------------------------------------------
    # 4. Delta_ij Calculation (Pairwise Distance Squared)
    # ---------------------------------------------------------
    t0 = time.time()
    delta_ij0, delta_ij1 = gen_delta_ij(local_model_share0, local_model_share1, A_share0, A_share1, D_share0, D_share1, args=args)
    if device.type == 'cuda': torch.cuda.synchronize()
    times['4. Delta_ij Calculation'] = time.time() - t0
    print(f"Delta_ij Calculation: {times['4. Delta_ij Calculation']:.4f}s")

    if not run_crypto:
        print("Skipping Crypto stages as requested.")
        return times

    # ---------------------------------------------------------
    # 5. Key Generation (Paillier)
    # ---------------------------------------------------------
    t0 = time.time()
    key_length = 128 # Reduced key length for benchmark speed, production uses 2048 usually or 1024
    print(f"Generating Paillier keys (length={key_length})...")
    args.paillier_public_key, args.paillier_private_key = paillier.generate_paillier_keypair(n_length=key_length)
    args.paillier_R = random.randint(10, 100)
    args.paillier_scale_factor = 1000
    times['5. Key Generation'] = time.time() - t0
    print(f"Key Generation: {times['5. Key Generation']:.4f}s")

    # ---------------------------------------------------------
    # 6. Secure Cosine Similarity Calculation
    # ---------------------------------------------------------
    t0 = time.time()
    
    # Precompute Y ciphertexts
    gy10_list, gy11_list = precompute_y_ciphertexts_for_clients(delta0, delta1, args)
    
    # Precompute Delta ciphertexts
    g_d0_list, g_d1_list = precompute_delta_ciphertexts_for_clients(delta0, delta1, args)
    
    # Decrypt Y sums
    y_sum_plain = []
    private_key = args.paillier_private_key
    for k in range(num_clients):
        y_sum_plain.append(float(private_key.decrypt(gy10_list[k] + gy11_list[k])))
        
    # Matrix calculation
    cos_distance = [[0.0 for _ in range(num_clients)] for _ in range(num_clients)]
    
    for i in range(num_clients):
        j_max = min(i, len(delta_ij0[i]) if i < len(delta_ij0) else 0, len(delta_ij1[i]) if i < len(delta_ij1) else 0)
        for j in range(j_max):
             # For benchmarking, we execute the core logic
             # Note: logic copied from scram.py
             
             # x0_val/x1_val are local computations
             # x0_val = 0.5 * (float(delta0[i].item()) + float(delta0[j].item()) - float(delta_ij0[i][j].item())) # Simplified access
             
             # R_ij = random.randint(10, 100)
             # y1R_val = R_ij * y_sum_plain[i]
             # y2R_val = R_ij * y_sum_plain[j]
             
             # The heavy part:
             R_ij = random.randint(10, 100)
             y1R_val = R_ij * y_sum_plain[i]
             y2R_val = R_ij * y_sum_plain[j]
             
             xR_val = compute_xR_from_cipher_parts(i, j, delta_ij0, delta_ij1, g_d0_list, g_d1_list, R_ij, args)
             
             cos_distance[j][i] = cos_distance[i][j] = 1.0 - (
                xR_val / math.sqrt(max(y1R_val * y2R_val, 1e-12))
             )
        
        for j in range(j_max, i):
            cos_distance[j][i] = cos_distance[i][j] = 0.0
        cos_distance[i][i] = 0.0

    times['6. Secure Cosine Calc'] = time.time() - t0
    print(f"Secure Cosine Calc: {times['6. Secure Cosine Calc']:.4f}s")

    # ---------------------------------------------------------
    # 7. Clustering (HDBSCAN)
    # ---------------------------------------------------------
    t0 = time.time()
    # HDBSCAN expects distance matrix or features. scram.py passes `cos_list` which is the cosine distance matrix (1 - similarity).
    # Here `cos_distance` is computed as 1 - similarity (based on formula in scram.py).
    
    # Convert to list of lists if not already
    # Ensure cos_distance is a numpy array of float64 for HDBSCAN
    cos_distance_np = np.array(cos_distance, dtype=np.float64)
    
    # Check for NaN or Inf
    if np.isnan(cos_distance_np).any() or np.isinf(cos_distance_np).any():
        print("Warning: NaN or Inf values found in distance matrix. Replacing with 1.0 (max distance).")
        cos_distance_np = np.nan_to_num(cos_distance_np, nan=1.0, posinf=1.0, neginf=1.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(num_clients//2 + 1, 2),
        min_samples=1,
        allow_single_cluster=True,
        metric='precomputed'
    ).fit(cos_distance_np)
    
    times['7. Clustering'] = time.time() - t0
    print(f"Clustering: {times['7. Clustering']:.4f}s")

    return times

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SCRAM stages")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--size", type=int, default=1000, help="Model parameter size (number of floats)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--no-crypto", action="store_true", help="Skip Paillier crypto stages (which are slow)")
    parser.add_argument("--client-block", type=int, default=None, help="Block size for clients (for GPU optimization)")
    parser.add_argument("--param-block", type=int, default=None, help="Block size for parameters (for GPU optimization)")
    
    args = parser.parse_args()
    
    run_benchmark(args.clients, args.size, args.device, not args.no_crypto, args.client_block, args.param_block)
