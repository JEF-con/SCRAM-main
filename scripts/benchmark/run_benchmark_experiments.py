import subprocess
import re
import csv
import sys
import os
from pathlib import Path

def run_experiment(clients, size, device, output_file=None):
    repo_root = Path(__file__).resolve().parents[2]
    bench_script = Path(__file__).resolve().parent / "benchmark_scram.py"

    # Prepare environment with CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    target_device = device
    
    # If a specific cuda device is requested (e.g., "cuda:1"), 
    # use CUDA_VISIBLE_DEVICES to isolate it and tell PyTorch to use "cuda:0" or "cuda"
    if "cuda" in device and ":" in device:
        try:
            gpu_id = device.split(":")[-1]
            # Verify gpu_id is an integer
            int(gpu_id)
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            target_device = "cuda:0" # Inside the isolated environment, it's always 0
        except ValueError:
            pass # Fallback to original device string if parsing fails

    cmd = [
        sys.executable, str(bench_script),
        "--clients", str(clients),
        "--size", str(size),
        "--device", target_device,
        "--param-block", "500000", # Use large block for GPU efficiency
        "--client-block", "16"     # Use block 16 for bi x bj vectorization (approx 10GB VRAM usage)
    ]
    
    print(f"Running: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'Not Set')})")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, cwd=str(repo_root))
        output = result.stdout
        if output_file:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- Experiment: Clients={clients}, Size={size} ---\n")
                f.write(output)
        
        # Parse output for timings
        timings = {}
        patterns = {
            "Data Gen": r"Data Generation: ([\d\.]+)s",
            "Arith Share": r"Arithmetic Share: ([\d\.]+)s",
            "SC Gen/Ver": r"SC Gen & Verify: ([\d\.]+)s",
            "Delta": r"Delta Calculation: ([\d\.]+)s",
            "Delta_ij": r"Delta_ij Calculation: ([\d\.]+)s",
            "Key Gen": r"Key Generation: ([\d\.]+)s",
            "Sec Cos": r"Secure Cosine Calc: ([\d\.]+)s",
            "Clustering": r"Clustering: ([\d\.]+)s"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                timings[key] = float(match.group(1))
            else:
                timings[key] = 0.0
                
        return timings
        
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(e.stderr)
        return None

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run SCRAM benchmarks with varying sizes and clients.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., cuda:0, cpu)")
    args = parser.parse_args()

    device = args.device
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_log = out_dir / "benchmark_results.log"
    
    # Clear log file
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("Benchmark Logs\n")

    results_size = []
    results_clients = []

    # Experiment 1: Varying Model Size
    # print("\n=== Experiment 1: Varying Model Size (Clients=100) ===")
    # sizes = [300000, 600000, 900000, 1200000, 1500000, 1800000]
    # fixed_clients = 100
    # 
    # for size in sizes:
    #     timings = run_experiment(fixed_clients, size, device, output_log)
    #     if timings:
    #         timings["Size"] = size
    #         results_size.append(timings)

    # Experiment 2: Varying Client Count
    # print("\n=== Experiment 2: Varying Client Count (Size=1500000) ===")
    # clients_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
    # fixed_size = 1500000
    # 
    # for clients in clients_list:
    #     timings = run_experiment(clients, fixed_size, device, output_log)
    #     if timings:
    #         timings["Clients"] = clients
    #         results_clients.append(timings)

    # Experiment 3: Varying Model Size (Clients=50)
    print("\n=== Experiment 3: Varying Model Size (Clients=50) ===")
    sizes_exp3 = [100000, 400000, 1200000]
    fixed_clients_exp3 = 50
    results_exp3 = []

    for size in sizes_exp3:
        timings = run_experiment(fixed_clients_exp3, size, device, output_log)
        if timings:
            timings["Size"] = size
            results_exp3.append(timings)

    # Print Summary Tables
    # print("\n\n=== Summary: Varying Model Size (Clients=100) ===")
    # headers = ["Size", "Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
    # print(f"{' | '.join(headers)}")
    # for res in results_size:
    #     row = [str(res.get(h, 0)) for h in headers]
    #     print(f"{' | '.join(row)}")

    # print("\n\n=== Summary: Varying Client Count (Size=1500k) ===")
    # headers = ["Clients", "Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
    # print(f"{' | '.join(headers)}")
    # for res in results_clients:
    #     row = [str(res.get(h, 0)) for h in headers]
    #     print(f"{' | '.join(row)}")
    
    print("\n\n=== Summary: Varying Model Size (Clients=50) ===")
    headers = ["Size", "Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
    print(f"{' | '.join(headers)}")
    for res in results_exp3:
        row = [str(res.get(h, 0)) for h in headers]
        print(f"{' | '.join(row)}")

    # Save to CSV
    # csv_headers_size = ["Size", "Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
    # with open("benchmark_summary_size.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=csv_headers_size)
    #     writer.writeheader()
    #     writer.writerows(results_size)
        
    # csv_headers_clients = ["Clients", "Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
    # with open("benchmark_summary_clients.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=csv_headers_clients)
    #     writer.writeheader()
    #     writer.writerows(results_clients)
    
    csv_headers_exp3 = ["Size", "Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
    with open(out_dir / "benchmark_summary_exp3.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers_exp3)
        writer.writeheader()
        writer.writerows(results_exp3)

if __name__ == "__main__":
    main()
