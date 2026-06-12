# 复现实验命令（SCRAM）

本文档给出按当前归档结构运行的常用命令示例。默认在仓库根目录执行。

## 1. 训练主实验（SCRAM）

```bash
python main_fed.py ^
  --dataset fashion_mnist ^
  --model cnn ^
  --attack dba ^
  --defence scram ^
  --epochs 200 ^
  --num_users 100 ^
  --frac 1 ^
  --malicious 0.1 ^
  --poison_frac 1.0 ^
  --local_ep 2 ^
  --local_bs 64 ^
  --attack_begin 0 ^
  --attack_label 5 ^
  --attack_goal -1 ^
  --iid 1 ^
  --gpu 0 ^
  --save save/my_experiments
```

无隐私版本（用于对比）：

```bash
python main_fed.py --defence scram_without_privacy ...
```

## 2. 性能分段计时（Benchmark）

单次运行：

```bash
python scripts/benchmark/benchmark_scram.py --clients 50 --size 1200000 --device cpu
```

批量运行并输出汇总 CSV 到 `results/benchmark/`：

```bash
python scripts/benchmark/run_benchmark_experiments.py --device cpu
```

## 3. 性能绘图

确保 `results/benchmark/` 下存在：
- `SCRAM_clients_timing_summary.csv`
- `SCRAM_param_scale_table.csv`

然后运行：

```bash
python scripts/plotting/plot_scram_clients_lines.py
python scripts/plotting/plot_param_scale_lines.py
```

图片输出到：
- `results/benchmark/figures/`

## 4. 日志解析与防御指标绘图

解析 `output_malicious_*.log` 并输出到 `results/analysis/`：

```bash
python scripts/analysis/analyze_defense_results.py --log-dir . --pattern "output_malicious_*.log"
```

从 summary CSV 出图（默认读写 `results/analysis/`）：

```bash
python scripts/analysis/plot_defense_metrics.py
```

