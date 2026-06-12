# 文件清单（SCRAM 论文归档版）

本清单以“论文方案 = SCRAM”为主线，标注每类文件在复现实验中的角色与建议保留级别。

## 1. 训练主线（保留在仓库根目录，最小改动）

- [main_fed.py](../main_fed.py)：联邦训练入口。论文使用 SCRAM 时运行 `--defence scram` 或 `--defence scram_without_privacy`。
- [models/](../models)
  - [Nets.py](../models/Nets.py)：模型结构定义。
  - [Update.py](../models/Update.py)：良性本地训练 + 多种攻击训练实现（供对比实验使用）。
  - [MaliciousUpdate.py](../models/MaliciousUpdate.py)：恶意客户端训练封装。
  - [test.py](../models/test.py)：评测（MA/BA/ASR 等）。
- [utils/](../utils)
  - [scram.py](../utils/scram.py)：SCRAM 核心实现（算术共享、SC 生成/验证、delta/delta_ij、同态/安全余弦与聚类等）。
  - [scram_memory_optimized.py](../utils/scram_memory_optimized.py)：SCRAM 内存优化版本（可选）。
  - [options.py](../utils/options.py)：命令行参数。

## 2. 论文实验脚本（统一放入 scripts/）

### 2.1 Benchmark（性能分段计时）

- [scripts/benchmark/benchmark_scram.py](../scripts/benchmark/benchmark_scram.py)：SCRAM 各阶段分段计时基准。
- [scripts/benchmark/run_benchmark_experiments.py](../scripts/benchmark/run_benchmark_experiments.py)：批量调用 benchmark，默认输出到 `results/benchmark/`。
- [scripts/benchmark/run_scram_simulation.py](../scripts/benchmark/run_scram_simulation.py)：SCRAM 流程仿真/压力测试（不依赖主训练）。

### 2.2 绘图（性能图）

- [scripts/plotting/plot_scram_clients_lines.py](../scripts/plotting/plot_scram_clients_lines.py)：客户端数量扩展曲线；读 `results/benchmark/SCRAM_clients_timing_summary.csv`，图输出到 `results/benchmark/figures/`。
- [scripts/plotting/plot_param_scale_lines.py](../scripts/plotting/plot_param_scale_lines.py)：参数规模扩展曲线；读 `results/benchmark/SCRAM_param_scale_table.csv`，图输出到 `results/benchmark/figures/`。
- [scripts/plotting/generate_benchmark_plots.py](../scripts/plotting/generate_benchmark_plots.py)：用硬编码数据快速出图（适合论文临时出图）；图输出到 `results/benchmark/figures/`。

### 2.3 日志分析（检测指标/MA/BA 汇总）

- [scripts/analysis/analyze_defense_results.py](../scripts/analysis/analyze_defense_results.py)：解析 `output_malicious_*.log`，输出 `results/analysis/defense_analysis_raw.csv` 与 `defense_analysis_summary.csv` 并生成图。
- [scripts/analysis/plot_defense_metrics.py](../scripts/analysis/plot_defense_metrics.py)：从 summary CSV 出图，默认输入/输出都在 `results/analysis/`。
- [scripts/analysis/debug_defense_stats.py](../scripts/analysis/debug_defense_stats.py)：调试日志解析（支持 `--log-file` 指定文件）。

## 3. 结果目录（统一放入 results/）

- [results/benchmark/](../results/benchmark)
  - 基准计时 CSV、日志（如 `benchmark_summary_*.csv`、`benchmark_results.log`）与 SCRAM 性能表（`SCRAM_*_summary.csv`）。
  - 生成图片统一在 `results/benchmark/figures/`。
- `results/analysis/`（运行分析脚本后生成）：防御检测指标、MA/BA 汇总与绘图输出。

## 4. 归档/历史代码（archive/）

- [archive/legacy/Calculate/](../archive/legacy/Calculate)：历史原型与对照实现（含旧版 scram/newFlame 拆分实现），不作为论文主线依赖。
- [archive/dev_misc/](../archive/dev_misc)：临时测试/占位脚本。
