# SCRAM 论文实验代码归档报告

本报告记录本次“以 SCRAM 为论文主线”的代码归档操作：目录结构、迁移清单、脚本兼容性修改点、以及复现入口。

## 1. 归档目标与原则

- 目标：让仓库具备“论文复现友好”的结构，读者能快速定位：
  - 训练主实验入口（SCRAM）
  - SCRAM 性能 benchmark 与出图脚本
  - 日志解析与指标汇总脚本
- 原则：采用最小改动方式，避免大规模重构 import；保留原训练入口与 `models/`、`utils/` 包结构不变。

## 2. 新增目录

- `scripts/benchmark/`：基准与仿真脚本
- `scripts/plotting/`：绘图脚本
- `scripts/analysis/`：日志解析与指标绘图
- `docs/`：归档说明文档
- `results/benchmark/`：benchmark 数据与汇总
- `archive/legacy/`：历史原型与对照实现（不作为论文主线依赖）
- `archive/dev_misc/`：临时脚本与占位脚本

## 3. 文件迁移清单（已执行）

### 3.1 脚本迁移

- 根目录 -> `scripts/benchmark/`
  - `benchmark_scram.py` -> `scripts/benchmark/benchmark_scram.py`
  - `run_benchmark_experiments.py` -> `scripts/benchmark/run_benchmark_experiments.py`
  - `run_scram_simulation.py` -> `scripts/benchmark/run_scram_simulation.py`
- 根目录 -> `scripts/analysis/`
  - `analyze_defense_results.py` -> `scripts/analysis/analyze_defense_results.py`
  - `plot_defense_metrics.py` -> `scripts/analysis/plot_defense_metrics.py`
  - `debug_defense_stats.py` -> `scripts/analysis/debug_defense_stats.py`
- `timing_analysis/` -> `scripts/plotting/`
  - `plot_scram_clients_lines.py` -> `scripts/plotting/plot_scram_clients_lines.py`
  - `plot_param_scale_lines.py` -> `scripts/plotting/plot_param_scale_lines.py`
  - `plot_param_scale_table.py` -> `scripts/plotting/plot_param_scale_table.py`
  - `aggregate_rnd_big_cpu_param50.py` -> `scripts/plotting/aggregate_rnd_big_cpu_param50.py`
- 根目录 -> `scripts/plotting/`
  - `generate_benchmark_plots.py` -> `scripts/plotting/generate_benchmark_plots.py`

### 3.2 结果文件迁移（根目录清理）

- 根目录 -> `results/benchmark/`
  - `benchmark_results.log`
  - `benchmark_summary_clients.csv`
  - `benchmark_summary_exp3.csv`
  - `benchmark_summary_size.csv`
  - `defense_analysis_raw.csv`
  - `defense_analysis_summary.csv`
  - `defense_metrics_plot.png`
  - `defense_metrics_plot_from_csv.png`
- `timing_analysis/` -> `results/benchmark/`
  - `SCRAM_clients_timing_summary.csv`
  - `SCRAM_param_scale_table.csv`

### 3.3 历史代码/临时脚本归档

- `Calculate/` -> `archive/legacy/Calculate/`
- 根目录/`timing_analysis/` -> `archive/dev_misc/`
  - `test.py`
  - `anotherTest.py`
  - `plot_.py`

### 3.4 命令文件归档

- `command_*.txt` -> `docs/commands/`

## 4. 脚本兼容性修订（已执行）

为保证脚本在新目录下仍可运行，做了以下关键改动：

- `scripts/benchmark/benchmark_scram.py`
  - 从“依赖当前工作目录”改为“自动定位仓库根目录并加入 sys.path”。
- `scripts/benchmark/run_benchmark_experiments.py`
  - 以脚本自身路径定位 `benchmark_scram.py`，并以仓库根目录作为 `cwd` 启动子进程。
  - 输出默认写入 `results/benchmark/`。
- `scripts/plotting/plot_scram_clients_lines.py` 与 `scripts/plotting/plot_param_scale_lines.py`
  - 输入 CSV 改为读取 `results/benchmark/`。
  - 图片输出统一写入 `results/benchmark/figures/`（自动创建目录）。
- `scripts/analysis/analyze_defense_results.py`
  - 移除硬编码 Linux 路径，新增参数：
    - `--log-dir`（默认仓库根目录）
    - `--pattern`（默认 `output_malicious_*.log`）
    - `--out-dir`（默认 `results/analysis/`）
- `scripts/analysis/plot_defense_metrics.py`
  - 默认输入/输出改为 `results/analysis/`。
- `scripts/analysis/debug_defense_stats.py`
  - 改为支持 `--log-file` 传入一组日志文件。

## 5. 当前推荐复现入口

- 训练主实验：`python main_fed.py --defence scram ...`
- SCRAM 性能 benchmark：`python scripts/benchmark/run_benchmark_experiments.py --device cpu`
- 性能出图：
  - `python scripts/plotting/plot_scram_clients_lines.py`
  - `python scripts/plotting/plot_param_scale_lines.py`
- 日志解析与指标汇总：
  - `python scripts/analysis/analyze_defense_results.py --log-dir . --pattern "output_malicious_*.log"`

更完整命令示例见：[commands.md](./commands.md)。

## 6. 重要说明

- 本次归档不更改 `main_fed.py` 的默认 `--save save/...` 行为，因此原有 `save/` 目录保持不动，避免影响后续复现实验。
- `archive/legacy/Calculate/` 中的代码不作为论文 SCRAM 主线依赖，仅作为历史/对照实现保留。
