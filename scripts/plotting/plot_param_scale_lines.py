import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# LaTeX/MathText label mapping for legend (module-level)
latex_labels = {
    'clents_model_split_time': 'model split',
    'SC_BV_verify': 'SC-BV verify',
    'zeta': r'$\zeta$',
    'delta': r'$\delta$',
    'cosij': r'$\cos(i,j)$',
    'clustering_total_s': 'clustering',
    'total_time_s': 'total time'
}


def to_latex_label(name: str) -> str:
    return latex_labels.get(name, name)


def load_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 统一数值类型
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    return df


def setup_cjk_font():
    """选择可用的中文字体，避免中文字符缺失警告。"""
    candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK',
        'WenQuanYi Zen Hei',
        'Source Han Sans SC',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            print(f'Using font: {name}')
            return name
    # fallback
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    print('Chinese font not found, fallback to DejaVu Sans')
    return 'DejaVu Sans'


def parse_scales(columns):
    # 列名形如 300k、600k ... 提取数值部分用于 x 轴
    x_vals = []
    x_labels = []
    for c in columns[1:]:
        label = str(c)
        x_labels.append(label)
        try:
            if label.endswith('k'):
                x_vals.append(float(label[:-1]))
            else:
                x_vals.append(float(label))
        except Exception:
            # 兜底：无法解析时按顺序编号
            x_vals.append(len(x_vals) + 1)
    return x_vals, x_labels


def plot_stages_lines(df: pd.DataFrame, out_png: str):
    metrics = df['metric'].tolist()
    x_vals, x_labels = parse_scales(df.columns)

    # 选择需要绘制的分项（排除总耗时）
    stage_rows = [m for m in metrics if m != 'total_time_s']

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    markers = ['o', 's', 'D', '^', 'v', 'x', 'P', '*']

    fig_w = max(10, 1.1 * len(x_vals))
    fig_h = 6
    plt.figure(figsize=(fig_w, fig_h))

    # LaTeX/MathText label mapping for legend
    latex_labels = {
        'clents_model_split_time': r'\mathbf{\bullet~ShareGen}',
        'SC_BV_verify': r'\mathbf{\bullet~Verify}',
        'zeta': r'\zeta',
        'delta': r'\delta',
        'cosij': fr'\cos(i,j)',
        
        'clustering_total_s': r'\mathbf{\bullet~GlobalModelUpdate}',
        'total_time_s': r'\mathrm{total\ time}'
    }


    def to_latex_label(name: str) -> str:
        s = latex_labels.get(name, name)
        if '\\' in s or s != name:
            return f'${s}$'
        return s

    for idx, m in enumerate(stage_rows):
        row = df[df['metric'] == m].iloc[0]
        y = [row[c] for c in df.columns[1:]]
        plt.plot(
            x_vals, y,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=3,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='white',
            label=to_latex_label(m)
        )

    plt.xticks(x_vals, x_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('size of model (k)', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (s)', fontsize=14, fontweight='bold')
    plt.title('Runtime of SCRAM Stages', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(
        loc='upper left',
        fontsize=15,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        markerscale=2.0,
        handlelength=3.0,
        handletextpad=1.0,
        borderpad=0.8,
        columnspacing=1.4,
        labelspacing=0.7
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved stages lines to: {out_png}")


def plot_stages_lines_log(df: pd.DataFrame, out_png: str):
    metrics = df['metric'].tolist()
    x_vals, x_labels = parse_scales(df.columns)
    stage_rows = [m for m in metrics if m != 'total_time_s']

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    markers = ['o', 's', 'D', '^', 'v', 'x', 'P', '*']

    fig_w = max(10, 1.1 * len(x_vals))
    fig_h = 6
    plt.figure(figsize=(fig_w, fig_h))

    for idx, m in enumerate(stage_rows):
        row = df[df['metric'] == m].iloc[0]
        y = [row[c] for c in df.columns[1:]]
        plt.plot(
            x_vals, y,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=3,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='white',
            label=to_latex_label(m)
        )

    plt.xticks(x_vals, x_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('size of model (k)', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (s, log scale)', fontsize=14, fontweight='bold')
    plt.title('Runtime of SCRAM Stages (Log Scale)', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.yscale('log')
    plt.legend(
        loc='upper left',
        fontsize=15,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        markerscale=2.0,
        handlelength=3.0,
        handletextpad=1.0,
        borderpad=0.8,
        columnspacing=1.4,
        labelspacing=0.7
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved log-scale stages lines to: {out_png}")


def plot_total_line(df: pd.DataFrame, out_png: str):
    x_vals, x_labels = parse_scales(df.columns)
    row = df[df['metric'] == 'total_time_s'].iloc[0]
    y = [row[c] for c in df.columns[1:]]

    fig_w = max(8, 1.0 * len(x_vals))
    fig_h = 5
    plt.figure(figsize=(fig_w, fig_h))
    plt.plot(x_vals, y, color='#1f77b4', marker='o', linewidth=3, markersize=8, markeredgewidth=2, markeredgecolor='white')
    plt.xticks(x_vals, x_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('size of model (k)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Runtime (s)', fontsize=14, fontweight='bold')
    plt.title('Total Runtime of SCRAM Stages', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved total line to: {out_png}")


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / 'results' / 'benchmark' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = repo_root / 'results' / 'benchmark' / 'SCRAM_param_scale_table.csv'
    setup_cjk_font()
    df = load_table(csv_path)

    # Diagnostics: compare sum of plotted stages to total_time_s
    stage_rows = [m for m in df['metric'].tolist() if m != 'total_time_s']
    total_row = df[df['metric'] == 'total_time_s'].iloc[0]
    sum_series = pd.Series({})
    for c in df.columns[1:]:
        sum_series[c] = df[df['metric'].isin(stage_rows)][c].sum()
    total_series = total_row[df.columns[1:]]
    diff_series = total_series - sum_series
    diag_df = pd.DataFrame({
        'scale_k': df.columns[1:],
        'total_s': [total_series[c] for c in df.columns[1:]],
        'sum_plotted_s': [sum_series[c] for c in df.columns[1:]],
        'diff_s': [diff_series[c] for c in df.columns[1:]]
    })
    print('Diagnostic (scale_k, total_s, sum_plotted_s, diff_s):')
    print(diag_df.to_string(index=False))

    out_total = out_dir / 'SCRAM_param_scale_lines_total2026.png'
    out_stages = out_dir / 'SCRAM_param_scale_lines_stages2026.png'
    out_log = out_dir / 'SCRAM_param_scale_lines_stages_log2026.png'

    plot_total_line(df, out_total)
    plot_stages_lines(df, out_stages)
    plot_stages_lines_log(df, out_log)
