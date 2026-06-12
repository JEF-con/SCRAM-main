import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path


def setup_cjk_font():
    candidates = [
        'Noto Sans CJK SC', 'Noto Sans CJK', 'WenQuanYi Zen Hei',
        'Source Han Sans SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            print(f'Using font: {name}')
            return
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    print('Chinese font not found, fallback to DejaVu Sans')

# LaTeX/MathText label mapping for legend
# latex_labels = {
#     'clents_model_split_time': r'\\mathrm{model\\ split}',
#     'SC_BV_verify': r'\\mathrm{SC\\text{-}BV\\ verify}',
#     'zeta': r'\\zeta',
#     'delta': r'\\delta',
#     'cosjj': r'\\cos(i,j)',
#     'cosij': r'\\cos(i,j)',
#     'clustering_total_s': r'\\mathrm{clustering}',
#     'total_time_s': r'\\mathrm{total\\ time}'
# }
latex_labels = {
    'clents_model_split_time': r'{ShareGen}',
    'SC_BV_verify': r'{Verify}',
    'delta': r'\delta',
    'cosij': r'\cos_ij',
    'clustering_total_s': r'{GlobalModelUpdate}',
    'total_time_s': r'\mathrm{total\ time}'
}
def to_latex_label(name: str) -> str:
    s = latex_labels.get(name, name)
    if '\\' in s or s != name:
        return f'${s}$'
    return s


def load_clients_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['n_clients'] = df['n_clients'].astype(int)
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    return df


def plot_total(df: pd.DataFrame, out_png: str):
    x = df['n_clients'].tolist()
    y = df['total_time_s'].tolist()
    # 紧凑画布尺寸
    fig_w, fig_h = 12, 5
    plt.figure(figsize=(fig_w, fig_h))
    plt.plot(x, y, color='#1f77b4', marker='o', linewidth=3, markersize=8, markeredgewidth=2, markeredgecolor='white')
    plt.xticks(x, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('the number of clients (n)', fontsize=14, fontweight='bold')
    plt.ylabel('total time (s)', fontsize=14, fontweight='bold')
    plt.title('SCRAM total time vs number of clients (line plot)', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f'Saved total line to: {out_png}')


def plot_stages(df: pd.DataFrame, out_png: str, log_scale: bool = False):
    x = df['n_clients'].tolist()
    stage_cols = [c for c in df.columns if c not in ['n_clients', 'total_time_s']]
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'
    ]
    markers = ['o', 's', 'D', '^', 'v', 'x', 'P', '*', 'h']

    # 紧凑画布尺寸
    fig_w, fig_h = 12, 6
    plt.figure(figsize=(fig_w, fig_h))
    for idx, col in enumerate(stage_cols):
        y = df[col].tolist()
        plt.plot(
            x, y,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=3,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='white',
            label=to_latex_label(col)
        )

    plt.xticks(x, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('the number of clients (n)', fontsize=14, fontweight='bold')
    plt.ylabel('time (s)' + (',log scale' if log_scale else ''), fontsize=14, fontweight='bold')
    title = 'SCRAM stages time vs number of clients (line plot)'
    if log_scale:
        plt.yscale('log')
        title = 'SCRAM stages time(log scale)vs number of clients'
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(
        loc='upper left',
        fontsize=15,
        ncol=2,
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
    print(f'Saved stages lines to: {out_png}')


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / 'results' / 'benchmark' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_cjk_font()
    csv_path = repo_root / 'results' / 'benchmark' / 'SCRAM_clients_timing_summary.csv'
    df = load_clients_csv(csv_path)

    # Diagnostics: compare sum of plotted stages to total_time_s
    stage_cols = [c for c in df.columns if c not in ['n_clients', 'total_time_s']]
    df['plotted_sum_s'] = df[stage_cols].sum(axis=1)
    df['diff_s'] = df['total_time_s'] - df['plotted_sum_s']
    print('Diagnostic (n, total, sum_plotted, diff):')
    print(df[['n_clients','total_time_s','plotted_sum_s','diff_s']].to_string(index=False))

    out_total = out_dir / 'SCRAM_clients_lines_total2026.png'
    out_stages = out_dir / 'SCRAM_clients_lines_stages2026.png'
    out_log = out_dir / 'SCRAM_clients_lines_stages_log2026.png'

    plot_total(df, out_total)
    plot_stages(df, out_stages, log_scale=False)
    plot_stages(df, out_log, log_scale=True)
