import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 保留原始小数，但也确保是字符串以便渲染
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    return df


def render_table(df: pd.DataFrame, out_png: str):
    # 列标签（模型规模，单位 k）
    col_labels = df.columns.tolist()
    col_labels[0] = "操作项 (s)"
    # 将列名中的数值加上单位说明（k）
    for i in range(1, len(col_labels)):
        col_labels[i] = f"{col_labels[i]}"

    # 表格数据
    cell_text = []
    for _, row in df.iterrows():
        r = [row[0]]
        for c in df.columns[1:]:
            r.append(f"{row[c]:.6f}")
        cell_text.append(r)

    n_rows = len(df)
    n_cols = len(col_labels)

    # 根据行列数自适应画布尺寸
    fig_w = max(12, 1.2 * n_cols)
    fig_h = max(2 + 0.6 * n_rows, 6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    # 创建表格
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # 标题：单位说明
    ax.set_title(
        "newFlame 模型参数规模 vs 操作耗时表\n模型规模单位：k；耗时单位：s",
        fontsize=14,
        pad=20
    )

    # 保存图片
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved table image to: {out_png}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "newflame_param_scale_table.csv")
    out_png = os.path.join(base_dir, "newflame_param_scale_table.png")
    df = load_data(csv_path)
    render_table(df, out_png)