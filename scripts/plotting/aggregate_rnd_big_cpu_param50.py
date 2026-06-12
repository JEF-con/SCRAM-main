import os
import csv
from typing import List, Dict, Tuple

try:
    import numpy as np
except Exception:
    np = None

BASE_DIR = os.path.dirname(__file__)
FILES = [
    'newflame_timing_rnd_big_cpu_20251015_50_30.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_60.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_70.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_80.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_90.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_100.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_120.csv',
    'newflame_timing_rnd_big_cpu_20251015_50_150.csv',
]

STAGE_MAP = {
    '模型拆分成算术共享操作': 'model_split_s',
    '张量A和D生成及算术共享操作': 'A_D_s',
    '张量alpha、beta、gamma生成及算术共享操作': 'alpha_beta_gamma_s',
    'zeta相关操作': 'zeta_s',
    'deta相关操作': 'deta_s',
    '同态加密操作': 'homo_enc_s',
    '聚类算法': 'clustering_total_s',
    '余弦相似度矩阵计算': 'cosine_matrix_s',
    'HDBSCAN聚类算法': 'hdbscan_s',
    '客户端分类和选择': 'client_selection_s',
    '总体用时': 'total_time_s',
}

STAGE_ORDER = [
    'model_split_s','A_D_s','alpha_beta_gamma_s','zeta_s','deta_s','homo_enc_s',
    'clustering_total_s','cosine_matrix_s','hdbscan_s','client_selection_s','total_time_s'
]

def parse_file(path: str) -> Tuple[int, Dict[str, float]]:
    """Return (n_clients, stage_times_dict) parsed from a CSV file."""
    fname = os.path.basename(path)
    # Expect pattern ..._50_<n>.csv
    try:
        parts = fname.replace('.csv','').split('_')
        n_clients = int(parts[-1])
        param_size = int(parts[-2])  # currently not used, assumed 50
    except Exception:
        raise ValueError(f'Unexpected filename format: {fname}')

    stages: Dict[str, float] = {}
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # columns: 训练会话ID,调用次数,时间戳,操作名称,用时(秒),占比(%)
        for row in reader:
            if not row or len(row) < 6:
                continue
            op_name = row[3].strip()
            # strip leading dash for sub-ops
            op_name = op_name.lstrip('-').strip()
            time_s = float(row[4])
            key = STAGE_MAP.get(op_name)
            if key:
                stages[key] = time_s
    return n_clients, stages

def aggregate(files: List[str]) -> List[Dict[str, float]]:
    rows = []
    for rel in files:
        path = os.path.join(BASE_DIR, rel)
        n, s = parse_file(path)
        row = {'param_size': 50, 'n_clients': n}
        for k in STAGE_ORDER:
            row[k] = s.get(k)
        rows.append(row)
    rows.sort(key=lambda r: r['n_clients'])
    return rows

def write_csv(path: str, rows: List[Dict[str, float]]):
    fields = ['param_size','n_clients'] + STAGE_ORDER
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote', path)

def poly2_fit(xs: List[float], ys: List[float]):
    if np is None:
        # Fallback: simple linear fit a+bn ignoring quad
        A = [[1, x] for x in xs]
        # Solve via normal equations
        import math
        s00 = len(xs)
        s01 = sum(xs)
        s11 = sum(x*x for x in xs)
        t0 = sum(ys)
        t1 = sum(x*y for x,y in zip(xs,ys))
        det = s00*s11 - s01*s01
        a = (t0*s11 - s01*t1)/det
        b = (s00*t1 - s01*t0)/det
        return [b, a]  # return linear coeffs [b,a], interpret as [c1, c0]
    coeffs = np.polyfit(xs, ys, 2)  # returns [c2, c1, c0]
    return coeffs.tolist()

def predict_rows(rows: List[Dict[str, float]], targets: List[int]) -> List[Dict[str, float]]:
    # Fit per-stage vs n_clients using quadratic; if numpy missing, linear fallback
    xs = [r['n_clients'] for r in rows]
    preds = []
    for n in targets:
        out = {'param_size': 50, 'n_clients': n}
        for k in STAGE_ORDER:
            ys = [r[k] for r in rows if r.get(k) is not None]
            if not ys or len(ys) < 3 and np is not None:
                # if insufficient data, skip
                out[k] = None
                continue
            if np is None:
                # linear y ~ c1*x + c0
                coeffs = poly2_fit(xs, ys)
                if len(coeffs) == 2:
                    c1, c0 = coeffs
                    y = c1*n + c0
                else:
                    c2, c1, c0 = coeffs
                    y = c2*n*n + c1*n + c0
            else:
                c2, c1, c0 = poly2_fit(xs, ys)
                y = c2*n*n + c1*n + c0
            out[k] = float(y)
        # recompute total as sum of stage components if available
        parts = ['model_split_s','A_D_s','alpha_beta_gamma_s','zeta_s','deta_s','homo_enc_s','clustering_total_s','client_selection_s']
        if all(out.get(p) is not None for p in parts):
            out['total_time_s'] = sum(out[p] for p in parts)
        preds.append(out)
    return preds

def main():
    agg = aggregate(FILES)
    agg_path = os.path.join(BASE_DIR, 'newflame_rnd_big_cpu_param50_agg.csv')
    write_csv(agg_path, agg)
    targets = [160, 180, 200, 250]
    pred = predict_rows(agg, targets)
    pred_path = os.path.join(BASE_DIR, 'newflame_rnd_big_cpu_param50_predictions.csv')
    write_csv(pred_path, pred)

if __name__ == '__main__':
    main()