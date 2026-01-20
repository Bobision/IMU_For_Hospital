import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.preprocess_imu import PreprocessConfig,preprocess_df,extract_wavelet_coeffs,apply_autocorrelation
import matplotlib.pyplot as plt
import matplotlib as mpl
import pywt
mpl.rcParams['font.family'] = ['Times New Roman', 'SimSun']

# 简单滑动平均函数
def moving_average(x, w):
    # 简单滑动平均（用于能量），保证 w>=1
    w = max(int(w), 1)
    kernel = np.ones(w, dtype=float) / w
    # same 对齐，边缘用nearest填充（避免nan）
    pad = w // 2
    xpad = np.pad(x, (pad, w - 1 - pad), mode='edge')
    return np.convolve(xpad, kernel, mode='valid')
# 示例函数：基于Hiphaar小波系数的脚跟着地检测,下降零点
def detect_HS__Hiphaar(coeffs_dict, fs, columns,
                       energy_win_ms=500, # 局部能量计算窗口（毫秒）
                       energy_factor=0.8, # 自适应：阈值 = 中位能量 * factor
                       energy_abs=None# 绝对能量阈值，优先于 factor
                       ):
    """
    基于Hiphaar小波系数的足尖离地检测示例函数。
    
    参数:
    - coeffs_dict: dict，包含小波系数的字典，键为列名，值为对应的小波系数数组。
    - fs: float，采样频率。
    - columns: list，需要进行TO检测的列名列表。
    - threshold: float，检测阈值。
    - energy_win_ms : float，局部能量计算窗口大小（毫秒）。
    - min_sep_ms : float，最小间隔（毫秒）。
    返回:
    - to_events: dict，包含每个列名对应的TO事件时间点列表。
    """
    to_events = {}
    to_index={}
    win = max(int(round(energy_win_ms / 1000.0 * fs)), 3)   # 能量窗口点数
    #min_sep = max(int(round(min_sep_ms / 1000.0 * fs)), 1) # 最小间隔点数
    for col in columns:
        if col in coeffs_dict:
            coeffs = coeffs_dict[col]
            data = coeffs.iloc[:, -1].values  # 假设使用第一个小波系数列进行检测
            #局部能量计算
            energy_series = moving_average(data**2, win)
            # 2) 自适应阈值
            # 幅值阈值,暂不选用
            # if min_amp is None:
            #     amp_thresh = np.percentile(np.abs(data), amp_percentile)
            # else:
            #     amp_thresh = float(min_amp)

            # 能量阈值
            if energy_abs is None:
                baseline_energy = np.median(energy_series)
                energy_thresh = baseline_energy * float(energy_factor)
            else:
                energy_thresh = float(energy_abs)

            zero_indices = []
            scores = []
            zero_indices = []
            for i in range(1, len(data)):
                # 检查是否穿过零点（前一个点正，当前点负）并且是递减的
                if data[i-1] > 0 and data[i] < 0:
                    # 使用线性插值找到更精确的零点位置
                    x1, x2 = data[i-1], data[i]
                    # 插值权重
                    t = -x1 / (x2 - x1)
                    # 零点索引（浮点数，表示在i-1和i之间的位置）
                    zero_index = (int)(round(i - 1 + t))
                    if zero_index <0 or zero_index >= len(energy_series):
                        continue
                    e_local = energy_series[zero_index]
                    # 能量阈值检测
                    if e_local < energy_thresh:
                        continue

                    zero_indices.append(zero_index)
            # 去重并按最小间隔筛选
            to_times = np.array(zero_indices) / fs  # 转换为时间
            to_index[col]=zero_indices
            to_events[col] = to_times.tolist()
    return to_index,to_events
# 示例函数：基于Hiphaar小波系数的足尖离地检测
def detect_TO__Hiphaar(coeffs_dict, fs, columns,energy_win_ms=500, # 局部能量计算窗口（毫秒）
                       energy_factor=0.8, # 自适应：阈值 = 中位能量 * factor
                       energy_abs=None  # 绝对能量阈值，优先于 factor
                       ):
    """
    基于Hiphaar小波系数的足尖离地检测示例函数。
    
    参数:
    - coeffs_dict: dict，包含小波系数的字典，键为列名，值为对应的小波系数数组。
    - fs: float，采样频率。
    - columns: list，需要进行TO检测的列名列表。
    - threshold: float，检测阈值。
    
    返回:
    - to_events: dict，包含每个列名对应的TO事件时间点列表。
    """
    to_events = {}
    to_index={}
    win = max(int(round(energy_win_ms / 1000.0 * fs)), 3)   # 能量窗口点数
    for col in columns:
        if col in coeffs_dict:
            coeffs = coeffs_dict[col]
            data = coeffs.iloc[:, -1].values  # 假设使用第一个小波系数列进行检测
            #局部能量计算
            energy_series = moving_average(data**2, win)
            zero_indices = []
            # 能量阈值
            if energy_abs is None:
                baseline_energy = np.median(energy_series)
                energy_thresh = baseline_energy * float(energy_factor)
            else:
                energy_thresh = float(energy_abs)
            for i in range(1, len(data)):
                # 检查是否穿过零点（前一个点正，当前点负）并且是递减的
                if data[i-1] <0 and data[i] > 0:
                    # 使用线性插值找到更精确的零点位置
                    x1, x2 = data[i-1], data[i]
                    # 插值权重
                    t = -x1 / (x2 - x1)
                    # 零点索引（浮点数，表示在i-1和i之间的位置）
                    zero_index = (int)(round(i - 1 + t))
                    if zero_index <0 or zero_index >= len(energy_series):
                        continue
                    e_local = energy_series[zero_index]
                    # 能量阈值检测
                    if e_local < energy_thresh:
                        continue
                    zero_indices.append(zero_index)
                    # 简单阈值检测
            to_times = np.array(zero_indices) / fs  # 转换为时间
            to_index[col]=zero_indices
            to_events[col] = to_times.tolist()
    return to_index,to_events
# 基于周期百分比窗口对检测结果进行修剪
def prune_events_by_period_window(
    detection_results: Dict[str, List[int]],
    df_proc: pd.DataFrame,
    peaks_df_acf: pd.DataFrame,
    fs: float,
    signal_col_name: str = "signal",
    peak_col_name: str = "peak_lag_seconds",
    window_pct: float = 0.20,         # 窗口半宽 = window_pct * 周期
    pick: str = "max",                # "max" 或 "min"
    use_abs: bool = False,            # True 用绝对值比较
    min_gap_pct: Optional[float] = None  # 例如 0.5 表示 ≥0.5*T 的最小间隔
) -> Tuple[Dict[str, List[int]], Dict[str, dict]]:
    """
    按“周期百分比窗口”在每个窗口内只保留一个索引（最大或最小值所在点），其余剔除。

    Returns
    -------
    filtered_results : {signal: [idx, ...]}
    debug_info       : {signal: {"period_s":..., "period_samples":..., "win_half":..., 
                                 "kept": [...], "dropped": [...]} }
    """
    assert pick in ("max", "min"), "pick 必须是 'max' 或 'min'"

    # 把 peaks_df_acf 变成 {signal -> peak_lag_seconds} 的字典
    if signal_col_name in peaks_df_acf.columns:
        acf_map = dict(zip(peaks_df_acf[signal_col_name].astype(str),
                           peaks_df_acf[peak_col_name].astype(float)))
    else:
        # 兼容：signal 在 index
        acf_map = peaks_df_acf[peak_col_name].astype(float).to_dict()

    filtered_results: Dict[str, List[int]] = {}
    debug_info: Dict[str, dict] = {}

    for sig, idx_list in detection_results.items():
        # 无候选或 df 中无该列 -> 跳过
        if not idx_list or sig not in df_proc.columns:
            filtered_results[sig] = []
            debug_info[sig] = {"period_s": None, "period_samples": None, "win_half": None,
                               "kept": [], "dropped": []}
            continue

        # 取周期（秒/样本）
        if str(sig) not in acf_map:
            # 没有对应周期信息
            filtered_results[sig] = sorted(set(int(round(i)) for i in idx_list))
            debug_info[sig] = {"period_s": None, "period_samples": None, "win_half": None,
                               "kept": filtered_results[sig], "dropped": []}
            continue

        period_s = float(acf_map[str(sig)])
        period_samples = max(1, int(round(period_s * fs)))
        win_half = max(1, int(round(window_pct * period_samples)))
        min_gap = None if (min_gap_pct is None) else max(1, int(round(min_gap_pct * period_samples)))

        # 候选索引 & 对应值
        arr = np.unique(np.sort(np.round(np.asarray(idx_list, dtype=float)).astype(int)))
        y = pd.to_numeric(df_proc[sig], errors="coerce").to_numpy(dtype=float)

        kept: List[int] = []
        dropped: List[int] = []

        # 贪心：每次以最早的剩余索引为中心，合并窗口内的所有点，只留一个（最大/最小）
        remaining = arr.tolist()
        while remaining:
            center = remaining[0]
            # 窗口边界
            a = center-win_half
            b = center + win_half

            # 找窗口内的所有候选
            in_window = [i for i in remaining if a <= i <= b]
            # 选最大/最小值的索引
            if use_abs:
                vals = [abs(y[i]) if 0 <= i < len(y) else -np.inf for i in in_window]
            else:
                vals = [y[i] if 0 <= i < len(y) else -np.inf for i in in_window]

            if pick == "max":
                best_idx = in_window[int(np.nanargmax(vals))] if in_window else center
            else:
                # 取最小值
                if use_abs:
                    # 绝对值最小
                    vals_min = [abs(y[i]) if 0 <= i < len(y) else np.inf for i in in_window]
                    best_idx = in_window[int(np.nanargmin(vals_min))] if in_window else center
                else:
                    best_idx = in_window[int(np.nanargmin(vals))] if in_window else center

            kept.append(best_idx)
            # 其余都丢弃
            dropped.extend([i for i in in_window if i != best_idx])

            # 从 remaining 移除整个窗口
            remaining = [i for i in remaining if not (a <= i <= b)]

        # 可选：再做一次“最小间隔”约束（按时间顺序，保留更“好”的那个）
        if min_gap is not None and len(kept) > 1:
            kept = sorted(kept)
            final_kept = [kept[0]]
            for i in kept[1:]:
                if i - final_kept[-1] >= min_gap:
                    final_kept.append(i)
                else:
                    # 冲突：离得太近，保留幅值更“好”的一个
                    prev = final_kept[-1]
                    vi = abs(y[i]) if use_abs else y[i]
                    vp = abs(y[prev]) if use_abs else y[prev]
                    choose_new = (vi > vp) if pick == "max" else (vi < vp)
                    if choose_new:
                        # 替换
                        final_kept[-1] = i
                        dropped.append(prev)
                    else:
                        dropped.append(i)
            kept = final_kept

        filtered_results[sig] = kept
        debug_info[sig] = {
            "period_s": period_s,
            "period_samples": period_samples,
            "win_half": win_half,
            "kept": kept,
            "dropped": sorted(set(dropped))
        }

    return filtered_results, debug_info
# 基于 ACF 峰值间隔对 TO 检测结果进行配对与筛选
def pair_to_by_acf_gap(
    to_detection_results: Dict[str, List[int]],
    peaks_df_acf: pd.DataFrame,
    fs: float,
    peak_col: str = "peak_lag_seconds",
    tol_ratio: float = 0.2,                 # 相对容差，比如 ±20%
    tol_abs_samples: Optional[float] = None,# 绝对容差（样本数），优先于 ratio
    tol_abs_seconds: Optional[float] = None,# 或绝对容差（秒），优先于 ratio
    allow_overlap: bool = False             # 是否允许配对重叠（默认不重叠，贪心向前）
) -> Tuple[Dict[str, List[Tuple[int,int]]],
           Dict[str, List[int]],
           Dict[str, dict]]:
    """
    将 TO_detection_results 中的“相邻索引”按 ACF 的 peak_lag_seconds×fs 做配对与筛选。

    参数
    ----
    to_detection_results : {col: [idx0, idx1, ...]}  # 索引（样本点，下标整数或可转为int）
    peaks_df_acf         : DataFrame，index 为列名，包含列 `peak_lag_seconds`
    fs                   : 采样率 (Hz)
    peak_col             : ACF中表示滞后秒的列名
    tol_ratio            : 相对容差（无绝对容差时生效），例如 0.2 表示 ±20%
    tol_abs_samples      : 绝对容差（样本数）；若提供则优先使用
    tol_abs_seconds      : 绝对容差（秒）；若提供则优先使用（内部换算成样本数）
    allow_overlap        : True 允许(i,i+1)与(i+1,i+2)都配；False 不重叠（贪心跳过下一个）

    返回
    ----
    pairs_by_col         : {col: [(i,j), ...]}       # 通过筛选的成对索引（样本下标）
    kept_indices_by_col  : {col: [i,j,i2,j2,...]}    # 展开的非重叠索引（或按 allow_overlap 可能重复）
    meta_by_col          : {col: {"expected_gap_samples":..., "expected_gap_seconds":...,
                                  "tol_samples":..., "tol_seconds":...}}
    """
    pairs_by_col: Dict[str, List[Tuple[int,int]]] = {}
    kept_indices_by_col: Dict[str, List[int]] = {}
    meta_by_col: Dict[str, dict] = {}

    # 预处理容差
    tol_samples_from_seconds = None
    if tol_abs_seconds is not None:
        tol_samples_from_seconds = tol_abs_seconds * fs

    for col, idx_list in to_detection_results.items():
        # 没有 ACF 信息就跳过
        row = peaks_df_acf.loc[peaks_df_acf["signal"] == col]
        if row.empty:
            # 没找到对应的列名，跳过
            pairs_by_col[col] = []
            kept_indices_by_col[col] = []
            meta_by_col[col] = {"expected_gap_samples": None, "expected_gap_seconds": None,
                                "tol_samples": None, "tol_seconds": None}
            continue

        # 提取该列对应的 peak_lag_seconds
        lag_sec = float(row[peak_col].iloc[0])
        expected_gap_samples = lag_sec * fs
        expected_gap_seconds = lag_sec
        print(f"Column '{col}': expected gap = {expected_gap_samples} samples ({expected_gap_seconds:.3f} s)")
        # 计算容差（以样本为准）
        if tol_abs_samples is not None:
            tol_samples = float(tol_abs_samples)
            tol_seconds = tol_samples / fs
        elif tol_samples_from_seconds is not None:
            tol_samples = float(tol_samples_from_seconds)
            tol_seconds = tol_samples / fs
        else:
            tol_samples = float(abs(tol_ratio) * expected_gap_samples)
            tol_seconds = tol_samples / fs

        # 清洗索引（排序、唯一、整数化）
        if idx_list is None or len(idx_list) < 2:
            pairs_by_col[col] = []
            kept_indices_by_col[col] = []
            meta_by_col[col] = {"expected_gap_samples": expected_gap_samples,
                                "expected_gap_seconds": expected_gap_seconds,
                                "tol_samples": tol_samples, "tol_seconds": tol_seconds}
            continue

        arr = np.array(idx_list, dtype=float)
        arr = np.unique(np.sort(arr))
        # 强制取整样本下标（若你的列表保证是整型可去掉这步）
        arr = np.round(arr).astype(int)

        pairs = []
        used = np.zeros(len(arr), dtype=bool)

        i = 0
        while i < len(arr) - 1:
            gap = arr[i+1] - arr[i]
            if abs(gap - expected_gap_samples) <= tol_samples:
                pairs.append((arr[i], arr[i+1]))
                used[i] = True
                used[i+1] = True
                # 不重叠：跳过下一个起点
                i = i + 1 if allow_overlap else i + 2
            else:
                i += 1

        pairs_by_col[col] = pairs
        flat = [x for pair in pairs for x in pair]  # 扁平化

        if flat:
            kept_indices_by_col[col] = sorted(set(flat))
        else:
            kept_indices_by_col[col] = []

        meta_by_col[col] = {"expected_gap_samples": expected_gap_samples,
                            "expected_gap_seconds": expected_gap_seconds,
                            "tol_samples": tol_samples, "tol_seconds": tol_seconds}

    return pairs_by_col, kept_indices_by_col, meta_by_col
# 保留“且仅存在一个” between_source 索引的配对
def keep_pairs_with_unique_between(
    pairs_by_col: Dict[str, List[Tuple[int, int]]],
    HS_detection_results: Dict[str, List[int]],
    TO_detection_results: Dict[str, List[int]],
    between_source: Literal["TO", "HS"] = "TO",
    inclusive: bool = False
) -> Tuple[
    Dict[str, List[dict]],
    Dict[str, List[Tuple[int,int]]],
    Dict[str, List[Tuple[int,int]]],
    pd.DataFrame
]:
    """
    对每个配对 (i,j)，检查在两者之间是否存在“且仅存在一个”
    between_source（TO/HS）索引。若成立：
      - 保留该配对，并将中间索引一并记录，且与左右端点区分。
    返回:
      triplets_by_col: {col: [{'left':i,'between':k,'right':j,'between_tag':'TO'}, ...]}
      kept_pairs_by_col: {col: [(i,j), ...]}
      dropped_pairs_by_col: {col: [(i,j), ...]}
      triplets_df: DataFrame 扁平结果，列：signal,left,between,right,between_tag
    """
    triplets_by_col: Dict[str, List[dict]] = {}
    kept_pairs_by_col: Dict[str, List[Tuple[int,int]]] = {}
    dropped_pairs_by_col: Dict[str, List[Tuple[int,int]]] = {}

    # 选择要检查的索引集合
    source_map = {
        "TO": TO_detection_results,
        "HS": HS_detection_results
    }
    between_dict = source_map[between_source]

    for col, pairs in pairs_by_col.items():
        triplets_by_col[col] = []
        kept_pairs_by_col[col] = []
        dropped_pairs_by_col[col] = []

        # 该列的“中间候选”列表（排序去重为数组）
        between_list = between_dict.get(col, [])
        if not between_list:
            # 该列没有任何中间候选，全部丢弃
            dropped_pairs_by_col[col].extend(pairs or [])
            continue

        mid_arr = np.array(between_list, dtype=float)
        mid_arr = np.unique(np.sort(np.round(mid_arr).astype(int)))

        for i, j in (pairs or []):
            a, b = (i, j) if i <= j else (j, i)

            if inclusive:
                # [a, b] 内
                lo = np.searchsorted(mid_arr, a, side="left")
                hi = np.searchsorted(mid_arr, b, side="right")
            else:
                # (a, b) 内
                lo = np.searchsorted(mid_arr, a, side="right")
                hi = np.searchsorted(mid_arr, b, side="left")

            candidates = mid_arr[lo:hi]
            if len(candidates) == 1:
                k = int(candidates[0])
                triplets_by_col[col].append({
                    "left": int(a),
                    "between": k,
                    "right": int(b),
                    "between_tag": between_source
                })
                kept_pairs_by_col[col].append((int(a), int(b)))
            else:
                # 不是“恰好一个”，丢弃该配对
                dropped_pairs_by_col[col].append((int(a), int(b)))

    # 扁平化 DataFrame
    rows = []
    for col, lst in triplets_by_col.items():
        for d in lst:
            rows.append({
                "signal": col,
                **d
            })
    triplets_df = pd.DataFrame(rows, columns=["signal", "left", "between", "right", "between_tag"])

    return triplets_by_col, kept_pairs_by_col, dropped_pairs_by_col, triplets_df
# 绘制包含关键点和小波系数的信号图
def plot_triplets_with_coeffs(
    df_proc: pd.DataFrame,
    fs: float,
    triplets_df: pd.DataFrame,             # 列包含: ["signal","left","between","right","between_tag"]
    coeffs_dict: Optional[Dict[str, pd.DataFrame]] = None,  # 如 coeffs_d4
    show_coeff: bool = True,               # 是否绘制小波系数
    coeff_same_axis: bool = False,         # True: z-score 后同轴叠加；False: 右侧副轴
    outdir: str = "./eval_viz/triplet_plots",
    fig_width: float = 12,
    fig_height: float = 4,
    title_prefix: str = "",
    dpi: int = 300
):
    """
    为每个 signal 生成一张图：原始信号 + 关键点 (left/between/right) + 可选小波系数线。
    说明：
      - triplets_df['left','between','right'] 为样本下标（int）；本函数用 fs 转秒。
      - coeffs_dict[signal] 必须是单列 DataFrame，index 为秒（time_s）。
    """
    os.makedirs(outdir, exist_ok=True)
    if triplets_df.empty:
        return

    # 统一时间轴（原始信号）
    N = len(df_proc)
    t_raw = np.arange(N) / fs

    # 分 signal 出图
    for sig in triplets_df["signal"].unique():
        if sig not in df_proc.columns:
            # 该信号在 df_proc 里不存在，跳过
            continue

        y = pd.to_numeric(df_proc[sig], errors="coerce").to_numpy(float)
        # 画布
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        ax1.plot(t_raw, y, color="tab:gray", lw=1.0, alpha=0.85, label=f"{sig}")

        # 关键点（仅该 signal 的行）
        rows = triplets_df[triplets_df["signal"] == sig]

        # 画垂线 & 标注点
        # left / between / right 用不同颜色和标记
        colors = {"left": "tab:green", "between": "tab:orange", "right": "tab:red"}
        markers = {"left": "o", "between": "x", "right": "s"}
        labels_done = set()

        for _, r in rows.iterrows():
            iL = int(r["left"]); iM = int(r["between"]); iR = int(r["right"])
            tL, tM, tR = iL/fs, iM/fs, iR/fs

            # 安全范围判断
            if 0 <= iL < N:
                ax1.axvline(tL, color=colors["left"], ls="--", lw=0.8, alpha=0.6)
                ax1.plot(tL, y[iL], marker=markers["left"], color=colors["left"], ms=5,
                         label="left" if "left" not in labels_done else None)
                labels_done.add("left")
            if 0 <= iM < N:
                ax1.axvline(tM, color=colors["between"], ls="--", lw=0.8, alpha=0.6)
                ax1.plot(tM, y[iM], marker=markers["between"], color=colors["between"], ms=6,
                         label=f"{r.get('between_tag','between')}" if "between" not in labels_done else None)
                labels_done.add("between")
            if 0 <= iR < N:
                ax1.axvline(tR, color=colors["right"], ls="--", lw=0.8, alpha=0.6)
                ax1.plot(tR, y[iR], marker=markers["right"], color=colors["right"], ms=5,
                         label="right" if "right" not in labels_done else None)
                labels_done.add("right")

        # 可选：绘制小波系数
        if show_coeff and coeffs_dict is not None and sig in coeffs_dict:
            dfc = coeffs_dict[sig]
            # 取唯一一列
            if dfc.shape[1] == 1:
                coeff_name = dfc.columns[0]
            else:
                coeff_name = dfc.columns[0]
            tc = dfc.index.to_numpy(dtype=float)  # 秒
            cc = pd.to_numeric(dfc[coeff_name], errors="coerce").to_numpy(float)

            if coeff_same_axis:
                # z-score 后同轴叠加
                eps = 1e-12
                y_n = (y - np.nanmean(y)) / (np.nanstd(y) + eps)
                cc_n = (cc - np.nanmean(cc)) / (np.nanstd(cc) + eps)
                # 重画归一后的原始与系数
                ax1.clear()
                ax1.plot(t_raw, y_n, color="tab:gray", lw=1.0, alpha=0.9, label=f"{sig} (z)")
                ax1.plot(tc, cc_n, color="tab:blue", lw=1.1, alpha=0.9, label=f"{coeff_name} (z)")
                # 重新画关键点
                labels_done = set()
                for _, r in rows.iterrows():
                    for key, colr in colors.items():
                        idx = int(r[key])
                        t_idx = idx/fs
                        if 0 <= idx < N:
                            ax1.axvline(t_idx, color=colr, ls="--", lw=0.8, alpha=0.6)
                            ax1.plot(t_idx, 0, marker=markers[key], color=colr, ms=6,
                                     label=key if key not in labels_done else None)
                            labels_done.add(key)
                ax1.set_ylabel("z-score")
            else:
                # 右侧副轴
                ax2 = ax1.twinx()
                ax2.plot(tc, cc, color="tab:blue", lw=1.1, alpha=0.9, label=coeff_name)
                ax2.set_ylabel("Coeff")

                # 合并图例（双轴）
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            # 单轴图例
            ax1.legend(loc="upper right")

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(sig)
        title = f"{title_prefix}{sig}"
        if "between_tag" in triplets_df.columns:
            title += f"  | between={rows['between_tag'].iloc[0] if len(rows)>0 else ''}"
        ax1.set_title(title)
        ax1.grid(alpha=0.25)

        # 保存
        safe_sig = str(sig).replace("/", "_").replace("\\", "_")
        fpath = os.path.join(outdir, f"{safe_sig}_triplets.png")
        plt.tight_layout()
        plt.savefig(fpath, dpi=dpi)
        plt.close(fig)
# 根据 Triplets DataFrame 计算支撑相/摆动相占比
def compute_stance_swing_from_triplets_df(
    triplets_df: pd.DataFrame,
    left_cols: List[str],
    right_cols: List[str],
) -> Tuple[pd.DataFrame, List[float], List[float]]:
    """
    根据 Triplets DataFrame 计算支撑相/摆动相占比。
    输入:
      - triplets_df: 必须包含列 ["signal","left","between","right"]（样本下标，int）
                     可选包含 "between_tag"（例如 'TO'），会原样保留。
      - left_cols:   左腿相关的列名列表（与 triplets_df['signal'] 匹配）
      - right_cols:  右腿相关的列名列表

    输出:
      - out_df: 原始信息 + stance_s, stride_s(以样本为单位), stance_pct, swing_pct
      - left_stance_pct_list:  左腿对应 triplets 的 stance_pct 列表
      - right_stance_pct_list: 右腿对应 triplets 的 stance_pct 列表
    """
    required_cols = {"signal", "left", "between", "right"}
    missing = required_cols - set(triplets_df.columns)
    if missing:
        raise ValueError(f"triplets_df 缺少必要列: {sorted(missing)}")

    df = triplets_df.copy()

    # 确保是整数下标
    for c in ["left", "between", "right"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # 丢弃无效行：有空值或索引不满足 left < between < right
    df = df.dropna(subset=["left", "between", "right"]).reset_index(drop=True)
    valid_order = (df["left"] < df["between"]) & (df["between"] < df["right"])
    df = df[valid_order].reset_index(drop=True)

    # 计算 stride / stance（以样本计），再得到占比
    df["stride_samples"] = (df["right"] - df["left"]).astype(int)
    df = df[df["stride_samples"] > 0].reset_index(drop=True)

    df["stance_samples"] = (df["between"] - df["left"]).astype(int)
    # 占比
    df["stance_pct"] = df["stance_samples"] / df["stride_samples"]
    df["stance_pct"] = df["stance_pct"].clip(0.0, 1.0)
    df["swing_pct"]  = 1.0 - df["stance_pct"]

    # 标注左右腿（方便后续筛选）
    left_set  = set(map(str, left_cols))
    right_set = set(map(str, right_cols))
    sig_str = df["signal"].astype(str)

    df["leg_side"] = pd.Series(["unknown"] * len(df))
    df.loc[sig_str.isin(left_set),  "leg_side"] = "left"
    df.loc[sig_str.isin(right_set), "leg_side"] = "right"

    # 导出左右腿占比列表
    left_stance_list  = df.loc[df["leg_side"] == "left",  "stance_pct"].astype(float).tolist()
    right_stance_list = df.loc[df["leg_side"] == "right", "stance_pct"].astype(float).tolist()

    # 输出的完整表（保留你原来的所有列 + 新增列）
    out_cols = list(triplets_df.columns) + [
        "stride_samples", "stance_samples", "stance_pct", "swing_pct", "leg_side"
    ]
    out_df = df[out_cols].reset_index(drop=True)

    return out_df, left_stance_list, right_stance_list
# 根据 Triplets DataFrame 计算各周期内的最大值与最小值
def compute_cycle_max_min_from_triplets(
    triplets_df: pd.DataFrame,
    df_proc: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[float]], Dict[str, List[float]]]:
    """
    根据 Triplets by column 计算各周期内的最大值与最小值。

    参数
    ----------
    triplets_df : pd.DataFrame
        必须包含列 ["signal","left","between","right"]。
        left/right 为样本索引（int）。
    df_proc : pd.DataFrame
        对应的信号数据（经过预处理），列名应包含 columns 中的每个列。
    columns : list[str]
        想要计算的列名列表。

    返回
    ----------
    out_df : pd.DataFrame
        原 triplets_df + 对应周期内的最大值、最小值。
        每个 signal 独立计算。
    max_dict : dict[str, list[float]]
        键为列名，值为该列每个周期的最大值列表。
    min_dict : dict[str, list[float]]
        键为列名，值为该列每个周期的最小值列表。
    """
    required_cols = {"signal", "left", "right"}
    missing = required_cols - set(triplets_df.columns)
    if missing:
        raise ValueError(f"triplets_df 缺少必要列: {sorted(missing)}")

    df = triplets_df.copy().reset_index(drop=True)
    # 确保索引为整数
    df["left"] = pd.to_numeric(df["left"], errors="coerce").astype("Int64")
    df["right"] = pd.to_numeric(df["right"], errors="coerce").astype("Int64")

    N = len(df_proc)
    for col in columns:
        if col not in df_proc.columns:
            raise ValueError(f"列 '{col}' 不在 df_proc 中")

    # 准备结果
    max_dict: Dict[str, List[float]] = {c: [] for c in columns}
    min_dict: Dict[str, List[float]] = {c: [] for c in columns}

    # 依次遍历 triplets 的每一行
    cycle_max_list, cycle_min_list = [], []

    for i, row in df.iterrows():
        sig = row["signal"]
        iL, iR = int(row["left"]), int(row["right"])
        if not (0 <= iL < N and 0 <= iR < N and iL < iR):
            # 无效区间跳过
            cycle_max_list.append({c: None for c in columns})
            cycle_min_list.append({c: None for c in columns})
            continue

        # 提取该周期数据段
        seg = df_proc.iloc[iL:iR+1]

        # 计算各列的最大值/最小值
        seg_max = {}
        seg_min = {}
        for c in columns:
            vals = pd.to_numeric(seg[c], errors="coerce").dropna()
            if len(vals) == 0:
                vmax = vmin = None
            else:
                vmax = float(vals.max())
                vmin = float(vals.min())
            seg_max[c] = vmax
            seg_min[c] = vmin
            max_dict[c].append(vmax)
            min_dict[c].append(vmin)

        cycle_max_list.append(seg_max)
        cycle_min_list.append(seg_min)

    # 将每周期的结果展开到 df
    for c in columns:
        df[f"{c}_max"] = [cm[c] if cm[c] is not None else float("nan") for cm in cycle_max_list]
        df[f"{c}_min"] = [cm[c] if cm[c] is not None else float("nan") for cm in cycle_min_list]

    return df, max_dict, min_dict
# 对多个 signal 统一计算中位筛选均值
def compute_midmean_from_signals(
    out_df: pd.DataFrame,
    signals: List[str],
    columns: List[str],
    mid_num: int
) -> Dict[str, float]:
    """
    对多个 signal（如 cols_for_envelope）统一进行中位筛选平均。

    步骤：
    1. 对每个 signal 的每个列：
        - 去掉 NaN；
        - 计算中位数；
        - 选取 mid_num 个最接近中位数的值；
        - 求均值；
    2. 对所有 signal 的结果再次取平均（即最终每列一个均值）。

    参数
    ----------
    out_df : pd.DataFrame
        包含 signal 列、要处理的数值列（如 *_max, *_min 等）
    signals : list[str]
        要统一处理的信号列表（如 cols_for_envelope）
    columns : list[str]
        要处理的列名
    mid_num : int
        每个信号保留的中位点数量（应 ≤ 有效样本数）

    返回
    ----------
    global_mean_dict : dict[str, float]
        键为列名，值为在所有 signal 上平均后的均值。
    """
    if "signal" not in out_df.columns:
        raise KeyError("DataFrame 缺少必需列 'signal'。")

    df = out_df[out_df["signal"].isin(signals)].copy()

    mean_dict: Dict[str, float] = {}

    for col in columns:
        if col not in df.columns:
            # 和原实现保持一致，遇到缺列给个提示，但不中断
            print(f"⚠️ 列 '{col}' 不在 DataFrame 中，跳过。")
            continue

        # 统一转为数值，过滤无效
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vals)

        if n == 0:
            mean_dict[col] = np.nan
            continue

        if n <= mid_num:
            mean_dict[col] = float(np.mean(vals))
            continue

        median = np.median(vals)
        dist = np.abs(vals - median)

        # 选取与中位数差值最小的 mid_num 个样本
        # 用 argpartition 可降到 O(n)；再对这 mid_num 个做一次排序可选。
        keep_idx = np.argpartition(dist, kth=mid_num - 1)[:mid_num]
        kept_vals = vals[keep_idx]

        mean_dict[col] = float(np.mean(kept_vals))

    # 对于传入但在 df 中都不存在的列，补回键并设为 NaN（可选）
    for col in columns:
        if col not in mean_dict:
            mean_dict[col] = np.nan

    return mean_dict
    # per_signal_means: Dict[str, Dict[str, float]] = {}

    # for sig in signals:
    #     df_sub = out_df[out_df["signal"] == sig].reset_index(drop=True)
    #     mean_dict = {}

    #     for col in columns:
    #         if col not in df_sub.columns:
    #             print(f"⚠️ 列 '{col}' 不在 DataFrame 中，跳过。")
    #             continue

    #         vals = pd.to_numeric(df_sub[col], errors="coerce").dropna().to_numpy(float)
    #         n = len(vals)
    #         if n == 0:
    #             mean_dict[col] = np.nan
    #             continue

    #         if n <= mid_num:
    #             mean_dict[col] = float(np.mean(vals))
    #             continue

    #         median = np.median(vals)
    #         print(median)
    #         dist = np.abs(vals - median)
    #         keep_idx = np.argsort(dist)[:mid_num]
    #         kept_vals = vals[keep_idx]
    #         print(f"kept_vals:{kept_vals}")
    #         mean_dict[col] = float(np.mean(kept_vals))

    #     per_signal_means[sig] = mean_dict

    # # === 汇总：对所有信号的均值再求平均 ===
    # global_mean_dict: Dict[str, float] = {}
    # for col in columns:
    #     all_vals = [
    #         per_signal_means[sig][col]
    #         for sig in per_signal_means
    #         if col in per_signal_means[sig] and not np.isnan(per_signal_means[sig][col])
    #     ]
    #     global_mean_dict[col] = float(np.mean(all_vals)) if all_vals else np.nan

    # return global_mean_dict
#对多个 signal 统一计算最大值和最小值
def get_signalwise_max_min(
    length_angle_vals: pd.DataFrame,
    signals: List[str],
    columns: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    从 length_angle_vals 中获取指定 signals 对应 columns 的最大值和最小值。

    参数
    ----------
    length_angle_vals : pd.DataFrame
        包含 signal 列以及要处理的列（columns）
    signals : list[str]
        要处理的信号名称列表
    columns : list[str]
        要计算最大值、最小值的列名

    返回
    ----------
    max_dict : dict[str, float]
        键为列名，值为所有输入 signals 中该列的最大值
    min_dict : dict[str, float]
        键为列名，值为所有输入 signals 中该列的最小值
    """
    # --- 检查输入 ---
    if "signal" not in length_angle_vals.columns:
        raise ValueError("length_angle_vals 必须包含 'signal' 列。")
    for col in columns:
        if col not in length_angle_vals.columns:
            raise ValueError(f"列 '{col}' 不在 length_angle_vals 中。")

    # --- 过滤指定信号 ---
    df_sel = length_angle_vals[length_angle_vals["signal"].isin(signals)]
    if df_sel.empty:
        print("⚠️ 警告：指定的 signals 在 DataFrame 中不存在或匹配不到数据。")
        return {}, {}

    # --- 计算 ---
    max_dict: Dict[str, float] = {}
    min_dict: Dict[str, float] = {}

    for col in columns:
        vals = pd.to_numeric(df_sel[col], errors="coerce").dropna()
        if vals.empty:
            max_dict[col] = float("nan")
            min_dict[col] = float("nan")
        else:
            max_dict[col] = float(vals.max())
            min_dict[col] = float(vals.min())

    return max_dict, min_dict
# 从 triplets_df 的 right 索引提取对应列的数值
def extract_right_values_from_triplets(
    df_proc: pd.DataFrame,
    triplets_df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    根据 triplets_df 的 right 索引，从 df_proc 中提取对应列的数值。

    参数
    ----------
    df_proc : pd.DataFrame
        原始数据（按时间或样本索引行排列）
    triplets_df : pd.DataFrame
        含至少 ["signal", "right"] 列的 DataFrame
    columns : list[str]
        想要提取的 df_proc 中的列名

    返回
    ----------
    out_df : pd.DataFrame
        triplets_df 的拷贝 + 每列对应的右端点取值（如列名加后缀 _right_val）
    values_dict : dict[str, list[float]]
        键为列名，值为该列所有右端点的取值列表
    """
    # === 检查 ===
    if "right" not in triplets_df.columns:
        raise ValueError("triplets_df 必须包含列 'right'")
    for c in columns:
        if c not in df_proc.columns:
            raise ValueError(f"列 '{c}' 不在 df_proc 中")

    N = len(df_proc)
    out_df = triplets_df.copy().reset_index(drop=True)

    # 初始化结果字典
    values_dict: Dict[str, List[float]] = {c: [] for c in columns}

    # 遍历 triplets_df，每一行的 right 对应一个样本索引
    right_indices = pd.to_numeric(out_df["right"], errors="coerce").astype("Int64")

    for c in columns:
        vals = []
        col_data = pd.to_numeric(df_proc[c], errors="coerce").to_numpy(float)
        for idx in right_indices:
            if pd.isna(idx) or int(idx) < 0 or int(idx) >= N:
                vals.append(float("nan"))
            else:
                vals.append(float(col_data[int(idx)]))
        # 保存结果
        out_df[f"{c}_right_val"] = vals
        values_dict[c] = vals

    return out_df, values_dict
#计算左右腿步幅
def calculated_step_length2(left_mean_dict,
                                right_mean_dict,
                                thigh_length=50,
                                shank_length=40)->Tuple[Dict[str,float],Dict[str,float]]:
    """
    根据左右腿的最大最小值计算步幅
    """
    left_hipz_max=(left_mean_dict.get("imu2ang_x_max_sin",np.nan))
    left_hipz_min=(left_mean_dict.get("imu2ang_x_min_sin",np.nan))
    left_kneez_max=(left_mean_dict.get("imu3ang_x_max_sin",np.nan))
    left_kneez_min=(left_mean_dict.get("imu3ang_x_min_sin",np.nan))
    right_hipz_max=(right_mean_dict.get("imu5ang_x_max_sin",np.nan))
    right_hipz_min=(right_mean_dict.get("imu5ang_x_min_sin",np.nan))
    right_kneez_max=(right_mean_dict.get("imu6ang_x_max_sin",np.nan))
    right_kneez_min=(right_mean_dict.get("imu6ang_x_min_sin",np.nan))
    if np.isnan(left_hipz_max) or np.isnan(left_hipz_min) or np.isnan(right_hipz_max) or np.isnan(right_hipz_min):
        print("⚠️ 无法计算步幅，缺少必要的最大/最小值。")
        return np.nan, np.nan
    # 左腿步幅计算
    left_step_length_value= thigh_length * left_hipz_max+thigh_length*(-left_hipz_min)+ shank_length * left_kneez_max+shank_length * (-left_kneez_min)

    # 右腿步幅计算
    right_step_length_value = thigh_length * (right_hipz_max)+thigh_length*(-right_hipz_min)+ shank_length * (right_kneez_max)+shank_length *(-right_kneez_min)
    
    left_step_length={"left_step_length_value":left_step_length_value}
    right_step_length={"right_step_length_value":right_step_length_value}
    return left_step_length, right_step_length

def calculated_step_length(left_mean_dict,
                                right_mean_dict,
                                thigh_length=50,
                                shank_length=40)->Tuple[Dict[str,float],Dict[str,float]]:
    """
    根据左右腿的最大最小值计算步幅
    """
    left_hipz_max=np.deg2rad(left_mean_dict.get("imu2ang_x_max",np.nan))
    left_hipz_min=np.deg2rad(left_mean_dict.get("imu2ang_x_min",np.nan))
    left_kneez_max=np.deg2rad(left_mean_dict.get("imu3ang_x_max",np.nan))
    left_kneez_min=np.deg2rad(left_mean_dict.get("imu3ang_x_min",np.nan))
    right_hipz_max=np.deg2rad(right_mean_dict.get("imu5ang_x_max",np.nan))
    right_hipz_min=np.deg2rad(right_mean_dict.get("imu5ang_x_min",np.nan))
    right_kneez_max=np.deg2rad(right_mean_dict.get("imu6ang_x_max",np.nan))
    right_kneez_min=np.deg2rad(right_mean_dict.get("imu6ang_x_min",np.nan))
    if np.isnan(left_hipz_max) or np.isnan(left_hipz_min) or np.isnan(right_hipz_max) or np.isnan(right_hipz_min):
        print("⚠️ 无法计算步幅，缺少必要的最大/最小值。")
        return np.nan, np.nan
    # 左腿步幅计算
    left_step_length_value= thigh_length * np.sin(left_hipz_max)+thigh_length*np.sin(-left_hipz_min)+ shank_length * np.sin(left_kneez_max)+shank_length * np.sin(-left_kneez_min)

    # 右腿步幅计算
    right_step_length_value = thigh_length * np.sin(right_hipz_max)+thigh_length*np.sin(-right_hipz_min)+ shank_length * np.sin(right_kneez_max)+shank_length * np.sin(-right_kneez_min)
    
    left_step_length={"left_step_length_value":left_step_length_value}
    right_step_length={"right_step_length_value":right_step_length_value}
    return left_step_length, right_step_length
#计算左右腿步宽
def calculated_gait_width(left_width_mean,
                          right_width_mean,
                          left_hipz_col,
                          right_hipz_col,
                          thigh_length=50,
                          shank_length=40,
                          origin_width=20)->Tuple[Dict[str,float],Dict[str,float]]:
    """
    根据髋关节角度计算步宽
    """
    left_width_value=(
    origin_width
    +np.sin(np.deg2rad(left_width_mean.get(f"{left_hipz_col}_right_val",np.nan)))*(thigh_length+shank_length)
    -np.sin(np.deg2rad(left_width_mean.get(f"{right_hipz_col}_right_val",np.nan)))*(thigh_length+shank_length)
    )
    left_width={"left_width_value":left_width_value}
    #(8.4)计算右步宽
    right_width_value=(
        origin_width
    +np.sin(np.deg2rad(right_width_mean.get(f"{left_hipz_col}_right_val",np.nan)))*(thigh_length+shank_length)
    -np.sin(np.deg2rad(right_width_mean.get(f"{right_hipz_col}_right_val",np.nan)))*(thigh_length+shank_length)
    )
    right_width={"right_width_value":right_width_value}
    return left_width, right_width
#计算左右步长
def calculated_gait_length(left_length_mean,
                           right_length_mean,
                           left_hipx_col,
                           right_hipx_col,
                           left_kneex_col,
                           right_kneex_col,
                           thigh_length=50,
                           shank_length=40)->Tuple[Dict[str,float],Dict[str,float]]:
    """
    根据髋关节角度计算步长
    """
    left_length_value=(
        thigh_length * np.sin(np.deg2rad(-left_length_mean.get(f"{left_hipx_col}_right_val",np.nan)))
        +shank_length*np.sin(np.deg2rad(-left_length_mean.get(f"{left_kneex_col}_right_val",np.nan)))
        #+shank_length* np.sin(np.deg2rad(-left_length_mean.get(f"{left_kneex_col}_right_val",np.nan)))
        +thigh_length * np.sin(np.deg2rad(left_length_mean.get(f"{right_hipx_col}_right_val",np.nan)))
        +shank_length* np.sin(np.deg2rad(left_length_mean.get(f"{right_kneex_col}_right_val",np.nan)))
    )
    left_length={"left_length_value":left_length_value}
    #(8.6)计算右步长
    right_length_value=(
        thigh_length * np.sin(np.deg2rad(-right_length_mean.get(f"{right_hipx_col}_right_val",np.nan)))
        +shank_length* np.sin(np.deg2rad(-right_length_mean.get(f"{right_kneex_col}_right_val",np.nan)))
        +thigh_length * np.sin(np.deg2rad(right_length_mean.get(f"{left_hipx_col}_right_val",np.nan)))
        +shank_length* np.sin(np.deg2rad(right_length_mean.get(f"{left_kneex_col}_right_val",np.nan)))
    )
    right_length={"right_length_value":right_length_value}
    return left_length, right_length
#计算左右步长方法2
def calculated_gait_length_raw(left_length_mean,
                           right_length_mean,
                           left_hipx_col,
                           right_hipx_col,
                           left_kneex_col,
                           right_kneex_col,
                           thigh_length=50,
                           shank_length=40)->Tuple[Dict[str,float],Dict[str,float]]:
    """
    根据髋关节角度计算步长
    """
    left_length_value=(
        thigh_length * np.cos(np.deg2rad(-left_length_mean.get(f"{left_hipx_col}_right_val",np.nan)))
        +shank_length*np.cos(np.deg2rad(-left_length_mean.get(f"{left_kneex_col}_right_val",np.nan)))
        #+shank_length* np.sin(np.deg2rad(-left_length_mean.get(f"{left_kneex_col}_right_val",np.nan)))
        +thigh_length * np.cos(np.deg2rad(left_length_mean.get(f"{right_hipx_col}_right_val",np.nan)))
        +shank_length* np.cos(np.deg2rad(left_length_mean.get(f"{right_kneex_col}_right_val",np.nan)))
    )
    left_length={"left_length_value":left_length_value}
    #(8.6)计算右步长
    right_length_value=(
        thigh_length * np.cos(np.deg2rad(-right_length_mean.get(f"{right_hipx_col}_right_val",np.nan)))
        +shank_length* np.cos(np.deg2rad(-right_length_mean.get(f"{right_kneex_col}_right_val",np.nan)))
        +thigh_length * np.cos(np.deg2rad(right_length_mean.get(f"{left_hipx_col}_right_val",np.nan)))
        +shank_length* np.cos(np.deg2rad(right_length_mean.get(f"{left_kneex_col}_right_val",np.nan)))
    )                            
    right_length={"right_length_value":right_length_value}
    return left_length, right_length

#角度转sin值
def degree_to_sin(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    将角度列转换为 sin 值列（安全无副作用）
    """
    df = df.copy()  # 防止切片赋值问题
    for col in columns:
        if col in df.columns:
            sin_col = f"{col}_sin"
            df[sin_col] = np.sin(np.deg2rad(pd.to_numeric(df[col], errors="coerce").astype(float)))
    return df

#根据main的调试情况，将所有步态参数计算的过程改成一个函数
def calculate_gait_parameters(df_raw: pd.DataFrame,
                              cfg: PreprocessConfig,
                              thigh_length: float = 50,
                              shank_length: float = 40,
                              origin_width: float = 20,
                              plot: bool=False,
                              save: bool=False,
                              ) -> Dict[str, float]:
    df_proc, info=preprocess_df(df_raw,cfg)
    cols_for_envelope = []
    left_cols=[]
    right_cols=[]
    for ax in ["imu2ang_x","imu3ang_x","imu4ang_x"]:
        if ax in df_proc.columns:
            cols_for_envelope.append(ax)
            left_cols.append(ax)
    for ax in ["imu5ang_x","imu6ang_x","imu7ang_x"]:
        if ax in df_proc.columns:
            cols_for_envelope.append(ax)
            right_cols.append(ax)
    left_hipz_col="imu2ang_z"
    right_hipz_col="imu5ang_z"
    left_hipx_col="imu2ang_x"
    right_hipx_col="imu5ang_x"
    left_kneex_col="imu3ang_x"
    right_kneex_col="imu6ang_x"
    #print("预处理信息:", info)
    #print("用于包络和后续分析的列:", cols_for_envelope)
    print("----------------------------------------------")
    # ----------------------------步态参数识别流程---------------------------
    if True:
        acf_df, peaks_df_acf = apply_autocorrelation(
        df_proc,
        columns=cols_for_envelope,                  # 需要做ACF的列
        plot=plot,
        plot_outdir="./eval_viz/ACF_plots",
        fs=cfg.fs,      # 图片保存目录
        max_lag_seconds=10.0,                        # 最多看5秒滞后（可选）
        fill_limit=10,                              # 连续≤10个NaN线性补
        detrend=True,                               # 去均值（推荐）
        unbiased=True                               # 无偏归一
        )
        
        #(1.0)对“峰值间隔”统一计算中位筛选均值,当做步态周期时间
        left_maen_period=compute_midmean_from_signals(
            out_df=peaks_df_acf,
            signals=left_cols,
            columns=["peak_lag_seconds"],
            mid_num=3   
        )
        right_mean_period=compute_midmean_from_signals(
            out_df=peaks_df_acf,
            signals=right_cols,
            columns=["peak_lag_seconds"],
            mid_num=3
        )
        #根据均值来选择步频
        peaks_df_acf.loc[peaks_df_acf['signal'].isin(left_cols), 'peak_lag_seconds'] = left_maen_period["peak_lag_seconds"]
        peaks_df_acf.loc[peaks_df_acf['signal'].isin(right_cols), 'peak_lag_seconds'] = right_mean_period["peak_lag_seconds"]
        #(3)提取小波特征,用于后续的TO检测
        coeffs_d4, bands_d4 = extract_wavelet_coeffs(
        df_proc, columns=cols_for_envelope, fs=cfg.fs,
        wavelet="haar", max_level=5,
        transform_type="swt",
        part="detail", select_level=2,
        mode="symmetric",
        plot=plot, plot_outdir="./eval_viz/wavelet_coeffs/swtbior"
        )   
        #提取小波特征,用于后续的HS检测
        coeffs_d3, bands_d3 = extract_wavelet_coeffs(
        df_proc, columns=cols_for_envelope, fs=cfg.fs,
        wavelet="haar", max_level=5,
        transform_type="swt",
        part="detail", select_level=5,
        mode="symmetric",
        plot=plot, plot_outdir="./eval_viz/wavelet_coeffs/swtbior"
        )

        #(4)基于小波系数的足尖离地检测
        coeffs=coeffs_d4.copy()
        TO_detection_results,TO_time = detect_TO__Hiphaar(coeffs, cfg.fs, cols_for_envelope)
        HS_detection_results,HS_time = detect_HS__Hiphaar(coeffs_d3, cfg.fs, cols_for_envelope)
        print("----------------------------------------------")
        print("TO_time:",TO_time)
        print("HS_time:",HS_time)   
        print("----------------------------------------------")
        #基于 ACF 峰值间隔对 HS 检测结果进行修剪
        filtered_HS, dbg = prune_events_by_period_window(
            detection_results=HS_detection_results,   # {signal: [idx...]}
            df_proc=df_proc,
            peaks_df_acf=peaks_df_acf,
            fs=cfg.fs,
            window_pct=0.6,          # 每个周期 ±20% 的窗口
            pick="min",               # 在窗口内选最大值所在的索引
            use_abs=False,            # 如果你的波峰有时是负的，可改 True
            min_gap_pct=0.50          # 可选：再保证相邻保留点至少相隔 0.5 个周期
        )   

        filtered_TO, dbg = prune_events_by_period_window(
            detection_results=TO_detection_results,   # {signal: [idx...]}
            df_proc=df_proc,
            peaks_df_acf=peaks_df_acf,
            fs=cfg.fs,
            window_pct=0.6,          # 每个周期 ±20% 的窗口
            pick="max",               # 在窗口内选最大值所在的索引
            use_abs=False,            # 如果你的波峰有时是负的，可改 True
            min_gap_pct=0.50          # 可选：再保证相邻保留点至少相隔 0.5 个周期
        )
        print("----------------------------------------------")
        print("Filtered HS indices:", filtered_HS)
        print("Filtered TO indices:", filtered_TO)
        print("----------------------------------------------")
        #(5)基于 ACF 峰值间隔对 HS 检测结果进行配对与筛选
        print("Pairing HS events based on ACF peak gaps...")
        pairs_by_col, kept_idx_by_col, meta_by_col = pair_to_by_acf_gap(
        filtered_HS,      # dict: {col: [idx...]}
        peaks_df_acf,              # df: index=col, 有列 'peak_lag_seconds'
        fs=cfg.fs,
        peak_col="peak_lag_seconds",
        tol_ratio=0.15,             # ±20% 的相对窗口
        # 或者用绝对容差：
        # tol_abs_seconds=0.06,    # ±60 ms
        # tol_abs_samples=3,       # ±3 个样本
        allow_overlap=True          # 允许重叠配对
        )
       
        print("----------------------------------------------")
        print("Pairing TO events based on ACF peak gaps...")
        
        #(5)基于 ACF 峰值间隔对 TO 检测结果进行配对与筛选
        _, kept_TO_idx, meta_by_col_TO = pair_to_by_acf_gap(
            filtered_TO,      # dict: {col: [idx...]}
            peaks_df_acf,              # df: index=col, 有列 'peak_lag_seconds'
            fs=cfg.fs,
            peak_col="peak_lag_seconds",
            tol_ratio=0.15,             # ±20% 的相对窗口
            # 或者用绝对容差：
            # tol_abs_seconds=0.06,    # ±60 ms
            # tol_abs_samples=3,       # ±3 个样本
            allow_overlap=True          # 允许重叠配对
            )
        
        print("----------------------------------------------")
        #(6)保留“且仅存在一个” TO 索引的配对
        triplets_by_col, kept_pairs_by_col, dropped_pairs_by_col, triplets_df = keep_pairs_with_unique_between(
            pairs_by_col=pairs_by_col,                         # 你的配对结果
            HS_detection_results=HS_detection_results,         # dict: {col: [idx...]}
            TO_detection_results=kept_TO_idx,         # dict: {col: [idx...]}
            between_source="TO",                               # 在配对两端之间找“唯一 TO”
            inclusive=False                                    # 严格介于两端；如需把端点也算上改 True
        )
        print("----------------------------------------------")
        print("Triplets by column:")
        print(triplets_df.head(20))

        #(6.0)基于 Triplets 计算支撑相/摆动相占比
        out_df, left_list, right_list = compute_stance_swing_from_triplets_df(
            triplets_df=triplets_df,          # 必含: signal,left,between,right
            left_cols=left_cols,  # 你的左腿相关列名
            right_cols=right_cols  # 你的右腿相关列名
        )

        print(out_df.head())
        print("左腿支撑相占比：", left_list[:10])
        print("右腿支撑相占比：", right_list[:10])
        #(6.1)对左腿支撑相摆动相统一计算中位筛选均值`
        left_mean_phase=compute_midmean_from_signals(
            out_df=out_df,
            signals=left_cols,
            columns=["stance_pct","swing_pct"],
            mid_num=3 
        )
        #（6.2)对右腿支撑相摆动相统一计算中位筛选均值`
        right_mean_phase=compute_midmean_from_signals(
            out_df=out_df,
            signals=right_cols,
            columns=["stance_pct","swing_pct"],
            mid_num=3
        )
        #(7)基于 Triplets 计算各周期内的最大值与最小值，计算用于步幅的角度值
        Num_df, max_dict, min_dict = compute_cycle_max_min_from_triplets(
            triplets_df=triplets_df,
            df_proc=df_proc,
            columns=cols_for_envelope
        )
        print(Num_df.head())

        # Num_df_sin=degree_to_sin(df=Num_df,columns=[f"{col}_max" for col in cols_for_envelope] +
        #             [f"{col}_min" for col in cols_for_envelope])
        #(7.1)对左侧腿的角度极值筛选，统一计算中位筛选均值
        left_mean_dict = compute_midmean_from_signals(
            out_df=Num_df,
            signals=left_cols,
            columns=[f"{col}_max" for col in left_cols] +
                    [f"{col}_min" for col in left_cols],
            mid_num=3
        )
        
        #(7.2)对右侧腿的角度极值筛选，统一计算中位筛选均值
        right_mean_dict = compute_midmean_from_signals(
            out_df=Num_df,
            signals=right_cols,
            columns=[f"{col}_max" for col in right_cols] +
                    [f"{col}_min" for col in right_cols],
            mid_num=3
        )
        #(7.3)计算左右腿步幅长度
        gaitlength_left,gaitlength_right=calculated_step_length(left_mean_dict,right_mean_dict,thigh_length=thigh_length,shank_length=shank_length)

        kneez_col=["imu3ang_z","imu6ang_z"]
        #(8)计算用于步宽计算的角度信息
        width_angle_vals, right_vals = extract_right_values_from_triplets(
            df_proc=df_proc,
            triplets_df=triplets_df,
            columns=[left_hipz_col, right_hipz_col]+kneez_col
        )
        hipz_col=[left_hipz_col, right_hipz_col]

        #(8.1)对左侧腿筛选，统一计算中位筛选均值
        left_width_mean=compute_midmean_from_signals(
            out_df=width_angle_vals,
            signals=left_cols,
            columns=[f"{col}_right_val" for col in hipz_col],
            mid_num=1
        )
        #(8.2)对左侧腿筛选，统一计算中位筛选均值
        right_width_mean=compute_midmean_from_signals(
            out_df=width_angle_vals,
            signals=right_cols,
            columns=[f"{col}_right_val" for col in hipz_col],
            mid_num=1
        )
        #(8.3)计算左步宽
        left_width,right_width=calculated_gait_width(
            left_width_mean=left_width_mean,
            right_width_mean=right_width_mean,
            left_hipz_col=left_hipz_col,
            right_hipz_col=right_hipz_col,
            thigh_length=thigh_length,
            shank_length=shank_length,
            origin_width=origin_width)

        step_width={"step_width":0.5*(left_width.get("left_width_value",np.nan)+right_width.get("right_width_value",np.nan))}
        #(8.4)计算右步宽
        #(9)计算用于步长计算的角度信息
        length_angle_vals, right_vals = extract_right_values_from_triplets(
            df_proc=df_proc,
            triplets_df=triplets_df,
            columns=["imu2ang_x", "imu3ang_x","imu5ang_x","imu6ang_x"]
        )
        hip_knee_cols=["imu2ang_x","imu3ang_x","imu5ang_x","imu6ang_x"]

        max_dict_left, min_dict_left = get_signalwise_max_min(
        length_angle_vals=length_angle_vals,
        signals=left_cols,
        columns=[f"{col}_right_val" for col in hip_knee_cols]
    )

        max_dict_right, min_dict_right = get_signalwise_max_min(
        length_angle_vals=length_angle_vals,
        signals=right_cols,
        columns=[f"{col}_right_val" for col in hip_knee_cols]
    )
        keys_to_extract_left = ['imu2ang_x_right_val', 'imu3ang_x_right_val']
        keys_to_extract_right = ['imu5ang_x_right_val', 'imu6ang_x_right_val']
        #提取计算步长要用的最大最小值
        left_length_use={**{key: min_dict_left[key] for key in keys_to_extract_left},
                         **{key: max_dict_left[key] for key in keys_to_extract_right}}
        right_length_use={**{key: max_dict_right[key] for key in keys_to_extract_left},
                          **{key: min_dict_right[key] for key in keys_to_extract_right}}
        # #(9.1)对左侧腿筛选，统一计算中位筛选均值
        # left_length_mean=compute_midmean_from_signals(
        #     out_df=length_angle_vals,
        #     signals=left_cols,
        #     columns=[f"{col}_right_val" for col in hip_knee_cols],
        #     mid_num=1
        # )
        # #(9.2)对右侧腿筛选，统一计算中位筛选均值
        # right_length_mean=compute_midmean_from_signals(
        #     out_df=length_angle_vals,
        #     signals=right_cols,
        #     columns=[f"{col}_right_val" for col in hip_knee_cols],
        #     mid_num=1
        # )

        #(9.3)计算左右腿步长
        left_length,right_length=calculated_gait_length(left_length_mean=left_length_use,
                                                        right_length_mean= right_length_use,
                                                        left_hipx_col=left_hipx_col,
                                                        right_hipx_col=right_hipx_col,
                                                        left_kneex_col=left_kneex_col,
                                                        right_kneex_col=right_kneex_col,
                                                        thigh_length=thigh_length,
                                                        shank_length=shank_length)
       
        #(9.4)计算步幅
        step_length={"step_length":left_length.get(("left_length_value"),np.nan)+right_length.get("right_length_value",np.nan)}
        if left_maen_period.get("peak_lag_seconds",np.nan)==0:
            step_speed_mean={"step_speed":np.nan}
        elif right_mean_period.get("peak_lag_seconds",np.nan)==0:
            step_speed_mean={"step_speed":np.nan}
        else:
            step_speed_mean={"step_speed":step_length.get("step_length",np.nan)/((left_maen_period.get("peak_lag_seconds",np.nan)+right_mean_period.get("peak_lag_seconds",np.nan))/2)}

        if plot:
            #(10)可视化
            plot_triplets_with_coeffs(
            df_proc=df_proc,
            fs=cfg.fs,
            triplets_df=triplets_df,
            coeffs_dict=coeffs_d4,     # 或 None
            show_coeff=True,           # 是否画系数
            coeff_same_axis=False,     # True=同轴(z-score)，False=右侧副轴
            outdir="./eval_viz/triplet_plots_1026",
            title_prefix="Triplets | "
            )
     
    #(11)保存步态参数
    if True:
        #计算步态参数
        gait_params = {}
        if left_maen_period.get("peak_lag_seconds",np.nan)!=0:
            gait_params[f"左腿周期(s)"] = round(left_maen_period.get("peak_lag_seconds",np.nan),2)
            gait_params[f"左腿步频(步/s)"] = round(1/left_maen_period.get("peak_lag_seconds",np.nan),2)
        else:
            gait_params[f"左腿周期(s)"] =np.nan
            gait_params[f"左腿步频(步/s)"] = np.nan
        gait_params[f"左腿支撑相占比(%)"] = 100*round(left_mean_phase.get("stance_pct",np.nan),4)
        gait_params[f"左腿摆动相占比(%)"] = 100*round(left_mean_phase.get("swing_pct",np.nan),4)
        gait_params[f"左腿步长(cm)"] = round(left_length.get("left_length_value",np.nan),2)
        if right_mean_period.get("peak_lag_seconds",np.nan)!=0:
            gait_params[f"右腿周期(s)"] = round(right_mean_period.get("peak_lag_seconds",np.nan),2)
            gait_params[f"右腿步频(步/s)"] = round(1/right_mean_period.get("peak_lag_seconds",np.nan),2)
        else:
            gait_params[f"右腿周期(s)"] =np.nan
            gait_params[f"右腿步频(步/s)"] = np.nan
        gait_params[f"右腿支撑相占比(%)"] = 100*round(right_mean_phase.get("stance_pct",np.nan),4)
        gait_params[f"右腿摆动相占比(%)"] = 100*round(right_mean_phase.get("swing_pct",np.nan),4)
        gait_params[f"右腿步长(cm)"] = round(right_length.get("right_length_value",np.nan),2)
        gait_params["步宽(cm)"] = abs(round(step_width.get("step_width",np.nan),2))
        gait_params[f"平均步速(cm/s)"] = round(step_speed_mean.get("step_speed",np.nan),2)
        gait_params["步幅(cm)"] = round(step_length.get("step_length",np.nan),2)
        if left_maen_period.get("peak_lag_seconds",np.nan)!=0:
            if right_mean_period.get("peak_lag_seconds",np.nan)!=0:
                gait_params[f"步频(步/s)"] =0.5*(round(1/left_maen_period.get("peak_lag_seconds",np.nan),2)+round(1/right_mean_period.get("peak_lag_seconds",np.nan),2))
            else:
                gait_params[f"步频(步/s)"] =round(1/left_maen_period.get("peak_lag_seconds",np.nan),2)
        else:
            if right_mean_period.get("peak_lag_seconds",np.nan)!=0:
                gait_params[f"步频(步/s)"] =round(1/right_mean_period.get("peak_lag_seconds",np.nan),2)
            else:
                gait_params[f"步频(步/s)"] =0

        # print("计算得到的步态参数示例:")
        # for k, v in gait_params.items():
        #     print(f"{k}: {v}")
        #保存为 CSV 文件
        #gait_params_df = pd.DataFrame([gait_params])
        #gait_params_df.to_csv("gait_parameters_output.csv", index=False)
        #print("步态参数已保存到 'gait_parameters_output.csv' 文件中。")
        return gait_params

if __name__ == "__main__":
    #df_raw=pd.read_csv("E:/demo/python/Gait1016/datasets/ZJH/gait01_results/gait01_pre00.csv",encoding='utf-8-sig')
    df_raw=pd.read_csv("E:/demo/python/Gait1016/datasets/rawZJH/gait09_results/gait09_pre00.csv",encoding='utf-8-sig')
    #(1)配置预处理参数
    # thigh_length=55
    # shank_length=44
    # origin_width=30
    cfg = PreprocessConfig(
        fs=50.0,                 # 你的原始采样率
        time_col="时间",           # 若CSV没有时间列就用None；有的话写列名，如 "time"
        drop_initial_seconds=0,# 丢弃起步0.5s（可选）
        baseline_frames=20,      # 去基线
        interpolate_nan=True,    # 插值
        resample_hz=None,        # 是否统一重采样，通常保持None
        unwrap_angle=False,      # 角度是否去包裹
        use_angle_sincos=False,  # 是否展开角度为sin/cos（做分类/回归时常用）
        # 滤波：角度/陀螺/加速度分组不同cut-off
        ang_low_hz=6.0,
        gyr_low_hz=15.0,
        acc_high_hz=0.3,         # 去漂移
        acc_low_hz=20.0,         # 平滑
        moving_median_k=0,
        moving_mean_k=0,
        enable_zupt=False,       # 如需对足IMU做简化ZUPT可开
        add_magnitude_for=("acc","vel"),  # 自动添加模长列
        add_rms_k=0,
        normalize="None",      # 训练/推理一致时建议用zscore
        stats_path=None,         # 如果已有训练统计量可填 npz 路径
        clip_after_std=None,    # 如需截断可设值
        plot=False               # 是否另存“预处理前后对比图”
    )

    #(2)读取数据并预处理
    df_proc, info=preprocess_df(df_raw,cfg)
    cols_for_envelope = []
    left_cols=[]
    right_cols=[]
    for ax in ["imu2ang_x","imu3ang_x","imu4ang_x"]:
        if ax in df_proc.columns:
            cols_for_envelope.append(ax)
            left_cols.append(ax)
    for ax in ["imu5ang_x","imu6ang_x","imu7ang_x"]:
        if ax in df_proc.columns:
            cols_for_envelope.append(ax)
            right_cols.append(ax)
    left_hipz_col="imu2ang_z"
    right_hipz_col="imu5ang_z"
    left_hipx_col="imu2ang_x"
    right_hipx_col="imu5ang_x"
    left_kneex_col="imu3ang_x"
    right_kneex_col="imu6ang_x"
    print("预处理信息:", info)
    print("用于包络和后续分析的列:", cols_for_envelope)
    print("----------------------------------------------")
    # ----------------------------步态参数识别流程---------------------------
    if True:
        acf_df, peaks_df_acf = apply_autocorrelation(
        df_proc,
        columns=cols_for_envelope,                  # 需要做ACF的列
        plot=True,
        plot_outdir="./eval_viz/ACF_plots",
        fs=cfg.fs,      # 图片保存目录
        max_lag_seconds=10.0,                        # 最多看5秒滞后（可选）
        fill_limit=10,                              # 连续≤10个NaN线性补
        detrend=True,                               # 去均值（推荐）
        unbiased=True                               # 无偏归一
        )
        
        #(1.0)对“峰值间隔”统一计算中位筛选均值,当做步态周期时间
        left_maen_period=compute_midmean_from_signals(
            out_df=peaks_df_acf,
            signals=left_cols,
            columns=["peak_lag_seconds"],
            mid_num=1   
        )
        right_mean_period=compute_midmean_from_signals(
            out_df=peaks_df_acf,
            signals=right_cols,
            columns=["peak_lag_seconds"],
            mid_num=1   
        )
        #根据均值来选择步频
        peaks_df_acf.loc[peaks_df_acf['signal'].isin(left_cols), 'peak_lag_seconds'] = left_maen_period["peak_lag_seconds"]
        peaks_df_acf.loc[peaks_df_acf['signal'].isin(right_cols), 'peak_lag_seconds'] = right_mean_period["peak_lag_seconds"]
        #(3)提取小波特征,用于后续的TO检测
        coeffs_d4, bands_d4 = extract_wavelet_coeffs(
        df_proc, columns=cols_for_envelope, fs=cfg.fs,
        wavelet="haar", max_level=5,
        transform_type="swt",
        part="detail", select_level=2,
        mode="symmetric",
        plot=True, plot_outdir="./eval_viz/wavelet_coeffs/swtbior"
        )   
        #提取小波特征,用于后续的HS检测
        coeffs_d3, bands_d3 = extract_wavelet_coeffs(
        df_proc, columns=cols_for_envelope, fs=cfg.fs,
        wavelet="haar", max_level=5,
        transform_type="swt",
        part="detail", select_level=5,
        mode="symmetric",
        plot=True, plot_outdir="./eval_viz/wavelet_coeffs/swtbior"
        )

        #(4)基于小波系数的足尖离地检测
        coeffs=coeffs_d4.copy()
        TO_detection_results,TO_time = detect_TO__Hiphaar(coeffs, cfg.fs, cols_for_envelope)
        HS_detection_results,HS_time = detect_HS__Hiphaar(coeffs_d3, cfg.fs, cols_for_envelope)
        print("----------------------------------------------")
        print("TO_time:",TO_time)
        print("HS_time:",HS_time)   
        print("----------------------------------------------")
        #基于 ACF 峰值间隔对 HS 检测结果进行修剪
        filtered_HS, dbg = prune_events_by_period_window(
            detection_results=HS_detection_results,   # {signal: [idx...]}
            df_proc=df_proc,
            peaks_df_acf=peaks_df_acf,
            fs=cfg.fs,
            window_pct=0.6,          # 每个周期 ±20% 的窗口
            pick="min",               # 在窗口内选最大值所在的索引
            use_abs=False,            # 如果你的波峰有时是负的，可改 True
            min_gap_pct=0.50          # 可选：再保证相邻保留点至少相隔 0.5 个周期
        )   

        filtered_TO, dbg = prune_events_by_period_window(
            detection_results=TO_detection_results,   # {signal: [idx...]}
            df_proc=df_proc,
            peaks_df_acf=peaks_df_acf,
            fs=cfg.fs,
            window_pct=0.6,          # 每个周期 ±20% 的窗口
            pick="max",               # 在窗口内选最大值所在的索引
            use_abs=False,            # 如果你的波峰有时是负的，可改 True
            min_gap_pct=0.50          # 可选：再保证相邻保留点至少相隔 0.5 个周期
        )
        print("----------------------------------------------")
        print("Filtered HS indices:", filtered_HS)
        print("Filtered TO indices:", filtered_TO)
        print("----------------------------------------------")
        #(5)基于 ACF 峰值间隔对 HS 检测结果进行配对与筛选
        print("Pairing HS events based on ACF peak gaps...")
        pairs_by_col, kept_idx_by_col, meta_by_col = pair_to_by_acf_gap(
        filtered_HS,      # dict: {col: [idx...]}
        peaks_df_acf,              # df: index=col, 有列 'peak_lag_seconds'
        fs=cfg.fs,
        peak_col="peak_lag_seconds",
        tol_ratio=0.15,             # ±20% 的相对窗口
        # 或者用绝对容差：
        # tol_abs_seconds=0.06,    # ±60 ms
        # tol_abs_samples=3,       # ±3 个样本
        allow_overlap=True          # 允许重叠配对
        )
       
        print("----------------------------------------------")
        print("Pairing TO events based on ACF peak gaps...")
        
        #(5)基于 ACF 峰值间隔对 TO 检测结果进行配对与筛选
        _, kept_TO_idx, meta_by_col_TO = pair_to_by_acf_gap(
            filtered_TO,      # dict: {col: [idx...]}
            peaks_df_acf,              # df: index=col, 有列 'peak_lag_seconds'
            fs=cfg.fs,
            peak_col="peak_lag_seconds",
            tol_ratio=0.15,             # ±20% 的相对窗口
            # 或者用绝对容差：
            # tol_abs_seconds=0.06,    # ±60 ms
            # tol_abs_samples=3,       # ±3 个样本
            allow_overlap=True          # 允许重叠配对
            )
        
        print("----------------------------------------------")
        #(6)保留“且仅存在一个” TO 索引的配对
        triplets_by_col, kept_pairs_by_col, dropped_pairs_by_col, triplets_df = keep_pairs_with_unique_between(
            pairs_by_col=pairs_by_col,                         # 你的配对结果
            HS_detection_results=HS_detection_results,         # dict: {col: [idx...]}
            TO_detection_results=kept_TO_idx,         # dict: {col: [idx...]}
            between_source="TO",                               # 在配对两端之间找“唯一 TO”
            inclusive=False                                    # 严格介于两端；如需把端点也算上改 True
        )
        print("----------------------------------------------")
        print("Triplets by column:")
        print(triplets_df.head(20))

        #(6.0)基于 Triplets 计算支撑相/摆动相占比
        out_df, left_list, right_list = compute_stance_swing_from_triplets_df(
            triplets_df=triplets_df,          # 必含: signal,left,between,right
            left_cols=left_cols,  # 你的左腿相关列名
            right_cols=right_cols  # 你的右腿相关列名
        )

        print(out_df.head())
        print("左腿支撑相占比：", left_list[:10])
        print("右腿支撑相占比：", right_list[:10])
        #(6.1)对左腿支撑相摆动相统一计算中位筛选均值`
        left_mean_phase=compute_midmean_from_signals(
            out_df=out_df,
            signals=left_cols,
            columns=["stance_pct","swing_pct"],
            mid_num=5   
        )
        #（6.2)对右腿支撑相摆动相统一计算中位筛选均值`
        right_mean_phase=compute_midmean_from_signals(
            out_df=out_df,
            signals=right_cols,
            columns=["stance_pct","swing_pct"],
            mid_num=5
        )
        #(7)基于 Triplets 计算各周期内的最大值与最小值，计算用于步幅的角度值
        Num_df, max_dict, min_dict = compute_cycle_max_min_from_triplets(
            triplets_df=triplets_df,
            df_proc=df_proc,
            columns=cols_for_envelope
        )
        print(Num_df.head())

        # Num_df_sin=degree_to_sin(df=Num_df,columns=[f"{col}_max" for col in cols_for_envelope] +
        #             [f"{col}_min" for col in cols_for_envelope])
        #(7.1)对左侧腿的角度极值筛选，统一计算中位筛选均值
        left_mean_dict = compute_midmean_from_signals(
            out_df=Num_df,
            signals=left_cols,
            columns=[f"{col}_max" for col in left_cols] +
                    [f"{col}_min" for col in left_cols],
            mid_num=5
        )
        
        #(7.2)对右侧腿的角度极值筛选，统一计算中位筛选均值
        right_mean_dict = compute_midmean_from_signals(
            out_df=Num_df,
            signals=right_cols,
            columns=[f"{col}_max" for col in right_cols] +
                    [f"{col}_min" for col in right_cols],
            mid_num=5
        )
        #(7.3)计算左右腿步幅长度
        gaitlength_left,gaitlength_right=calculated_step_length(left_mean_dict,right_mean_dict,thigh_length=thigh_length,shank_length=shank_length)

        kneez_col=["imu3ang_z","imu6ang_z"]
        #(8)计算用于步宽计算的角度信息
        width_angle_vals, right_vals = extract_right_values_from_triplets(
            df_proc=df_proc,
            triplets_df=triplets_df,
            columns=[left_hipz_col, right_hipz_col]+kneez_col
        )
        hipz_col=[left_hipz_col, right_hipz_col]

        #(8.1)对左侧腿筛选，统一计算中位筛选均值
        left_width_mean=compute_midmean_from_signals(
            out_df=width_angle_vals,
            signals=left_cols,
            columns=[f"{col}_right_val" for col in hipz_col],
            mid_num=1
        )
        #(8.2)对左侧腿筛选，统一计算中位筛选均值
        right_width_mean=compute_midmean_from_signals(
            out_df=width_angle_vals,
            signals=right_cols,
            columns=[f"{col}_right_val" for col in hipz_col],
            mid_num=1
        )
        #(8.3)计算左步宽
        left_width,right_width=calculated_gait_width(
            left_width_mean=left_width_mean,
            right_width_mean=right_width_mean,
            left_hipz_col=left_hipz_col,
            right_hipz_col=right_hipz_col,
            thigh_length=thigh_length,
            shank_length=shank_length,
            origin_width=origin_width)

        step_width={"step_width":0.5*(left_width.get("left_width_value",np.nan)+right_width.get("right_width_value",np.nan))}
        #(8.4)计算右步宽
        #(9)计算用于步长计算的角度信息
        length_angle_vals, right_vals = extract_right_values_from_triplets(
            df_proc=df_proc,
            triplets_df=triplets_df,
            columns=["imu2ang_x", "imu3ang_x","imu5ang_x","imu6ang_x"]
        )
        hip_knee_cols=["imu2ang_x","imu3ang_x","imu5ang_x","imu6ang_x"]

        max_dict_left, min_dict_left = get_signalwise_max_min(
        length_angle_vals=length_angle_vals,
        signals=left_cols,
        columns=[f"{col}_right_val" for col in hip_knee_cols]
    )

        max_dict_right, min_dict_right = get_signalwise_max_min(
        length_angle_vals=length_angle_vals,
        signals=right_cols,
        columns=[f"{col}_right_val" for col in hip_knee_cols]
    )
        keys_to_extract_left = ['imu2ang_x_right_val', 'imu3ang_x_right_val']
        keys_to_extract_right = ['imu5ang_x_right_val', 'imu6ang_x_right_val']
        #提取计算步长要用的最大最小值
        left_length_use={**{key: min_dict_left[key] for key in keys_to_extract_left},
                         **{key: max_dict_left[key] for key in keys_to_extract_right}}
        right_length_use={**{key: max_dict_right[key] for key in keys_to_extract_left},
                          **{key: min_dict_right[key] for key in keys_to_extract_right}}
        # #(9.1)对左侧腿筛选，统一计算中位筛选均值
        # left_length_mean=compute_midmean_from_signals(
        #     out_df=length_angle_vals,
        #     signals=left_cols,
        #     columns=[f"{col}_right_val" for col in hip_knee_cols],
        #     mid_num=1
        # )
        # #(9.2)对右侧腿筛选，统一计算中位筛选均值
        # right_length_mean=compute_midmean_from_signals(
        #     out_df=length_angle_vals,
        #     signals=right_cols,
        #     columns=[f"{col}_right_val" for col in hip_knee_cols],
        #     mid_num=1
        # )

        #(9.3)计算左右腿步长
        left_length,right_length=calculated_gait_length(left_length_mean=left_length_use,
                                                        right_length_mean= right_length_use,
                                                        left_hipx_col=left_hipx_col,
                                                        right_hipx_col=right_hipx_col,
                                                        left_kneex_col=left_kneex_col,
                                                        right_kneex_col=right_kneex_col,
                                                        thigh_length=thigh_length,
                                                        shank_length=shank_length)
       
        #(9.4)计算步幅
        step_length={"step_length":left_length.get(("left_length_value"),np.nan)+right_length.get("right_length_value",np.nan)}
        if left_maen_period.get("peak_lag_seconds",np.nan)==0:
            step_speed_mean={"step_speed":np.nan}
        elif right_mean_period.get("peak_lag_seconds",np.nan)==0:
            step_speed_mean={"step_speed":np.nan}
        else:
            step_speed_mean={"step_speed":step_length.get("step_length",np.nan)/((left_maen_period.get("peak_lag_seconds",np.nan)+right_mean_period.get("peak_lag_seconds",np.nan))/2)}

        #(10)可视化
        plot_triplets_with_coeffs(
        df_proc=df_proc,
        fs=cfg.fs,
        triplets_df=triplets_df,
        coeffs_dict=coeffs_d4,     # 或 None
        show_coeff=True,           # 是否画系数
        coeff_same_axis=False,     # True=同轴(z-score)，False=右侧副轴
        outdir="./eval_viz/triplet_plots_1026",
        title_prefix="Triplets | "
        )
     
    #(11)保存步态参数
    if True:
        #计算步态参数
        gait_params = {}
        if left_maen_period.get("peak_lag_seconds",np.nan)!=0:
            gait_params[f"左腿步频(步/s)"] = round(1/left_maen_period.get("peak_lag_seconds",np.nan),2)
        else:
            gait_params[f"左腿步频(步/s)"] = np.nan
        gait_params[f"左腿支撑相占比(%)"] = 100*round(left_mean_phase.get("stance_pct",np.nan),2)
        gait_params[f"左腿摆动相占比(%)"] = 100*round(left_mean_phase.get("swing_pct",np.nan),2)
        gait_params[f"左腿步长(cm)"] = round(left_length.get("left_length_value",np.nan),2)
        if right_mean_period.get("peak_lag_seconds",np.nan)!=0:
            gait_params[f"右腿步频(步/s)"] = round(1/right_mean_period.get("peak_lag_seconds",np.nan),2)
        else:
            gait_params[f"右腿步频(步/s)"] = np.nan
        gait_params[f"右腿支撑相(%)"] = 100*round(right_mean_phase.get("stance_pct",np.nan),2)
        gait_params[f"右腿摆动相(%)"] = 100*round(right_mean_phase.get("swing_pct",np.nan),2)
        gait_params[f"右腿步长(cm)"] = round(right_length.get("right_length_value",np.nan),2)
        gait_params["步宽(cm)"] = abs(round(step_width.get("step_width",np.nan),2))
        gait_params[f"平均步速(cm/s)"] = round(step_speed_mean.get("step_speed",np.nan),2)
        gait_params["步幅(cm)"] = round(step_length.get("step_length",np.nan),2)
        print("计算得到的步态参数示例:")
        for k, v in gait_params.items():
            print(f"{k}: {v}")
        #保存为 CSV 文件
        gait_params_df = pd.DataFrame([gait_params])
        gait_params_df.to_csv("gait_parameters_output.csv", index=False)
        print("步态参数已保存到 'gait_parameters_output.csv' 文件中。")
