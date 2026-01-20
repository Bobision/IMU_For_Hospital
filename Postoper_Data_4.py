import numpy as np
import csv
import os
import pandas as pd
from preprocessing.preprocess_imu import PreprocessConfig,preprocess_df,extract_wavelet_coeffs,apply_autocorrelation
from scripts.TO_detect import calculate_gait_parameters

# ============================ 读取CSV文件 ============================
def read_csv_file(filename, encoding='gbk'):
    """
    读取CSV文件。
    :param filename: str, 输入的CSV文件名
    :param encoding: str, 文件编码
    :return: tuple, 包括数据数组和标题行
    """
    encodings_to_try = [
        'gbk',
        'utf-8-sig',
        'Windows-1252',  # 优先尝试，0xd0在该编码中有效
        'cp1252',        # 与Windows-1252基本一致
        'ISO-8859-1',    # 兼容大部分西欧语言
        'KOI8-R',        # 俄语编码，0xd0在其中有效     # 带BOM的utf-8
        'gb18030',       # 中文编码超集
        'latin-1'        # 最后尝试，兼容所有字节
    ]
    for encoding in encodings_to_try:
        try:
            with open(filename, newline='', encoding=encoding) as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # 获取标题行
                rows = list(reader)
            data = np.array(rows, dtype=float)
            print(f"✅ 使用编码 '{encoding}' 成功读取文件：{filename}")
            return data, headers
        except UnicodeDecodeError:
                print(f"❌ 编码 '{encoding}' 读取失败，继续尝试...")
                continue
        except Exception as e:
            print(f"⚠️ 读取文件时发生其他错误：{str(e)}")
            raise
    
    raise ValueError(f"❌ 所有编码尝试失败，无法读取文件：{filename}")
# 保存csv文件
def save_csv_file(data, headers, filename, encoding='utf-8-sig'):
    """
    保存数据到CSV文件。
    :param data: np.ndarray, 数据数组
    :param headers: list, 标题行
    :param filename: str, 输出的文件名
    :param encoding: str, 文件编码
    """
    with open(filename, mode='w', newline='', encoding=encoding) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)
#创建文件路径
def create_result_folder(original_path):
    """
    根据原始文件路径创建结果文件夹，格式：原文件名_results
    :param original_path: str, 原始文件完整路径（如"C:/data/file.csv"）
    :return: str, 结果文件夹路径（如"C:/data/file_results/"）
    """
    # 分离文件路径与扩展名
    dir_path, filename = os.path.split(original_path)
    base_name = os.path.splitext(filename)[0]

    # 构建结果文件夹路径
    result_dir = os.path.join(dir_path, f"{base_name}_results")

    # 创建文件夹（若不存在）
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

# ============================ 处理关节角+IMU(+足底压力)数据 ============================
def process_imu_joint_and_plantar_safe_ndarray(
    data: np.ndarray,
    headers: list,
    target_hz: int = 50,
    time_col: str | None = None,
    interp_method: str = "pchip",
    max_gap_seconds: float = 10.0,
    allow_extrapolate: bool = False,
    unwrap_angle_cols: list | None = None,
    plantar_left_cols: list | None = None,
    plantar_right_cols: list | None = None,
):
    """
    安全处理关节角+IMU(+可选足底压力)的数据：
    - 关节角单列去重
    - IMU每9列整体去重
    - 可选：足底压力L(18列)/R(18列)整体去重
    - 短缺口插值到 target_hz，大缺口置NaN
    - 可选对“角度列”先 unwrap 再插值
    - 自动识别足底列名(L_1..L_18 / R_1..R_18)，找不到则跳过足底处理
    - 返回：processed_data(ndarray), processed_headers(list)
    """
    if time_col is None:
        time_col = headers[0]

    df = pd.DataFrame(data, columns=headers).astype(float)
    headers = df.columns.tolist()

    # === 生成目标时间轴 ===
    t_min, t_max = df[time_col].min(), df[time_col].max()
    dt = 1.0 / target_hz
    num_points = int(np.floor((t_max - t_min) * target_hz)) + 1
    t_target = np.linspace(t_min, t_min + (num_points - 1) * dt, num_points)
    t_out = t_target - t_target[0]  # 相对时间
    result_columns = [t_out]

    unwrap_set = set(unwrap_angle_cols or [])

    # ---------- 自动识别足底列（若未显式传入） ----------
    def _autodetect_plantar(prefix: str, total: int = 18):
        cname = [f"{prefix}_{i}" for i in range(1, total + 1)]
        return cname if all(c in headers for c in cname) else None

    if plantar_left_cols is None:
        plantar_left_cols = _autodetect_plantar("L", 18)
    if plantar_right_cols is None:
        plantar_right_cols = _autodetect_plantar("R", 18)

    has_left_plantar = isinstance(plantar_left_cols, (list, tuple)) and \
                       all(c in headers for c in plantar_left_cols) and len(plantar_left_cols) == 18
    has_right_plantar = isinstance(plantar_right_cols, (list, tuple)) and \
                        all(c in headers for c in plantar_right_cols) and len(plantar_right_cols) == 18

    # ---------- 拆分“关节角、IMU、足底”列域 ----------
    # 约定：前7列是关节角（你的原逻辑）
    joint_cols = headers[1:8]

    # 从整体 headers 中去掉关节角和足底（如果存在），剩下的是 IMU 列（9 的倍数）
    plantar_cols_all = []
    if has_left_plantar:
        plantar_cols_all += plantar_left_cols
    if has_right_plantar:
        plantar_cols_all += plantar_right_cols

    reserved = set([time_col] + joint_cols + plantar_cols_all)
    imu_cols = [c for c in headers if c not in reserved]

    # === 工具：组内“整体去重+插值” ===
    def _interp_block(cols,ignore_all_zero=False):
        """对给定列列表做：整体去重 -> 每列插值，返回插值后的列序列"""
        block_df = df[[time_col] + cols].copy().reset_index(drop=True)
        # 检测变化：任意一列变化才算“有变化”
        diff_mask = block_df[cols].diff().fillna(1).ne(0).any(axis=1)
        # 组内任一列变化才保留该行
        #mask = block_df[cols].diff().fillna(1).ne(0).any(axis=1).values
        all_zero_mask = (block_df[cols].abs().sum(axis=1) == 0)
        if ignore_all_zero:
            mask = diff_mask & (~all_zero_mask)
        else:
            mask = diff_mask
        reduced = block_df.loc[mask]
        t_orig = reduced[time_col].to_numpy(float)
        out_list = []
        for col in cols:
            v_orig = reduced[col].to_numpy(float)
            if col in unwrap_set:
                v_orig = np.unwrap(np.deg2rad(v_orig))
                v_interp = _interp_1d_with_gaps(
                    t_target, t_orig, v_orig,
                    method=interp_method,
                    max_gap_s=max_gap_seconds,
                    allow_extrapolate=allow_extrapolate
                )
                v_interp = np.rad2deg(v_interp)
            else:
                v_interp = _interp_1d_with_gaps(
                    t_target, t_orig, v_orig,
                    method=interp_method,
                    max_gap_s=max_gap_seconds,
                    allow_extrapolate=allow_extrapolate
                )
            out_list.append(v_interp)
        return out_list  # list of 1D arrays (len=t_target)

    # === 1) 关节角：单列去重 + 插值 ===
    for col in joint_cols:
        series = df[[time_col, col]].copy().reset_index(drop=True)
        mask = series[col].diff().fillna(1).ne(0).values
        reduced = series.iloc[mask]
        t_orig = reduced[time_col].to_numpy(float)
        v_orig = reduced[col].to_numpy(float)

        if col in unwrap_set:
            v_orig = np.unwrap(np.deg2rad(v_orig))
            v_interp = _interp_1d_with_gaps(
                t_target, t_orig, v_orig,
                method=interp_method, max_gap_s=max_gap_seconds, allow_extrapolate=allow_extrapolate
            )
            v_interp = np.rad2deg(v_interp)
        else:
            v_interp = _interp_1d_with_gaps(
                t_target, t_orig, v_orig,
                method=interp_method, max_gap_s=max_gap_seconds, allow_extrapolate=allow_extrapolate
            )
        result_columns.append(v_interp)

    # === 2) IMU：每9列整体去重 + 插值 ===
    imu_len = 9
    if len(imu_cols) % imu_len != 0:
        # 容错：若不是9的倍数，尽量按整块处理，多余的尾巴单列处理
        num_imus = len(imu_cols) // imu_len
        tail = imu_cols[num_imus * imu_len:]
        imu_groups = [imu_cols[i*imu_len:(i+1)*imu_len] for i in range(num_imus)]
        if tail:
            imu_groups.append(tail)
    else:
        imu_groups = [imu_cols[i*imu_len:(i+1)*imu_len] for i in range(len(imu_cols)//imu_len)]

    for cols in imu_groups:
        out_list = _interp_block(cols)
        result_columns.extend(out_list)

    # === 3) 足底压力：若存在，则每18列整体去重 + 插值（左右各一组） ===
    plantar_order = []
    if has_left_plantar:
        L_out = _interp_block(plantar_left_cols,ignore_all_zero=True)
        result_columns.extend(L_out)
        plantar_order += plantar_left_cols
    if has_right_plantar:
        R_out = _interp_block(plantar_right_cols,ignore_all_zero=True)
        result_columns.extend(R_out)
        plantar_order += plantar_right_cols

    # === 拼装 ===
    processed_data = np.column_stack(result_columns)

    # 输出列名顺序：time | 7关节 | IMU全部 | (可选)L18 | (可选)R18
    processed_headers = [time_col] + joint_cols + imu_cols + plantar_order

    return processed_data, processed_headers

def _interp_1d_with_gaps(t_target: np.ndarray,
                         t_orig: np.ndarray,
                         v_orig: np.ndarray,
                         method: str = "pchip",
                         max_gap_s: float = 1,
                         allow_extrapolate: bool = False) -> np.ndarray:
    """
    在 t_target 上对 (t_orig, v_orig) 插值：
      - 仅对“间隔 <= max_gap_s”的相邻观测区间插值，超过的“大片空洞”置 NaN
      - 不外推（端点外也置 NaN），除非 allow_extrapolate=True
      - method: "linear" | "pchip" | "cubic" | "akima" | "nearest"
    """
    y = np.full_like(t_target, np.nan, dtype=float)

    # 清理无效点 & 时间不单调/重复
    ok = np.isfinite(t_orig) & np.isfinite(v_orig)
    t_o = t_orig[ok]; v_o = v_orig[ok]
    if t_o.size == 0:
        return y
    # 去重 & 保持单调
    idx = np.argsort(t_o)
    t_o = t_o[idx]; v_o = v_o[idx]
    # 压掉重复时间戳，保留第一个
    uniq, first_idx = np.unique(t_o, return_index=True)
    t_o = t_o[first_idx]; v_o = v_o[first_idx]

    if t_o.size == 1:
        # 只有一个点：仅在该点处赋值（或者全 NaN）
        # 这里给出“常值”的替代策略：仍然只在 t_o 覆盖范围内赋值
        if allow_extrapolate:
            y[:] = v_o[0]
        else:
            mask = (t_target >= t_o[0]) & (t_target <= t_o[0])
            y[mask] = v_o[0]
        return y

    # 选择插值器
    if method == "linear":
        # 用 numpy.interp，外侧填 NaN
        # 先常规插值
        vals = np.interp(t_target, t_o, v_o, left=np.nan, right=np.nan)
    elif method == "nearest":
        from scipy.interpolate import interp1d
        f = interp1d(t_o, v_o, kind="nearest",
                     bounds_error=False,
                     fill_value=(np.nan, np.nan) if not allow_extrapolate else "extrapolate",
                     assume_sorted=True)
        vals = f(t_target)
    elif method == "pchip":
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(t_o, v_o, extrapolate=allow_extrapolate)
        vals = f(t_target)
        if not allow_extrapolate:
            vals[(t_target < t_o[0]) | (t_target > t_o[-1])] = np.nan
    elif method == "cubic":
        from scipy.interpolate import CubicSpline
        f = CubicSpline(t_o, v_o, bc_type="not-a-knot", extrapolate=allow_extrapolate)
        vals = f(t_target)
        if not allow_extrapolate:
            vals[(t_target < t_o[0]) | (t_target > t_o[-1])] = np.nan
    elif method == "akima":
        from scipy.interpolate import Akima1DInterpolator
        f = Akima1DInterpolator(t_o, v_o)
        vals = f(t_target)
        if not allow_extrapolate:
            vals[(t_target < t_o[0]) | (t_target > t_o[-1])] = np.nan
    else:
        raise ValueError(f"Unknown interp method: {method}")

    # —— 大缺口置 NaN（核心）——
    # 对于每个 t_target，找到它所在的“原始相邻观测点”间隔 [t_left, t_right]
    # 若 (t_right - t_left) > max_gap_s，则该区间的 t_target 全部置 NaN
    # 实现：用 searchsorted 找右侧索引，再回退左侧索引
    # right_idx = np.searchsorted(t_o, t_target, side="right")
    # left_idx = right_idx - 1
    # valid_pair = (left_idx >= 0) & (right_idx < t_o.size)
    # gap = np.full_like(t_target, np.inf, dtype=float)
    # gap[valid_pair] = t_o[right_idx[valid_pair]] - t_o[left_idx[valid_pair]]
    # vals[gap > 1.0] = np.nan

    return vals.astype(float)
    
if __name__ == "__main__":
    #w
    csv_path="E:/demo/python/Gait1016/datasets/rawZJH/gait09.csv"
    #df=pd.read_csv(csv_path,encoding='utf-8-sig')
    #csv配置
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
    thigh_length=55
    shank_length=44
    origin_width=30
    rawdata, rawheaders = read_csv_file(csv_path)
    #重复数据处理，并插值
    data,headers=process_imu_joint_and_plantar_safe_ndarray(rawdata,rawheaders,interp_method="cubic")
    #建立保存文件夹
    result_dir=create_result_folder(csv_path)
    processed_data,processed_headers=process_imu_joint_and_plantar_safe_ndarray(data,headers,cfg.fs)
    pre00_path = os.path.join(result_dir, os.path.basename(csv_path).replace('.csv', '_pre00.csv'))
    save_csv_file(data,headers,pre00_path)
    #保存插值文件
    raw_df=pd.read_csv(pre00_path,encoding="utf-8-sig")
    #计算步态参数
    gait_paparemters=calculate_gait_parameters(raw_df,cfg,thigh_length,shank_length,origin_width,plot=False)
