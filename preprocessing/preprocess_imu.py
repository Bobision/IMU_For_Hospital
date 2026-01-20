from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Sequence, Tuple, Dict, List, Literal, Union
import numpy as np
import pandas as pd
#from Postoper_Data_4 import read_csv_file,process_imu_joint_safe_ndarray
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import welch
from matplotlib import rcParams
from typing import List, Dict, Tuple, Optional
import pywt
# ----------------------------
# === 全局字体设置 ===
rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 顺序重要，中文自动回退到宋体
rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题
# 可选：调整体默认字号
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['legend.fontsize'] = 11
# ----------------------------
# 配置
# ----------------------------
@dataclass
class PreprocessConfig:
    # 基础
    fs: float = 50.0                        # 原始采样率(Hz)，用于滤波/重采样
    time_col: Optional[str] = None          # 时间列名；None=用行号
    # 选择列：二选一（手动列名/自动前缀）
    channels: Optional[List[str]] = None    # 显式指定参与处理的列（顺序即输出顺序）
    channels_prefix: Optional[str] = None   # 例如 "imu"；与上互斥，优先用 channels

    # 列分组（角度/陀螺/加速度）可选：留 None 则自动按列名关键字识别
    angle_cols: Optional[List[str]] = None
    gyro_cols: Optional[List[str]] = None
    acc_cols: Optional[List[str]] = None

    # 步骤开关
    drop_initial_seconds: float = 0.0       # >0 则丢弃最前面的秒数（常见起步干扰）
    baseline_frames: int = 20               # 去基线：前N帧的均值
    interpolate_nan: bool = True            # 是否做线性插值填NaN
    resample_hz: Optional[float] = None     # 目标重采样频率；None=不重采样
    unwrap_angle: bool = False              # 角度去包裹
    use_angle_sincos: bool = False          # 角度列展开为 sin/cos
    # 滤波参数（按分组）
    ang_low_hz: float = 6.0
    gyr_low_hz: float = 15.0
    acc_high_hz: float = 0.3                # 先高通去漂移
    acc_low_hz: float = 20.0                # 再低通平滑
    butter_order_lp: int = 4
    butter_order_hp: int = 2

    # 平滑/去噪
    moving_median_k: int = 0                # 0=关闭；3/5/7等奇数启用
    moving_mean_k: int = 0                  # 0=关闭

    # ZUPT（简化版；对足部 IMU 的 acc/gyro 做准静止检测）
    enable_zupt: bool = False
    zupt_gyro_thr_dps: float = 5.0          # 陀螺阈值(°/s)
    zupt_acc_thr_ms2: float = 0.25          # 加速度阈值(m/s^2)，需确保输入单位一致
    zupt_min_len: int = 5                   # 最小静止段长度(帧)
    zupt_zero_drift: bool = True            # 在静止段内将角速度/加速度拉到0
    zupt_target_prefixes: Tuple[str, ...] = ("foot","imu7","imu4")  # 哪些前缀的IMU做ZUPT
    # 特征工程（可选）
    add_magnitude_for: Tuple[str, ...] = ("acc", "gyro")  # 为这些组添加模长列
    add_rms_k: int = 0                     # >0 添加滑动RMS

    # 标准化/截断
    normalize: Literal["none", "zscore"] = "zscore"
    stats_path: Optional[str] = None       # mu/sigma npz 路径；None=不读写
    clip_after_std: Optional[float] = 5.0  # z-score 后截断；None=不截断

    # 可视化
    plot: bool = False
    plot_channels: Optional[List[str]] = None   # 指定对比的列名，None=前3列
    plot_outdir: str = "preproc_plots"
# ----------------------------
# 工具：列筛选/分组
# ----------------------------
def pick_channels(df: pd.DataFrame, cfg: PreprocessConfig) -> List[str]:
    if cfg.channels:
        missing = [c for c in cfg.channels if c not in df.columns]
        if missing:
            raise KeyError(f"显式指定的列缺失：{missing}")
        return cfg.channels
    if cfg.channels_prefix:
        cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith(cfg.channels_prefix.lower())]
        if not cols:
            raise KeyError(f"未找到以 '{cfg.channels_prefix}' 开头的列")
        return cols
    # 默认：所有数值列
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def infer_groups_by_name(cols: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    ang, gyr, acc = [], [], []
    for c in cols: 
        cl = c.lower()
        if "ang" in cl or "angle" in cl:
            ang.append(c)
        elif "gyro" in cl or "vel" in cl:
            gyr.append(c)
        elif "acc" in cl or "acceler" in cl or "acl" in cl:
            acc.append(c)
    return ang, gyr, acc
# ----------------------------
# 滤波核
# ----------------------------
def _butter_lowpass(cut_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    from scipy.signal import butter
    b, a = butter(order, cut_hz / nyq, btype="low")
    return b, a

def _butter_highpass(cut_hz: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    from scipy.signal import butter
    b, a = butter(order, cut_hz / nyq, btype="high")
    return b, a
# ----------------------------
# Hilbert 包络
def apply_hilbert_envelope(df: pd.DataFrame,
                           columns: list[str],
                           plot: bool = False,
                           plot_outdir: str = "hilbert_plots") -> pd.DataFrame:
    """
    对 DataFrame 指定列计算 Hilbert 包络，并返回添加新列后的 DataFrame。

    参数：
    ----------
    df : pd.DataFrame
        输入的 DataFrame。
    columns : list[str]
        需要计算包络的列名列表。
    plot : bool, optional
        是否绘制原始信号与包络的对比图。默认 False。
    plot_outdir : str, optional
        若 plot=True，图像保存路径。默认 "hilbert_plots"。

    返回：
    ----------
    df_out : pd.DataFrame
        原始 DataFrame 的副本，增加了 “列名_env” 列（对应 Hilbert 包络）。
    """

    df_out = df.copy()
    os.makedirs(plot_outdir, exist_ok=True)

    for col in columns:
        if col not in df.columns:
            print(f"[Warning] 列 {col} 不存在，已跳过。")
            continue

        signal = df[col].to_numpy(dtype=float)
        # 1️⃣ 计算希尔伯特包络
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)

        # 2️⃣ 添加新列
        new_col = f"{col}_env"
        df_out[new_col] = envelope

        # 3️⃣ 可选绘图
        if plot:
            plt.figure(figsize=(10, 3))
            plt.plot(signal, label=f"{col} 原始信号", linewidth=1.0, alpha=0.8)
            plt.plot(envelope, label=f"{col} 包络", linewidth=1.5)
            plt.title(f"Hilbert Envelope - {col}")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(plot_outdir, f"{col}_envelope.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

    return df_out
    """
    对输入信号计算 Hilbert 包络
    输入:
        signal: 1D numpy 数组
    输出:
        envelope: 1D numpy 数组，对应瞬时幅值
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope
# ----------------------------
# 频域分析（Welch PSD）
def apply_frequency_analysis(df: pd.DataFrame,
                             columns: list[str],
                             plot: bool = False,
                             plot_outdir: str = "freq_plots",
                             fs=50, window_seconds=2.0, nfft_factor=1.0):
    """
    对指定列做频域分析（Welch PSD）
    入口签名与 apply_hilbert_envelope 一致：
      df: 预处理后的 DataFrame
      columns: 需要分析的列名列表
      plot: 是否绘图保存
      plot_outdir: 图像保存目录

    返回:
      psd_df: 行索引为 'freq_hz' 的 PSD DataFrame（列名形如 '<col>_psd'）
      peaks_df: 每列的峰值频率与峰值（DataFrame: ['signal', 'peak_freq_hz', 'peak_psd']）
    """
    os.makedirs(plot_outdir, exist_ok=True) if plot else None

    # ---------- 2) 逐列计算 Welch PSD ----------
    psd_cols = {}
    peak_rows = []

    # 统一的段长：尽量用 ~2秒，且不超过长度
    N = len(df)
    nperseg_target = int(max(32, min(N, round(fs * window_seconds))))  # ~2s，至少32点
    nfft = int(nperseg_target * nfft_factor)
    noverlap = nperseg_target // 2

    # 先预构造频率向量（用第一列成功计算的结果为准）
    f_ref = None

    for col in columns:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        # 轻量插值补小缺口，首尾不外推
        if x.isna().any():
            x = x.interpolate("linear", limit_direction="both", limit=10)

        x = x.to_numpy(dtype=float)
        # NaN 仍残留的话置零（不理想但能保证 PSD 能算）
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)

        nseg = min(nperseg_target, len(x))
        if nseg < 32:
            # 序列太短，跳过
            continue

        f, Pxx = welch(x,
                       fs=fs,
                       nperseg=nseg,
                       nfft=nfft,
                       noverlap=min(noverlap, nseg//2),
                       window="hann",
                       detrend="constant",
                       scaling="density",
                       return_onesided=True)

        if f_ref is None:
            f_ref = f

        psd_cols[f"{col}_psd"] = Pxx

        # 峰值（剔除 DC 后再找峰更实用，这里从 >0Hz 开始）
        valid = f > 0
        if np.any(valid):
            idx = np.argmax(Pxx[valid])
            peak_freq = float(f[valid][idx])
            peak_val  = float(Pxx[valid][idx])
        else:
            peak_freq, peak_val = float("nan"), float("nan")

        peak_rows.append({"signal": col, "peak_freq_hz": peak_freq, "peak_psd": peak_val})

        # 可选绘图
        if plot:
            plt.figure(figsize=(9, 3))
            plt.semilogy(f, Pxx)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.title(f"Welch PSD - {col}  (fs≈{fs:.2f} Hz)")
            # 标注峰值
            if np.isfinite(peak_freq):
                plt.axvline(peak_freq, linestyle="--", linewidth=1)
                plt.text(peak_freq, max(Pxx)*0.6, f"{peak_freq:.2f} Hz", rotation=90, va="center")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"{col}_psd.png"), dpi=300)
            plt.close()

    if f_ref is None:
        # 没有任何列成功计算
        return pd.DataFrame(), pd.DataFrame(columns=["signal","peak_freq_hz","peak_psd"])

    # ---------- 3) 组装返回 DataFrame ----------
    psd_df = pd.DataFrame(psd_cols, index=pd.Index(f_ref, name="freq_hz"))
    peaks_df = pd.DataFrame(peak_rows, columns=["signal","peak_freq_hz","peak_psd"])

    return psd_df, peaks_df
# ----------------------------
# 自相关函数 ACF
def apply_autocorrelation(df: pd.DataFrame,
                          columns: list[str],
                          plot: bool = True,
                          plot_outdir: str = "./eval_viz/freq_plots",
                          fs=50,
                          max_lag_seconds: float | None = None,
                          max_lag: int | None = None,
                          fill_limit: int = 10,
                          detrend: bool = True,
                          unbiased: bool = True):
    """
    对指定列计算自相关函数 (ACF)，并可选绘图。
    返回:
      acf_df   : 行索引为 'lag' (样本) 的 DataFrame，各列为 '<signal>_acf'
      peaks_df : 每列的首个正峰(排除滞后0)信息: ['signal','peak_lag','peak_lag_seconds','peak_value']

    参数:
      columns          : 待分析的列名列表
      plot             : 是否保存自相关图
      plot_outdir      : 图像输出目录
      max_lag_seconds  : 限制最大滞后(秒)。若给出，会覆盖 max_lag
      max_lag          : 限制最大滞后(样本)。两者都 None 时默认到 N-1
      fill_limit       : 线性插值允许连续 NaN 的最大长度(样本)，超过保留 NaN→置0
      detrend          : 是否去均值（推荐 True）
      unbiased         : ACF 归一：unbiased 用 (N-lag) 归一；否则用 N 归一
    """
    if plot:
        os.makedirs(plot_outdir, exist_ok=True)

    # ---------- ACF 计算（FFT法，O(N log N)） ----------
    def _acf_fft(x: np.ndarray, maxlag: int, unbiased_norm: bool = True) -> np.ndarray:
        n = len(x)
        if detrend:
            # 去均值（ACF 必须，否则 0 滞后占主导）
            x = x - np.nanmean(x)
        # NaN 处理：短缺口内线性补，残留置0（等价用掩码修正也可）
        if np.isnan(x).any():
            s = pd.Series(x)
            x = s.interpolate("linear", limit_direction="both", limit=fill_limit).to_numpy()
            x = np.nan_to_num(x, nan=0.0)
        # FFT-based 自相关（相当于 irfft(|X|^2)）
        nfft = 1 << ((2*n - 1).bit_length())
       #nfft=n
        Xf = np.fft.rfft(x, n=nfft)
        acf_full = np.fft.irfft(Xf * np.conjugate(Xf), n=nfft)[:n]  # lag >= 0
        lags = np.arange(n, dtype=int)
        if unbiased_norm:
            denom = (n - lags).astype(float)
        else:
            denom = float(n)
        acf = acf_full / denom
        # 归一到 ACF[0] = 1
        if acf[0] != 0:
            acf = acf / acf[0]
        return acf[:maxlag+1]

    # ---------- 组装输出 ----------
    acf_cols = {}
    peak_rows = []

    # 统一 max_lag
    N = len(df)
    if max_lag_seconds is not None:
        max_lag = int(np.clip(round(max_lag_seconds * fs), 1, N-1))
    if max_lag is None:
        max_lag = min(N-1, int(fs * 10))  # 默认最多看 10 秒或 N-1

    # 遍历列
    for col in columns:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if len(x) < 3:
            continue

        acf = _acf_fft(x, maxlag=max_lag, unbiased_norm=unbiased)
        acf_cols[f"{col}_acf"] = acf

        # ---- 寻找首个正峰（排除 lag=0） ----
        # 简单峰值检测：上升→下降的拐点；也可用 scipy.signal.find_peaks
        peak_lag = np.nan
        peak_val = np.nan
        if len(acf) > 2:
            # 从 lag=1 开始找局部极大
            a = acf
            # 一阶差分
            da = np.diff(a)
            # 峰条件：da 前正后负
            for k in range(1, len(a)-1):
                if da[k-1] > 0 and da[k] < 0 and a[k] > 0:
                    peak_lag = int(k)
                    peak_val = float(a[k])
                    break

        peak_rows.append({
            "signal": col,
            "peak_lag": int(peak_lag) if np.isfinite(peak_lag) else np.nan,
            "peak_lag_seconds": (float(peak_lag)/fs) if np.isfinite(peak_lag) else np.nan,
            "peak_value": float(peak_val) if np.isfinite(peak_val) else np.nan
        })

        # ---- 可选绘图 ----
        if plot:
            lags = np.arange(len(acf))
            t_lag = lags / fs
            plt.figure(figsize=(9, 3))
            plt.plot(t_lag, acf, linewidth=1.5)
            plt.axhline(0, color="k", linewidth=0.8, alpha=0.4)
            plt.xlabel("Lag (s)")
            plt.ylabel("Autocorrelation")
            title = f"ACF - {col}  (fs≈{fs:.2f} Hz, max_lag={max_lag} samp≈{max_lag/fs:.2f}s)"
            plt.title(title)
            # 标注首峰
            if np.isfinite(peak_lag):
                plt.axvline(peak_lag/fs, linestyle="--", linewidth=1)
                plt.text(peak_lag/fs, 0.8, f"peak≈{peak_lag/fs:.2f}s", rotation=0, va="center")
            plt.tight_layout()
            fname = os.path.join(plot_outdir, f"{col}_acf.png")
            plt.savefig(fname, dpi=300)
            plt.close()

    # 组装 DataFrame
    if len(acf_cols) == 0:
        acf_df = pd.DataFrame()
        peaks_df = pd.DataFrame(columns=["signal","peak_lag","peak_lag_seconds","peak_value"])
        return acf_df, peaks_df

    acf_len = len(next(iter(acf_cols.values())))
    acf_df = pd.DataFrame(acf_cols, index=pd.Index(np.arange(acf_len), name="lag"))
    peaks_df = pd.DataFrame(peak_rows, columns=["signal","peak_lag","peak_lag_seconds","peak_value"])

    return acf_df, peaks_df
# ----------------------------
# 小波分解特征,返回重构时间序列和频带说明
def apply_wavelet_select(
    df: pd.DataFrame,
    columns: list[str],
    fs: float,                            # 采样率（Hz），不自动识别
    wavelet: str = "sym4",                # 小波基：如 'db4','sym4','coif3','bior3.5' 等
    max_level: int = 5,                   # 最大分解层数（会按数据长度自动截断）
    part: str = "detail",                 # 'detail' 选择 Dj；'approx' 选择 Aj
    select_level: int = 4,                # 要提取的层号（1..max_level）
    mode: str = "symmetric",              # 边界处理：'symmetric'/'periodization' 等
    interpolate_nan_limit: int = 10,      # 连续≤N个NaN线性补；更长保持NaN并按0处理
    plot: bool = True,
    plot_outdir: str = "./eval_viz/wavelet_plots"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对指定列做离散小波分解（DWT），选择某一层的细节(Dj)或近似(Aj)进行重构。
    返回：
      df_out  : 重构后的信号（与输入长度一致），列名加后缀
      bands_df: 每列对应子带的频率范围说明（low_hz, high_hz）
    """
    if plot:
        os.makedirs(plot_outdir, exist_ok=True)

    df_out_cols = {}
    band_rows = []

    # 小工具：根据层号给出理论频带（近似）
    # DWT（二倍抽取）下：
    # - Dj 覆盖 (fs/2^(j+1), fs/2^j)
    # - Aj 覆盖 (0, fs/2^(j+1))
    def band_of(level: int, kind: str) -> tuple[float, float]:
        if kind == "detail":
            low = fs / (2 ** (level + 1))
            high = fs / (2 ** level)
        else:  # 'approx'
            low = 0.0
            high = fs / (2 ** (level + 1))
        return float(low), float(high)

    # 逐列处理
    for col in columns:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        # 轻量 NaN 处理：短缺口线性补，残留用0
        if np.isnan(x).any():
            s = pd.Series(x)
            s = s.interpolate("linear", limit_direction="both", limit=interpolate_nan_limit)
            x = np.nan_to_num(s.to_numpy(dtype=float), nan=0.0)

        # 依据数据长度和滤波器长度限制层数
        w = pywt.Wavelet(wavelet)
        max_possible = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
        L = max(1, min(max_level, max_possible))
        if not (1 <= select_level <= L):
            # 自动夹到合法范围（也可 raise）
            select_level_eff = min(max(select_level, 1), L)
        else:
            select_level_eff = select_level

        # DWT 分解
        coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode, level=L)
        # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]

        # 构造“只保留目标层”的系数列表
        keep = []
        if part == "detail":
            # 细节 Dj（注意在 coeffs 列表中的索引位置）
            for i, c in enumerate(coeffs):
                if i == 0:
                    # cA_L 置零
                    keep.append(np.zeros_like(c))
                else:
                    # i=1 对应 cD_L, i=2->cD_{L-1}, ..., i=L->cD_1
                    level_here = L - (i - 1)
                    if level_here == select_level_eff:
                        keep.append(c)
                    else:
                        keep.append(np.zeros_like(c))
        else:
            # 近似 Aj：仅在 cA_L 层能直接拿到；若要 Aj( j<select_level )，需降低分解层数再分解
            # 这里按照“取 A_select_level”的语义，重做一次分解到 select_level_eff：
            if select_level_eff != L:
                coeffs2 = pywt.wavedec(x, wavelet=wavelet, mode=mode, level=select_level_eff)
            else:
                coeffs2 = coeffs
            keep = [coeffs2[0]] + [np.zeros_like(c) for c in coeffs2[1:]]
            # 用匹配长度的系数集重构
            x_rec = pywt.waverec(keep, wavelet=wavelet, mode=mode)
            # 对齐长度
            x_rec = x_rec[:len(x)]
            low_hz, high_hz = band_of(select_level_eff, "approx")
            suffix = f"{col}_A{select_level_eff}_{wavelet}"
            df_out_cols[suffix] = x_rec
            band_rows.append({
                "signal": col,
                "select": f"A{select_level_eff}",
                "wavelet": wavelet,
                "low_hz": low_hz,
                "high_hz": high_hz
            })
            # 绘图
            if plot:
                t = np.arange(len(x)) / fs
                plt.figure(figsize=(10, 3))
                plt.plot(t, x, label="orig", linewidth=1.0, alpha=0.8)
                plt.plot(t, x_rec, label=f"A{select_level_eff}", linewidth=1.1)
                plt.title(f"{col}  |  {wavelet}  A{select_level_eff}  "
                          f"[{low_hz:.3f}, {high_hz:.3f}] Hz")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_outdir, f"{col}_A{select_level_eff}_{wavelet}.png"), dpi=300)
                plt.close()
            # 近似分支已完成，跳到下一列
            continue

        # 细节分支：重构
        x_rec = pywt.waverec(keep, wavelet=wavelet, mode=mode)
        x_rec = x_rec[:len(x)]  # 对齐长度
        low_hz, high_hz = band_of(select_level_eff, "detail")
        suffix = f"{col}_D{select_level_eff}_{wavelet}"
        df_out_cols[suffix] = x_rec
        #df_out_cols[suffix] = keep # 直接用细节系数作为特征
        band_rows.append({
            "signal": col,
            "select": f"D{select_level_eff}",
            "wavelet": wavelet,
            "low_hz": low_hz,
            "high_hz": high_hz
        })

        # 绘图
        if plot:
            t = np.arange(len(x)) / fs
            plt.figure(figsize=(10, 3))
            plt.plot(t, x, label="orig", linewidth=1.0, alpha=0.8)
            plt.plot(t, x_rec, label=f"D{select_level_eff}", linewidth=1.1)
            plt.title(f"{col}  |  {wavelet}  D{select_level_eff}  "
                      f"[{low_hz:.3f}, {high_hz:.3f}] Hz")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"{col}_D{select_level_eff}_{wavelet}.png"), dpi=300)
            plt.close()

    # 组装输出
    df_out = pd.DataFrame(df_out_cols, index=df.index)
    bands_df = pd.DataFrame(band_rows, columns=["signal","select","wavelet","low_hz","high_hz"])
    return df_out, bands_df
#- ---------------------------
# 小波分解特征，只提取系数
def extract_wavelet_coeffs(
    df: pd.DataFrame,
    columns: List[str],
    fs: float,                             # 采样率（Hz），此函数不自动识别
    wavelet: str = "sym4",                 # 如 'db4','sym4','coif3','bior3.5'...
    max_level: int = 5,                    # 最大分解层数（会按长度与滤波器自动截断）
    part: str = "detail",                  # 'detail' 导出 cD_j；'approx' 导出 cA_j
    select_level: int = 4,                 # 目标层 j（1..max_level）
    mode: str = "symmetric", 
    transform_type: str = "dwt",               # 边界延拓：'symmetric'/'reflect'/'periodization'...
    interpolate_nan_limit: int = 10,       # 连续≤N个NaN线性补；更长保NaN后置0
    plot: bool = True,
    plot_outdir: str = "./eval_viz/wavelet_coeffs"
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    对指定列进行 DWT 分解，只导出目标层的小波系数（而非重构信号）。
    返回:
      coeffs_by_col : dict[col -> DataFrame]，索引为 time_s，列名形如 '{col}_cD4_sym4' 或 '{col}_cA5_sym4'
      bands_df      : 每个列/层的频带说明与系数采样信息（low_hz, high_hz, dt_eff_s, length）
    """
    if plot:
        os.makedirs(plot_outdir, exist_ok=True)

    coeffs_by_col: Dict[str, pd.DataFrame] = {}
    band_rows = []

    def band_of(level: int, kind: str):
        if kind == "detail":
            return fs / (2 ** (level + 1)), fs / (2 ** level)
        else:
            return 0.0, fs / (2 ** (level + 1))

    for col in columns:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        # ==== NaN 线性补全 ====
        if np.isnan(x).any():
            s = pd.Series(x).interpolate("linear", limit_direction="both", limit=interpolate_nan_limit)
            x = np.nan_to_num(s.to_numpy(float), nan=0.0)

        # ==== 限制层数 ====
        w = pywt.Wavelet(wavelet)
        Lmax = pywt.dwt_max_level(len(x), w.dec_len)
        L = max(1, min(max_level, Lmax))
        j = int(np.clip(select_level, 1, L))

        # ==== DWT 模式 ====
        if transform_type.lower() == "dwt":
            coeffs = pywt.wavedec(x, wavelet=wavelet, level=L, mode=mode)
            if part == "detail":
                idx = 1 + (L - j)
                c = coeffs[idx]
            else:
                if j != L:
                    coeffs_j = pywt.wavedec(x, wavelet=wavelet, level=j, mode=mode)
                    c = coeffs_j[0]
                else:
                    c = coeffs[0]
            dt_eff = (2 ** j) / fs
            t_coeff = np.arange(len(c)) * dt_eff
            # 限制时间范围不超过原信号
            t_coeff = t_coeff[t_coeff <= (len(x) - 1) / fs]
            c = c[:len(t_coeff)]

        # ==== SWT 模式 ====
        elif transform_type.lower() == "swt":
            x = pad_to_swt_length(x, L)
            if len(x) % 2 != 0:
                x = np.append(x, x[-1])
            coeffs_swt = pywt.swt(x, wavelet=wavelet, level=L)
            cA_j, cD_j= coeffs_swt[j - 1]
            c = cD_j if part == "detail" else cA_j
            dt_eff = 1 / fs
            t_coeff = np.arange(len(c)) / fs

        else:
            raise ValueError("transform_type 必须是 'dwt' 或 'swt'")

        low_hz, high_hz = band_of(j, part)
        coeff_name = f"{col}_c{'D' if part=='detail' else 'A'}{j}_{wavelet}_{transform_type}"
        coeff_df = pd.DataFrame({coeff_name: c}, index=pd.Index(t_coeff, name="time_s"))
        coeffs_by_col[col] = coeff_df

        band_rows.append({
            "signal": col, "coeff": coeff_name, "wavelet": wavelet,
            "mode": mode, "transform": transform_type,
            "level": j, "part": part,
            "low_hz": low_hz, "high_hz": high_hz,
            "dt_eff_s": dt_eff, "length": len(c)
        })

        # ==== 绘图 ====
        if plot:
            fig, ax1 = plt.subplots(figsize=(10, 3))
            color1, color2 = "tab:red", "tab:blue"

            t_orig = np.arange(len(x)) / fs
            ax1.plot(t_orig, x, color=color1, lw=0.8, alpha=0.6, label="orig (z-norm)")
            ax2 = ax1.twinx()
            ax2.plot(t_coeff, c, color=color2, lw=1.2, label=coeff_name)

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Original", color=color1)
            ax2.set_ylabel("Coeff", color=color2)
            ax1.tick_params(axis="y", labelcolor=color1)
            ax2.tick_params(axis="y", labelcolor=color2)
            plt.title(f"{coeff_name} | band≈[{low_hz:.2f}, {high_hz:.2f}] Hz | {transform_type.upper()}")
            fig.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"{coeff_name}.png"), dpi=300)
            plt.close()

    bands_df = pd.DataFrame(band_rows)
    return coeffs_by_col, bands_df
#调用函数
def pad_to_swt_length(x: np.ndarray, level: int) -> np.ndarray:
    """把长度补到可做指定 level 的 SWT：len(x_pad) % 2**level == 0"""
    mod = len(x) % (2 ** level)
    if mod == 0:
        return x
    need = (2 ** level) - mod
    return np.pad(x, (0, need), mode="edge")
# ----------------------------
# EMD 分解
def apply_emd(
    df: pd.DataFrame,
    columns: List[str],
    fs: float,                                 # 采样率（Hz）
    max_imfs: Optional[int] = None,            # 最多提取多少个 IMF（None 表示由算法自行决定）
    spline_kind: str = "cubic",                # EMD 内部包络拟合样条
    nbsym: int = 2,                            # 边界镜像点数（对边界稳健性有用）
    sift_max_iter: int = 1000,                 # sifting 最大迭代
    sift_fix_h: int = 0,                       # FIXE_H（0=禁用；>0 表示迭代固定次数）
    interpolate_nan_limit: int = 10,           # 连续≤N个NaN线性补；更长保NaN并用0替换
    plot: bool = True,
    plot_outdir: str = "./eval_viz/emd"
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    对指定列做 EMD 分解。返回:
      - imfs_by_col: {col -> DataFrame(time_s, IMF_1..IMF_K, residue)}
      - summary_df : 每列 IMF 个数、能量占比等信息
    说明：
      - IMF_1 为最高频 IMF，编号递增频率降低；residue 为最终残差
      - index 为时间（秒），方便与原始信号对齐
    依赖：
      pip install EMD-signal
    """
    try:
        from PyEMD import EMD
    except Exception as e:
        raise ImportError("需要安装 PyEMD（包名：EMD-signal）。请运行：pip install EMD-signal") from e

    if plot:
        os.makedirs(plot_outdir, exist_ok=True)

    imfs_by_col: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    for col in columns:
        if col not in df.columns:
            continue

        # ---- 取信号并做轻量 NaN 处理 ----
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(x).any():
            s = pd.Series(x).interpolate("linear", limit_direction="both", limit=interpolate_nan_limit)
            x = np.nan_to_num(s.to_numpy(float), nan=0.0)

        N = len(x)
        if N < 8:
            # 太短不适合 EMD，跳过
            continue

        t = np.arange(N, dtype=float) / fs

        # ---- 构建 EMD 实例与参数 ----
        # imf_options 里的键使用 PyEMD 约定
        imf_options = {
            "MAX_ITERATIONS": int(sift_max_iter),
            "FIXE_H": int(sift_fix_h)  # 0 表示根据停止准则收敛；>0 则固定迭代次数
        }
        emd = EMD(spline_kind=spline_kind, nbsym=nbsym, imf_options=imf_options)

        # PyEMD: max_imf = -1 表示不强制限制；否则为最大 IMF 数
        max_imf_arg = -1 if (max_imfs is None) else int(max_imfs)
        imfs = emd.emd(x, max_imf=max_imf_arg)   # imfs: shape (K, N)
        # 若没有分解出 IMF（极少见），则给空处理
        if imfs is None or len(imfs) == 0:
            K = 0
            residue = x.copy()
            imf_mat = np.empty((0, N), dtype=float)
        else:
            imf_mat = np.asarray(imfs, dtype=float)
            K = imf_mat.shape[0]
            residue = x - imf_mat.sum(axis=0)

        # ---- 组装输出 DataFrame：IMF_1..IMF_K + residue ----
        data_cols = {}
        for k in range(K):
            data_cols[f"IMF_{k+1}"] = imf_mat[k]
        data_cols["residue"] = residue
        df_imf = pd.DataFrame(data_cols, index=pd.Index(t, name="time_s"))
        imfs_by_col[col] = df_imf

        # ---- 计算能量占比摘要（按平方和）----
        total_energy = (x**2).sum() + 1e-12
        parts = {f"IMF_{k+1}": float((imf_mat[k]**2).sum() / total_energy) for k in range(K)}
        parts["residue"] = float((residue**2).sum() / total_energy)
        summary_rows.append({
            "signal": col,
            "N": N,
            "K_imfs": K,
            **parts
        })

        # ---- 可选绘图：原始 + 叠加（或网格）----
        if plot:
            # 叠加视图（原始标准化 & 每个 IMF 标准化，便于对比）
            fig, ax = plt.subplots(figsize=(11, 3.6))
            eps = 1e-12
            x_n = (x - x.mean()) / (x.std() + eps)
            ax.plot(t, x_n, color="k", lw=1.0, alpha=0.8, label=f"{col} (z-norm)")
            offset = 0.0
            for k in range(K):
                y = imf_mat[k]
                y_n = (y - y.mean()) / (y.std() + eps)
                ax.plot(t, y_n + offset, lw=0.9, label=f"IMF_{k+1} (+{offset:.1f})", alpha=0.9)
                offset += 1.5
            ax.set_title(f"EMD of {col} | K={K} | spline={spline_kind}, nbsym={nbsym}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normed amplitude (+offset)")
            ax.legend(ncol=2, fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_outdir, f"{col}_emd_stack.png"), dpi=150)
            plt.close(fig)

            # 网格视图（每个 IMF 单独一行 + residue）
            rows = min(K + 1, 6)  # 防止太多图太长，可自行调整
            fig2, axes = plt.subplots(rows, 1, figsize=(11, 1.8 * rows), sharex=True)
            if rows == 1:
                axes = [axes]
            # 画 IMF_1..IMF_rows-1
            for i in range(rows - 1):
                if i < K:
                    axes[i].plot(t, imf_mat[i], lw=0.9)
                    axes[i].set_ylabel(f"IMF_{i+1}")
                    axes[i].grid(alpha=0.25)
                else:
                    axes[i].axis("off")
            # 画 residue
            axes[-1].plot(t, residue, lw=0.9, color="C3")
            axes[-1].set_ylabel("residue")
            axes[-1].set_xlabel("Time (s)")
            axes[-1].grid(alpha=0.25)
            fig2.suptitle(f"EMD Grid of {col} (first {rows-1} IMFs + residue)")
            fig2.tight_layout(rect=[0, 0, 1, 0.96])
            fig2.savefig(os.path.join(plot_outdir, f"{col}_emd_grid.png"), dpi=150)
            plt.close(fig2)

    summary_df = pd.DataFrame(summary_rows)
    return imfs_by_col, summary_df

def save_stats(mu: np.ndarray, sigma: np.ndarray, path: str, columns: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, mu=mu, sigma=sigma, columns=np.array(columns, dtype=object))

def load_stats(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    d = np.load(path, allow_pickle=True)
    mu, sigma = d["mu"], d["sigma"]
    cols = d["columns"].tolist() if "columns" in d else None
    return mu, sigma, cols

# 主流程（修复版）
# ----------------------------
def preprocess_df(df_raw: pd.DataFrame, cfg: PreprocessConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    输入：原始 DataFrame（由 pd.read_csv 得到）
    输出：处理后的 DataFrame + 日志信息(dict)
    处理顺序（推荐）：
      0) 可选丢弃起始若干秒
      1) 去基线（前N帧均值）
      2) 插值 NaN
      3) 重采样（如需）
      4) 角度去包裹（可选）、角度 sin/cos 展开（可选）
      5) 分组滤波（角/陀/加分别的 cut-off）
      6) 平滑（移动中值/均值）
      7) ZUPT（可选）
      8) 特征工程（模长、RMS）
      9) 标准化/截断
    """
    info: Dict = {"cfg": asdict(cfg)}
    df = df_raw.copy()

    # 时间轴
    if cfg.time_col and cfg.time_col in df.columns:
        t = df[cfg.time_col].to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float) / cfg.fs
    info["len_in"] = len(df)

    # 选择列
    cols = pick_channels(df, cfg)
    info["cols_in"] = cols
    # >>> 新增：如果指定了时间列且在数值列里，剔除它，避免被当作信号去处理
    if cfg.time_col and cfg.time_col in cols:
        cols = [c for c in cols if c != cfg.time_col]
    # 0) 丢弃起始秒
    if cfg.drop_initial_seconds and cfg.drop_initial_seconds > 0:
        keep = t >= (t[0] + cfg.drop_initial_seconds)
        df = df.loc[keep].reset_index(drop=True)
        t = t[keep]
        info["dropped_initial_seconds"] = float(cfg.drop_initial_seconds)

    # 1) 去基线
    N = max(1, min(cfg.baseline_frames, len(df)))
    baseline = df[cols].iloc[:N].mean(axis=0)
    X = df[cols].to_numpy(dtype=np.float32)
    X -= baseline.to_numpy(dtype=np.float32)

    # 2) 插值
    if cfg.interpolate_nan:
        X = pd.DataFrame(X, columns=cols).interpolate(limit_direction="both").to_numpy(dtype=np.float32)

    # 3) 重采样
    fs_eff = cfg.fs
    if cfg.resample_hz and cfg.resample_hz > 0 and cfg.resample_hz != cfg.fs:
        t_new = np.arange(t[0], t[-1] + 1e-9, 1.0 / cfg.resample_hz)
        X_rs = np.empty((t_new.size, X.shape[1]), dtype=np.float32)
        for i in range(X.shape[1]):
            X_rs[:, i] = np.interp(t_new, t, X[:, i])
        X, t = X_rs, t_new
        fs_eff = cfg.resample_hz
        info["resampled_to_hz"] = float(cfg.resample_hz)

    # 分组
    if cfg.angle_cols is None or cfg.gyro_cols is None or cfg.acc_cols is None:
        ang_auto, gyr_auto, acc_auto = infer_groups_by_name(cols)
        angle_cols = ang_auto if cfg.angle_cols is None else cfg.angle_cols
        gyro_cols  = gyr_auto if cfg.gyro_cols  is None else cfg.gyro_cols
        acc_cols   = acc_auto if cfg.acc_cols   is None else cfg.acc_cols
    else:
        angle_cols, gyro_cols, acc_cols = cfg.angle_cols, cfg.gyro_cols, cfg.acc_cols

    name2idx = {c: i for i, c in enumerate(cols)}
    idx_ang = [name2idx[c] for c in angle_cols if c in name2idx]
    idx_gyr = [name2idx[c] for c in gyro_cols  if c in name2idx]
    idx_acc = [name2idx[c] for c in acc_cols   if c in name2idx]

    # 4) 角度处理
    if cfg.unwrap_angle and idx_ang:
        X[:, idx_ang] = np.unwrap(np.deg2rad(X[:, idx_ang]), axis=0) * (180.0/np.pi)
    if cfg.use_angle_sincos and idx_ang:
        keep_idx = [i for i in range(X.shape[1]) if i not in idx_ang]
        theta = np.deg2rad(X[:, idx_ang])
        X = np.concatenate([X[:, keep_idx],
                            np.sin(theta).astype(np.float32),
                            np.cos(theta).astype(np.float32)], axis=1)
        keep_cols = [cols[i] for i in keep_idx]
        sin_cols = [f"{cols[i]}_sin" for i in idx_ang]
        cos_cols = [f"{cols[i]}_cos" for i in idx_ang]
        cols = keep_cols + sin_cols + cos_cols
        name2idx = {c: i for i, c in enumerate(cols)}
        idx_ang = []  # 展开后不再单独按角度分组

    # 5) 分组滤波
    if idx_ang:
        b, a = _butter_lowpass(cfg.ang_low_hz, fs_eff, cfg.butter_order_lp)
        for c in idx_ang: X[:, c] = filtfilt(b, a, X[:, c])
    if idx_gyr:
        b, a = _butter_lowpass(cfg.gyr_low_hz, fs_eff, cfg.butter_order_lp)
        for c in idx_gyr: X[:, c] = filtfilt(b, a, X[:, c])
    if idx_acc:
        bh, ah = _butter_highpass(cfg.acc_high_hz, fs_eff, cfg.butter_order_hp)
        bl, al = _butter_lowpass(cfg.acc_low_hz,  fs_eff, cfg.butter_order_lp)
        for c in idx_acc:
            tmp = filtfilt(bh, ah, X[:, c])
            X[:, c] = filtfilt(bl, al, tmp)

    # 6) 平滑
    if cfg.moving_median_k and cfg.moving_median_k % 2 == 1:
        r = cfg.moving_median_k // 2
        X_med = X.copy()
        for i in range(X.shape[1]):
            for ti in range(X.shape[0]):
                L = max(0, ti - r); R = min(X.shape[0], ti + r + 1)
                X_med[ti, i] = np.median(X[L:R, i])
        X = X_med
    if cfg.moving_mean_k and cfg.moving_mean_k > 1:
        k = cfg.moving_mean_k
        kernel = np.ones(k, dtype=np.float32) / k
        X = np.vstack([np.convolve(X[:, i], kernel, mode="same") for i in range(X.shape[1])]).T

    # 7) ZUPT（修复版：np.ix_ + 守卫）
    if cfg.enable_zupt and (idx_acc or idx_gyr):
        # 目标列：名称中包含足部关键词 & 属于 acc/gyr
        foot_idx = [i for i, c in enumerate(cols) if any(k in c.lower() for k in cfg.zupt_target_prefixes)]
        zupt_cols = sorted(set(foot_idx) & set(idx_acc + idx_gyr))
        if zupt_cols:
            # 简化阈值：逐列都满足“接近零”视为静止
            g_thr = cfg.zupt_gyro_thr_dps
            a_thr = cfg.zupt_acc_thr_ms2
            quiet = np.ones(X.shape[0], dtype=bool)
            for i_col in zupt_cols:
                quiet &= (np.abs(X[:, i_col]) < max(g_thr, a_thr))

            # 最小静止段长度
            if cfg.zupt_min_len > 1 and quiet.any():
                q2 = quiet.copy()
                i = 0; Tlen = len(quiet)
                while i < Tlen:
                    if not quiet[i]:
                        i += 1; continue
                    j = i
                    while j < Tlen and quiet[j]:
                        j += 1
                    if (j - i) < cfg.zupt_min_len:
                        q2[i:j] = False
                    i = j
                quiet = q2

            rows = np.flatnonzero(quiet)
            if rows.size and cfg.zupt_zero_drift:
                # 安全赋值：行列组合索引
                X[np.ix_(rows, zupt_cols)] = 0.0
            # 记录信息
            info["zupt_frames"] = int(rows.size)
            info["zupt_cols"] = [cols[i] for i in zupt_cols]

    # 8) 特征工程：模长 / RMS
    name2idx = {c: i for i, c in enumerate(cols)}
    def _try_add_magnitude(prefix_words: Tuple[str, ...], tag: str):
        nonlocal X, cols, name2idx
        triads: Dict[str, List[str]] = {}
        for c in cols:
            cl = c.lower()
            if any(p in cl for p in prefix_words):
                root = c.rsplit("_", 1)[0] if "_" in c else c
                triads.setdefault(root, []).append(c)
        for root, subcols in triads.items():
            # 只处理 *_x/_y/_z 这样的三轴组
            axes = [s for s in subcols if s.endswith(("_x", "_y", "_z"))]
            if len(axes) < 3: continue
            idxs = [name2idx[s] for s in axes if s in name2idx]
            if len(idxs) != 3: continue
            mag = np.sqrt((X[:, idxs]**2).sum(axis=1, keepdims=True))
            X = np.concatenate([X, mag], axis=1)
            cols.append(f"{root}_{tag}_mag")
            name2idx = {c: i for i, c in enumerate(cols)}

    if "acc" in cfg.add_magnitude_for: _try_add_magnitude(("acc","acceler"), "acc")
    if "gyro" in cfg.add_magnitude_for: _try_add_magnitude(("gyro","gyr"), "gyro")

    if cfg.add_rms_k and cfg.add_rms_k > 1:
        k = int(cfg.add_rms_k)
        kernel = np.ones(k, dtype=np.float32) / k
        rms_cols = []
        for i, c in enumerate(cols):
            sq = X[:, i]**2
            m2 = np.convolve(sq, kernel, mode="same")
            rms = np.sqrt(m2)[:, None]
            X = np.concatenate([X, rms], axis=1)
            rms_cols.append(f"{c}_rms{k}")
        cols += rms_cols
        name2idx = {c: i for i, c in enumerate(cols)}

    # 9) 标准化 / 截断
    if cfg.normalize == "zscore":
        if cfg.stats_path and os.path.exists(cfg.stats_path):
            mu, sigma, _ = load_stats(cfg.stats_path)
            if mu.shape[0] != X.shape[1]:
                raise ValueError(f"stats长度不匹配：{mu.shape[0]} vs X={X.shape[1]}")
        else:
            mu = X.mean(axis=0); sigma = X.std(axis=0)
            sigma[sigma < 1e-6] = 1.0
            if cfg.stats_path:
                save_stats(mu.astype(np.float32), sigma.astype(np.float32), cfg.stats_path, cols)
        X = (X - mu) / (sigma + 1e-8)

    if cfg.clip_after_std is not None:
        X = np.clip(X, -cfg.clip_after_std, cfg.clip_after_std)

    # === 统一出口：构造输出并返回（无条件） ===
    out = pd.DataFrame(X, columns=cols)
    if cfg.time_col:
        # 确保长度一致
        if len(t) != len(out):
            raise ValueError(f"time vector length {len(t)} != out length {len(out)}")
        # 如果 out 里已有同名列，就覆盖并把它移到第0列；否则插入
        if cfg.time_col in out.columns:
            out[cfg.time_col] = t
            # 把时间列移动到最前
            ordered = [cfg.time_col] + [c for c in out.columns if c != cfg.time_col]
            out = out[ordered]
        else:
            out.insert(0, cfg.time_col, t)

    info["len_out"] = len(out)
    info["cols_out"] = cols

    # 可选绘图（不会 return）
    if cfg.plot:
        import matplotlib.pyplot as plt
        os.makedirs(cfg.plot_outdir, exist_ok=True)
        chs = cfg.plot_channels or cols[:min(3, len(cols))]
        for c in chs:
            try:
                if c not in df_raw.columns and c in out.columns:
                    fig, ax = plt.subplots(figsize=(10, 2.6))
                    ax.plot(out.index, out[c], linewidth=1.0, label="proc")
                    ax.set_title(c); ax.legend(); fig.tight_layout()
                    fig.savefig(os.path.join(cfg.plot_outdir, f"{c}.png"), dpi=150)
                    plt.close(fig)
                elif c in df_raw.columns and c in out.columns:
                    fig, ax = plt.subplots(figsize=(10, 2.6))
                    ax.plot(df_raw.index, df_raw[c], linewidth=0.8, label="raw", alpha=0.8)
                    ax.plot(out.index, out[c], linewidth=1.0, label="proc")
                    ax.set_title(c); ax.legend(); fig.tight_layout()
                    fig.savefig(os.path.join(cfg.plot_outdir, f"{c}.png"), dpi=150)
                    plt.close(fig)
            except Exception as e:
                print(f"[plot warn] {c}: {e}")

    return out, info
# ----------------------------
# 测试/示例
if __name__ == "__main__":
    
    df_raw=pd.read_csv("E:/demo/python/Gait1016/datasets/ZJH/gait01_results/gait01_pre00.csv",encoding='utf-8-sig')
    # 假设你已经读取好 IMU 数据
    # —---------------------------希尔伯特包络计算示例---------------------------
    # 配置
    # 对某个 IMU 的三轴加速度计算包络
    # 2) 配置预处理
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
        enable_zupt=True,       # 如需对足IMU做简化ZUPT可开
        add_magnitude_for=("acc","vel"),  # 自动添加模长列
        add_rms_k=0,
        normalize="none",      # 训练/推理一致时建议用zscore
        stats_path=None,         # 如果已有训练统计量可填 npz 路径
        clip_after_std=None,    # 如需截断可设值
        plot=False               # 是否另存“预处理前后对比图”
    )

    # 3) 运行统一预处理（关键）
    df_proc, info = preprocess_df(df_raw, cfg)

    # 4) 选择要做包络的列（示例：对加速度模长 & 某些轴做包络）
    cols_for_envelope = []
    #if "imu4ang_x" in df_proc.columns:
        #                          cols_for_envelope.append("imu4acc_z")
    if False:
        for ax in ["imu4acc_x","imu4acc_y", "imu4acc_z","imu7acc_x", "imu7acc_y", "imu7acc_z"]:
        #for ax in ["imu2ang_x", "imu7ang_x"]:
        #for ax in ["imu4acc_x", "imu7acc_x"]:
            if ax in df_proc.columns:
                cols_for_envelope.append(ax)
    if False:
        for ax in ["imu4vel_x","imu4vel_y","imu4vel_z", "imu7vel_x", "imu7vel_y", "imu7vel_z"]:
        #for ax in ["imu2ang_x", "imu7ang_x"]:
        #for ax in ["imu4acc_x", "imu7acc_x"]:
            if ax in df_proc.columns:
                cols_for_envelope.append(ax)
    
    if True:
        for ax in ["imu4ang_x", "imu7ang_x"]:
        #for ax in ["imu2ang_x", "imu7ang_x"]:
        #for ax in ["imu4acc_x", "imu7acc_x"]:
            if ax in df_proc.columns:
                cols_for_envelope.append(ax)

    # 5) 做希尔伯特包络（会新增 *_env 列；支持绘图保存）
    df_out = apply_hilbert_envelope(
        df_proc,
        columns=cols_for_envelope,
        plot=True,                   # 导出原始vs包络图（这里的“原始”是指预处理后的信号）
        plot_outdir="./eval_viz/hilbert_plots"  # 保存目录
    )
    
    # —---------------------------频域分析示例---------------------------   
    psd_df, peaks_df = apply_frequency_analysis(
        df_proc,
        columns=cols_for_envelope,  # 你想分析的列
        plot=True,
        plot_outdir="./eval_viz/freq_plots",fs=cfg.fs, window_seconds=1000.0, nfft_factor=1
    )
    # 查看峰值（步频通常在 0.8–3 Hz 区间）
    print(peaks_df.sort_values("peak_freq_hz"))
    # 取 0–10 Hz 频带的 PSD
    subset = psd_df.loc[(psd_df.index >= 0) & (psd_df.index <= 10)]
    
    # ----------------------------自相关函数分析示例---------------------------
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
    print(peaks_df_acf.sort_values("peak_lag_seconds"))
    # ACF主峰的滞后时间常≈一个“步周期”，其倒数≈步频
    # ----------------------------小波分解特征示例---------------------------
    # 取“细节 D4”（约 1.56–3.13 Hz，步频主带）：
    if  True:
        df_d4, bands_d4 = apply_wavelet_select(
            df_proc, columns=cols_for_envelope, fs=cfg.fs,
            wavelet="haar", max_level=4,mode="reflect",
            part="detail", select_level= 3,
            plot=True, plot_outdir="./eval_viz/DWT_plots/bio2.2")
    
        print("已完成小波分解特征提取，频带信息：")
        print(bands_d4)
    #----------------------------小波系数提取示例---------------------------
    if True:
        coeffs_d4, bands_d4 = extract_wavelet_coeffs(
        df_proc, columns=cols_for_envelope, fs=cfg.fs,
        wavelet="haar", max_level=5,
        transform_type="swt",
        part="detail", select_level=3,
        mode="symmetric",
        plot=True, plot_outdir="./eval_viz/wavelet_coeffs/haardwt"
        )
        print("已完成小波系数提取，频带信息：")
        print(bands_d4)
    #----------------------------EMD分解示例---------------------------
    if False:
        imfs_by_col, summary_df = apply_emd(
        df_proc, columns=cols_for_envelope, fs=cfg.fs,
        max_imfs=None,            # 或者 6/8 等上限
        spline_kind="cubic",
        nbsym=2,
        sift_max_iter=1000,
        plot=True,
        plot_outdir="./eval_viz/emd"
        )
        print("已完成 EMD 分解，摘要信息：")
        print(summary_df)
