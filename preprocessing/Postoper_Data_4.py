import numpy as np
import csv
import os
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks,savgol_filter
import re
#def read_csv_file(filename, encoding='gbk'):
#这个自动处理一些imu数据，用足部imu来计算判断相位和支撑相摆动相
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

def data_pre_analyze01(data, time_interval):
    """
    数据预处理01：时间列归一化并插值到固定时间间隔。
    :param data: np.ndarray, 输入数据
    :param time_interval: float, 时间间隔
    :return: np.ndarray, 插值后的数据
    """
    start_time = data[0, 0]
    data[:, 0] -= start_time  # 减去初始时间

    end_time = data[-1, 0]
    new_time = np.arange(0, end_time + time_interval, time_interval)

    new_data = np.zeros((len(new_time), data.shape[1]))
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(new_time, data[:, 0], data[:, i])

    new_data[:, 0] = new_time
    return new_data

def find_and_filter_peaks(data, col_a, col_b, col_c, min_distance=20):
    """
    寻找极大值点并处理异常值。
    :param data: np.ndarray, 输入数据
    :param col_a: int, 数据导数所在列的列号
    :param col_b: int, 数据值所在列的列号
    :param col_c: int, 时间列的列号
    :param min_distance: int, 相邻极大值点最小间隔（默认60点=1.2秒@50Hz）
    :return: list, 极大值点的行号（相对输入数据）
    """
    data_a = data[:, col_a]
    peaks = []

    for i in range(10, len(data_a)):
        # 检查当前点是否从负变为正，并且其前10个点均为负
        if data_a[i - 1] < 0 and data_a[i] >= 0 and all(data_a[i - 10:i] < 0):
            # 检查与上一个点的距离
            if peaks and (i - peaks[-1]) < min_distance:
                continue  # 跳过过近的极大值点
            peaks.append(i)
    return peaks

def data_pre_analyze02(pre01, col_a, col_b, col_c, col_d, col_time):
    """
    数据预处理02：通过极大值点过滤生成新数据。
    :param pre01: np.ndarray, 数据预处理后的数组
    :param col_a: int, 左髋角速度所在列的列号
    :param col_b: int, 左髋角度所在列的列号
    :param col_c: int, 右髋角速度所在列的列号
    :param col_d: int, 右髋角度所在列的列号
    :param col_time: int, 时间列的列号
    :return: tuple, 包括极大值点的行号（相对pre02）和新的数据数组pre02
    """
    filtered_peaks_01 = find_and_filter_peaks(pre01, col_a, col_b, col_time, min_distance=50)  # 左髋极大值点
    filtered_peaks_02 = find_and_filter_peaks(pre01, col_c, col_d, col_time, min_distance=50)  # 右髋极大值点

    # 找到两个列表的最大值和最小值
    max_value = max(max(filtered_peaks_01), max(filtered_peaks_02))
    min_value = min(min(filtered_peaks_01), min(filtered_peaks_02))

    pre02_start = min_value - 0
    pre02_end = max_value + 1
    pre02 = pre01[pre02_start:pre02_end]
    pre02[:, col_time] -= pre02[0, col_time]

    relative_peaks_01 = [idx - pre02_start for idx in filtered_peaks_01]
    relative_peaks_02 = [idx - pre02_start for idx in filtered_peaks_02]

    return relative_peaks_01, relative_peaks_02, pre02

def calculate_average_interval(data_list):
    """
    对列表排序，计算数据间隔，并返回其平均值。

    :param data_list: list, 包含浮点数的列表
    :return: float, 数据间隔的平均值
    """
    if len(data_list) < 2:
        raise ValueError("数据列表长度应至少为2，无法计算间隔。")

    # 排序列表
    sorted_list = sorted(data_list)

    # 计算间隔
    intervals = [sorted_list[i] - sorted_list[i - 1] for i in range(1, len(sorted_list))]

    # 求间隔的平均值
    average_interval = sum(intervals) / len(intervals)

    return average_interval

def cal_foot_acc(data_pre,
                 col_imu4_angle_x=35, col_imu4_acc_y=40, col_imu4_acc_z=43,
                 col_imu7_angle_x=62, col_imu7_acc_y=67, col_imu7_acc_z=70):
    """
    计算足部加速度，并将结果添加到数据组中。

    :param data_pre: np.ndarray, 输入数据组（如pre02）
    :param col_imu4_angle_x: int, IMU4 x方向角度所在列号（默认为35）
    :param col_imu4_acc_y: int, IMU4 y方向加速度所在列号（默认为40）
    :param col_imu4_acc_z: int, IMU4 z方向加速度所在列号（默认为43）
    :param col_imu7_angle_x: int, IMU7 x方向角度所在列号（默认为62）
    :param col_imu7_acc_y: int, IMU7 y方向加速度所在列号（默认为67）
    :param col_imu7_acc_z: int, IMU7 z方向加速度所在列号（默认为70）
    :return: np.ndarray, 添加新列后的数据组
    """
    if data_pre.shape[0] == 0:
        raise ValueError("输入数据组为空，无法计算加速度。")

    # IMU4 加速度计算
    imu4_angle_x_radians = np.radians(data_pre[:, col_imu4_angle_x])  # 将角度转换为弧度
    imu4_acc = - data_pre[:, col_imu4_acc_y] * np.cos(imu4_angle_x_radians) - \
               data_pre[:, col_imu4_acc_z] * np.sin(-imu4_angle_x_radians)

    # IMU7 加速度计算
    imu7_angle_x_radians = np.radians(data_pre[:, col_imu7_angle_x])  # 将角度转换为弧度
    imu7_acc = - data_pre[:, col_imu7_acc_y] * np.cos(imu7_angle_x_radians) - \
               data_pre[:, col_imu7_acc_z] * np.sin(-imu7_angle_x_radians)

    # 将新列添加到数据组中
    new_data_with_acc = np.column_stack((data_pre, imu4_acc, imu7_acc))

    return new_data_with_acc

def find_heel_strike(data, col, length=10, threshold=-2, min_distance=20):
    """
    在数据中寻找波谷点（极小值点）。

    :param data: np.ndarray, 输入数据组（如 pre02_with_acc）
    :param col: int, 要寻找波谷的列号
    :param length: int, 滑动窗口的长度（默认为 20）
    :param threshold: float, 极小值点的值必须小于该阈值（默认为 -2）
    :param min_distance: int, 相邻波谷点之间的最小距离（默认为 10）
    :return: list, 极小值点的位置（行号列表）
    """
    # 初始化极小值点的位置列表
    valley_positions = []

    # 滑动窗口范围必须足够大才能判断极小值
    if length < 10:
        raise ValueError("滑动窗口长度必须大于或等于10")

    i = length // 2
    # 遍历数据，确保滑动窗口不会超出边界
    while i < len(data) - length // 2:
        # 提取滑动窗口中的数据
        window = data[i - length // 2:i + length // 2, col]

        # 确保当前点在窗口中的位置为最小值，且处于窗口的中间部分
        if (
                data[i, col] == min(window)  # 当前点是窗口中的最小值
                and (length // 2 - 5 <= window.tolist().index(data[i, col]) <= length // 2 + 5)  # 位于窗口中间部分
                and data[i, col] < threshold  # 极小值点的值小于阈值
        ):
            # 检查与上一个波谷点的距离是否大于 min_distance
            if not valley_positions or (i - valley_positions[-1] >= min_distance):
                valley_positions.append(i)  # 记录当前点的位置
            i += length // 2  # 跳过一定范围内的点，避免重复检测
        else:
            i += 1

    return valley_positions

def find_toe_off(data, col, num=10, threshold=1, min_distance=20):
    """
    寻找数据中满足条件的点（Toe-Off时刻）。

    :param data: np.ndarray, 输入数据组（如 pre02_with_acc）
    :param col: int, 要处理的列号
    :param num: int, 判断当前点之前的点数（默认为 20）
    :param threshold: float, 阈值（默认为 1）
    :param min_distance: int, 相邻Toe-Off点之间的最小距离（默认为 10）
    :return: list, 满足条件的点所在的行号列表
    """
    # 初始化保存满足条件点的行号列表
    toe_off_positions = []

    # 遍历数据
    for i in range(num, len(data)):
        # 当前点是否大于阈值
        if data[i, col] > threshold:
            # 检查当前点前 num 个点是否都小于阈值
            if all(-threshold < data[i - j, col] < threshold for j in range(1, num + 1)):
                # 检查与上一个Toe-Off点的距离是否大于 min_distance
                if not toe_off_positions or (i - toe_off_positions[-1] >= min_distance):
                    toe_off_positions.append(i)

    return toe_off_positions

def phase_rate(heel_strike, toe_off):
    """
    计算站立相和摆动相的百分比（改进版）
    确保每个脚趾离地点位于两个连续的足跟着地点之间
    """
    # 排序输入列表
    hs = sorted(heel_strike)
    to = sorted(toe_off)

    # 初始化对齐后的事件点
    aligned_hs = []
    aligned_to = []

    # 使用双指针进行对齐
    to_ptr = 0
    for i in range(len(hs) - 1):
        current_hs = hs[i]
        next_hs = hs[i + 1]

        # 寻找位于当前HS和下一个HS之间的TO点
        while to_ptr < len(to):
            current_to = to[to_ptr]
            # 找到第一个满足 current_hs < current_to < next_hs 的TO点
            if current_hs < current_to < next_hs:
                aligned_hs.append(current_hs)
                aligned_to.append(current_to)
                to_ptr += 1  # 移动到下一个TO点
                break
            # 如果TO点太小，继续寻找
            elif current_to <= current_hs:
                to_ptr += 1
            # 如果TO点太大，保留到下次循环处理
            else:
                break

    # 计算各周期时长
    stance_durations = []
    swing_durations = []

    for i in range(len(aligned_hs)):
        # 支撑相：HS到TO
        stance = aligned_to[i] - aligned_hs[i]
        # 摆动相：TO到下一个HS
        if i < len(aligned_hs) - 1:
            swing = aligned_hs[i + 1] - aligned_to[i]
        else:
            # 处理最后一个周期的情况
            swing = hs[-1] - aligned_to[i] if aligned_to[i] < hs[-1] else 0

        if stance > 0 and swing > 0:
            stance_durations.append(stance)
            swing_durations.append(swing)

    # 计算平均值
    avg_stance = sum(stance_durations) / len(stance_durations) if stance_durations else 0
    avg_swing = sum(swing_durations) / len(swing_durations) if swing_durations else 0

    total = avg_stance + avg_swing
    if total == 0:
        return 0.0, 0.0, aligned_hs, aligned_to

    return (avg_stance / total) * 100, (avg_swing / total) * 100, aligned_hs, aligned_to

def cal_step_length(data,
                    row_heel_strike,
                    thigh_length=0.4,
                    shank_length=0.4,
                    col_hip_left=2,
                    col_knee_left=3,
                    col_hip_right=5,
                    col_knee_right=6):
    """
    计算步长，历遍足跟着地点的行号组成的列表row_heel_strike。

    :param data: np.ndarray, 输入数据组
    :param row_heel_strike: list, 足跟着地点的行号列表
    :param thigh_length: float, 大腿长度，默认值为0.5
    :param shank_length: float, 小腿长度，默认值为0.5
    :param col_hip_left: int, 左髋角度列号，默认值为2
    :param col_knee_left: int, 左膝角度列号，默认值为3
    :param col_hip_right: int, 右髋角度列号，默认值为5
    :param col_knee_right: int, 右膝角度列号，默认值为6
    :return: list, 包含每次步长的数列
    """
    # 转换为numpy数组（如果不是）
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # 存储步长的列表
    step_lengths = []

    # 遍历足跟着地点的行号
    for row in row_heel_strike:
        # 提取对应行的角度值
        left_hip_angle = data[row, col_hip_left]
        left_knee_angle = data[row, col_knee_left]
        right_hip_angle = data[row, col_hip_right]
        right_knee_angle = data[row, col_knee_right]

        # 将角度值转换为弧度值
        left_hip_rad = np.abs(np.radians(left_hip_angle))
        left_knee_rad = np.abs(np.radians(left_knee_angle))
        right_hip_rad = np.abs(np.radians(right_hip_angle))
        right_knee_rad = np.abs(np.radians(right_knee_angle))

        # 计算步长公式
        step_length = (
                thigh_length * (np.sin(left_hip_rad) + np.sin(right_hip_rad)) +
                shank_length * (np.sin(left_knee_rad) + np.sin(right_knee_rad))
        )

        # 保留小数点后三位
        step_lengths.append(round(step_length, 3))

    return step_lengths

def calculate_phase_percent(cycle_hs, cycle_to, time_interval, cycle_duration):
    """
    计算单个周期内的支撑相和摆动相百分比
    :param cycle_hs: list, 周期内足跟着地点的相对索引（如[0, 50]）
    :param cycle_to: list, 周期内脚趾离地点的相对索引（如[30]）
    :param time_interval: float, 数据时间间隔（秒）
    :param cycle_duration: float, 周期总时长（秒）
    :return: tuple, (支撑相占比, 摆动相占比)
    """
    # 有效性检查
    if not cycle_hs or not cycle_to or cycle_to[0] <= cycle_hs[0]:
        return 0.0, 0.0

    # 提取关键事件点
    hs_start = cycle_hs[0]
    to_event = cycle_to[0]
    # print(hs_start,to_event)

    # 计算支撑相时间
    stance_time = (to_event - hs_start) * time_interval

    # 计算摆动相时间
    if len(cycle_hs) > 1:  # 存在下个HS点
        hs_end = cycle_hs[1]
        swing_time = (hs_end - to_event) * time_interval
    else:  # 使用周期总时长
        swing_time = cycle_duration - (to_event * time_interval)

    # 计算百分比
    total_phase = stance_time + swing_time
    if total_phase <= 0:
        return 0.0, 0.0
    return (stance_time / total_phase) * 100, (swing_time / total_phase) * 100

def collect_cycle_params(cycles_data, cycle_indices, global_heel_strike, global_toe_off,
                         time_interval, side_tag, thigh_length=0.4, shank_length=0.4):
    params_list = []
    for idx, (cycle, (start_idx, end_idx)) in enumerate(zip(cycles_data, cycle_indices)):
        # cycle 包含完整周期数据
        # 筛选属于当前周期的事件
        cycle_hs = [hs for hs in global_heel_strike if start_idx <= hs < end_idx]
        cycle_to = [to for to in global_toe_off if start_idx <= to < end_idx]

        relative_hs = [hs - start_idx for hs in cycle_hs]
        relative_to = [to - start_idx for to in cycle_to]

        # 计算支撑相占比
        cycle_duration = (end_idx - start_idx) * time_interval
        stance_per, swing_per = calculate_phase_percent(
            sorted(relative_hs), sorted(relative_to),
            time_interval, cycle_duration
        )

        # 计算步长（使用完整周期数据）和步速
        step_lengths = cal_step_length(cycle, relative_hs, thigh_length, shank_length)
        avg_step_length = sum(step_lengths) / len(step_lengths) if step_lengths else 0
        step_speed = avg_step_length / cycle_duration if cycle_duration > 0 else 0

        params = {
            f"{side_tag}_cycle_num": idx + 1,
            f"{side_tag}_duration": cycle_duration,
            f"{side_tag}_stance_percent": stance_per,
            f"{side_tag}_swing_percent": swing_per,
            f"{side_tag}_step_length": avg_step_length,
            f"{side_tag}_step_speed": step_speed
        }
        params_list.append(params)
    return params_list

def step_width_from_ndarray(
    data: np.ndarray,
    peaks_left,
    peaks_right,
    L_thigh: float,
    L_shank: float,
    offset_thigh_left=0.0,
    offset_shank_left=0.0,
    offset_thigh_right=0.0,
    offset_shank_right=0.0,
    fs=50.0
):
    """
    基于 ndarray 计算真实步宽（米）

    data 形状: (N, 80)   # N 行，80 列（与原始 CSV 列数一致）
    列索引按原顺序：
      0:time, 1:上身, 2:左髋, 3:左膝, 4:左踝, 5:右髋, 6:右膝, 7:右踝,
      8~16: imu1 (ang_x,vel_x,acc_x,ang_y,vel_y,acc_y,ang_z,vel_z,acc_z),
      17~25: imu2, 26~34: imu3, 35~43: imu4, 44~52: imu5, 53~61: imu6, 62~70: imu7
    我们需要的 Y 轴角度列：
      imu2_y = 20, imu3_y = 29, imu5_y = 47, imu6_y = 56
    """
    # 提取需要的四列
    cols = np.array([20, 29, 47, 56])  # imu2_y, imu3_y, imu5_y, imu6_y
    angles = data[:, cols]  # shape (N, 4)

    # 低通滤波
    def lowpass(sig, cutoff=5, fs=fs, order=2):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype='low', analog=False)
        return filtfilt(b, a, sig)

    for i in range(4):
        angles[:, i] = lowpass(angles[:, i])

    # 角度补偿并转为弧度
    offsets = np.array([
        offset_thigh_left,
        offset_shank_left,
        offset_thigh_right,
        offset_shank_right
    ])
    angles_rad = np.radians(angles - offsets)

    # 计算横向位移（米）
    left_disp  = np.abs(angles_rad[:, 0]) * L_thigh + np.abs(angles_rad[:, 1]) * L_shank
    right_disp = np.abs(angles_rad[:, 2]) * L_thigh + np.abs(angles_rad[:, 3]) * L_shank

    # 合并周期起点并计算步宽
    all_peaks = sorted(peaks_left + peaks_right)
    widths = []
    for i in range(len(all_peaks) - 1):
        start, end = all_peaks[i], all_peaks[i + 1]
        left_max  = left_disp[start:end].max()
        right_max = right_disp[start:end].max()
        widths.append(abs(left_max - right_max))

    return widths

def merge_bilateral_params(left_params, right_params):
    """
    合并左右腿周期参数至统一表格
    :param left_params: list of dict, 左腿周期参数
    :param right_params: list of dict, 右腿周期参数
    :return: list of dict, 合并后的参数列表
    """
    merged = []
    max_cycles = max(len(left_params), len(right_params))

    for i in range(max_cycles):
        row = {}
        # 添加左腿参数（若存在）
        if i < len(left_params):
            row.update(left_params[i])
        else:
            row.update({k: None for k in left_params[0].keys()})

        # 添加右腿参数（若存在）
        if i < len(right_params):
            row.update(right_params[i])
        else:
            row.update({k: None for k in right_params[0].keys()})

        merged.append(row)
    return merged

def split_gait_cycles(data, peaks):
    """
    分割步态周期并返回数据段及起止索引
    :return: (周期数据列表, 起止索引列表)
    """
    cycles = []
    indices = []
    peaks = sorted(peaks)

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end <= start:
            continue  # 跳过无效周期
        cycle_data = data[start:end]
        cycles.append(cycle_data)  # 保留完整周期数据
        indices.append((start, end))
    return cycles, indices

def calculate_step_width(df, peaks_left, peaks_right, imu4_y_col='imu4ang_y', imu7_y_col='imu7ang_y'):
    """
    计算步宽（step width）

    参数:
    - df: 原始数据 DataFrame
    - peaks_left: 左髋关节峰值索引列表
    - peaks_right: 右髋关节峰值索引列表
    - imu4_y_col: 左脚Y方向角度列名，默认为 'imu4ang_y'
    - imu7_y_col: 右脚Y方向角度列名，默认为 'imu7ang_y'

    返回:
    - step_widths: 每个步态周期的步宽列表（单位：角度，若需转换为米需乘比例因子）
    """
    step_widths = []

    # 合并所有步态周期起点
    all_peaks = sorted(peaks_left + peaks_right)

    # 遍历每个步态周期
    for i in range(len(all_peaks) - 1):
        start = all_peaks[i]
        end = all_peaks[i + 1]

        # 提取该周期内左右脚的Y方向角度
        left_foot_y = df[imu4_y_col].iloc[start:end].abs().max()
        right_foot_y = df[imu7_y_col].iloc[start:end].abs().max()

        # 计算步宽（角度差值）
        step_width = abs(left_foot_y - right_foot_y)
        step_widths.append(step_width)

    return step_widths



    """
    处理 CSV 数据：
    - 关节角度单独处理
    - 每个 IMU 单独处理
    - 去重重复时间戳（保留第一个）
    - 按全局时间轴填充缺失值（上一帧保持）
    - 返回处理后的 DataFrame 和 headers

    Args:
        data: 原始 DataFrame
        headers: 列名列表
        time_col: 时间列名，默认 headers[0]

    Returns:
        processed_data: 处理后的 DataFrame（时间列为第一列）
        processed_headers: 列名列表
    """
    if time_col is None:
        time_col = headers[0]

    df = data.copy()
    
    # 确保时间列为 float 或 datetime
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # 设置时间列为索引
    df = df.set_index(time_col)

    # 构建全局时间轴
    t_min = df.index.min()
    t_max = df.index.max()
    if pd.api.types.is_datetime64_any_dtype(df.index):
        full_index = pd.date_range(start=t_min, end=t_max, freq=pd.Timedelta(milliseconds=10))
    else:
        full_index = pd.Index(np.round(np.arange(t_min, t_max + 0.01, 0.01), 5))

    processed_parts = []

    # --- 1. 处理关节角度 ---
    joint_cols = headers[1:8]
    joint_df = df[joint_cols].copy()
    joint_df = joint_df[~joint_df.index.duplicated(keep='first')]  # 保留第一个重复
    joint_df = joint_df.reindex(full_index).ffill()
    processed_parts.append(joint_df)

    # --- 2. 处理每个 IMU ---
    imu_cols = headers[8:]
    num_imus = 7
    imu_len = 9  # 每个IMU列数

    for i in range(num_imus):
        cols = imu_cols[i*imu_len : (i+1)*imu_len]
        imu_df = df[cols].copy()
        imu_df = imu_df[~imu_df.index.duplicated(keep='first')]  # 保留第一个重复
        imu_df = imu_df.reindex(full_index).ffill()
        processed_parts.append(imu_df)

    # --- 合并所有部分 ---
    result_df = pd.concat(processed_parts, axis=1)
    result_df.index.name = time_col

    # --- 返回 DataFrame + headers ---
    result_df = result_df.reset_index()  # 时间列恢复为普通列
    processed_headers = [time_col] + joint_cols + imu_cols

    return result_df, processed_headers

#处理IMU数据，删除重复数据

    """
    处理 CSV 数据（输入 ndarray + headers）：
    - 关节角度单独处理
    - 每个 IMU 单独处理
    - 去重重复时间戳（保留第一个）
    - 按全局时间轴填充缺失值（上一帧保持）
    - 返回处理后的 ndarray 和 headers

    Args:
        data: 原始 ndarray，形状 (n_samples, n_columns)
        headers: 列名列表
        time_col: 时间列名，默认 headers[0]

    Returns:
        processed_data: 处理后的 ndarray
        processed_headers: 列名列表
    """
    if time_col is None:
        time_col = headers[0]

    # 转成 DataFrame
    df = pd.DataFrame(data, columns=headers)

    # 确保时间列为 float 或 datetime
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # 设置时间列为索引
    df = df.set_index(time_col)

    # 构建全局时间轴
    t_min = df.index.min()
    t_max = df.index.max()
    if pd.api.types.is_datetime64_any_dtype(df.index):
        full_index = pd.date_range(start=t_min, end=t_max, freq=pd.Timedelta(milliseconds=10))
    else:
        full_index = pd.Index(np.round(np.arange(t_min, t_max + 0.01, 0.04), 5))

    processed_parts = []

    # --- 1. 处理关节角度 ---
    joint_cols = headers[1:8]
    joint_df = df[joint_cols].copy()
    joint_df = joint_df[~joint_df.index.duplicated(keep='first')]  # 保留第一个重复
    joint_df = joint_df.reindex(full_index).ffill()
    processed_parts.append(joint_df)

    # --- 2. 处理每个 IMU ---
    imu_cols = headers[8:]
    num_imus = 7
    imu_len = 9  # 每个IMU列数

    for i in range(num_imus):
        cols = imu_cols[i*imu_len : (i+1)*imu_len]
        imu_df = df[cols].copy()
        imu_df = imu_df[~imu_df.index.duplicated(keep='first')]  # 保留第一个重复
        imu_df = imu_df.reindex(full_index).ffill()
        processed_parts.append(imu_df)

    # --- 合并所有部分 ---
    result_df = pd.concat(processed_parts, axis=1)
    result_df.index.name = time_col

    # --- 恢复时间列为普通列 ---
    result_df = result_df.reset_index()
    processed_headers = [time_col] + joint_cols + imu_cols

    # 转回 ndarray
    processed_data = result_df.to_numpy()

    return processed_data, processed_headers


    """
    处理IMU数据：
    1. 每个关节角度单独处理
    2. 每个IMU单独处理
    3. 去掉连续重复数据（保留第一次出现）
    4. 对每个通道线性插值到 target_fs Hz
    
    参数:
        data: np.ndarray, 原始数据 (n_samples, n_columns)
        headers: list, 列名
        target_fs: float, 目标采样频率，默认25Hz
    
    返回:
        new_data: np.ndarray, 处理后的数据
        new_headers: list, 列名
    """
    
    time = data[:, 0]
    
    # 分离关节角度
    joint_angles = {f'joint_{i+1}': data[:, i+1].reshape(-1,1) for i in range(7)}
    
    # 分离IMU数据，每个IMU 9列
    imus = {}
    for i in range(7):
        start_col = 8 + i*9
        imus[f'imu_{i+1}'] = data[:, start_col:start_col+9]
    
    def remove_consecutive_duplicates(t, arr):
        """去掉连续重复行，保留第一次"""
        mask = np.ones(len(arr), dtype=bool)
        for i in range(1, len(arr)):
            if np.all(arr[i] == arr[i-1]):
                mask[i] = False
        return t[mask], arr[mask]
    
    def linear_interpolate(t_old, data_old, t_new):
        df = pd.DataFrame(data_old, index=t_old)
        df_new = df.reindex(df.index.union(t_new))
        df_new = df_new.interpolate(method='index')
        return df_new.loc[t_new].to_numpy()
    
    # 新时间轴
    t_new = np.arange(time[0], time[-1], 1/target_fs)
    
    # 处理关节角度
    joint_new = []
    for k in joint_angles:
        t_old, vals_old = remove_consecutive_duplicates(time, joint_angles[k])
        vals_new = linear_interpolate(t_old, vals_old, t_new)
        joint_new.append(vals_new)
    joint_new = np.hstack(joint_new)
    
    # 处理IMU数据
    imu_new = []
    for k in imus:
        t_old, vals_old = remove_consecutive_duplicates(time, imus[k])
        vals_new = linear_interpolate(t_old, vals_old, t_new)
        imu_new.append(vals_new)
    imu_new = np.hstack(imu_new)
    
    # 合并时间+关节+IMU
    new_data = np.hstack([t_new.reshape(-1,1), joint_new, imu_new])
    
    # 构建新的headers
    new_headers = ['time'] + [f'joint_{i+1}' for i in range(7)]
    for i in range(7):
        new_headers += [f'imu_{i+1}_d{j+1}' for j in range(9)]
    
    return new_data, new_headers



    """
    智能处理 CSV 数据：
    1. 每列（关节角或 IMU）独立去掉连续不变的数据（每段只保留第一个）
    2. 最后统一插值到 target_hz
    3. 返回 ndarray + headers
    """
    if time_col is None:
        time_col = headers[0]

    df = pd.DataFrame(data, columns=headers)

    # 确保时间列为 float
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = df[time_col].astype(float)

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    dt = 1 / target_hz
    target_time = np.arange(t_min, t_max, dt)
    target_df = pd.DataFrame({time_col: target_time})

    processed_parts = []

    # --- 1. 处理每列关节角 ---
    joint_cols = headers[1:8]  # 7列关节角
    for col in joint_cols:
        col_df = df[[time_col, col]].copy()
        diff_col = col_df[col].diff().fillna(1)
        mask_col = diff_col.ne(0)  # 保留变化或第一行
        reduced_col = col_df[mask_col].copy()
        t_orig = reduced_col[time_col].to_numpy()
        v_orig = reduced_col[col].to_numpy()
        v_interp
        interp_col = pd.merge_asof(target_df, reduced_col, on=time_col, direction='backward')
        interp_col = interp_col.interpolate(method='linear')
        processed_parts.append(interp_col[[col]])

    # --- 2. 处理每列 IMU ---
    imu_cols = headers[8:]
    for col in imu_cols:
        col_df = df[[time_col, col]].copy()
        diff_col = col_df[col].diff().fillna(1)
        mask_col = diff_col.ne(0)
        reduced_col = col_df[mask_col].copy()
        interp_col = pd.merge_asof(target_df, reduced_col, on=time_col, direction='backward')
        interp_col = interp_col.interpolate(method='linear')
        processed_parts.append(interp_col[[col]])

    # --- 合并所有部分 ---
    result_df = pd.concat([target_df] + processed_parts, axis=1)
    processed_headers = headers.copy()
    processed_data = result_df.to_numpy()

    return processed_data, processed_headers
#从imu数据中剥离开重复数据，并每个imu单独插值成为25Hz的数据
# =============== 你的函数：仅替换“插值块” ===============
def process_imu_joint_and_plantar_safe_ndarray(
    data: np.ndarray,
    headers: list,
    target_hz: int = 50,
    time_col: str | None = None,
    interp_method: str = "pchip",
    max_gap_seconds: float = 1.0,
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

#从"右侧 大腿长37 小腿长35 步宽17"这样的信息中捕捉3个数字
def extract_numbers_from_folder(folder_name):
        # 使用正则表达式提取所有数字
        numbers = re.findall(r'\d+', folder_name)
        
        # 检查是否提取到了3个数字
        if len(numbers) != 3:
            raise ValueError(f"文件夹名称 '{folder_name}' 中没有找到3个数字")
        
        # 将提取到的数字转换为浮点数
        thigh_length = float(numbers[0])  # 大腿长
        calf_length = float(numbers[1])   # 小腿长
        step_width = float(numbers[2])    # 步宽
        
        return thigh_length, calf_length, step_width
# =============== 新增：稳健1D插值工具 ===============
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
#在文件夹中定位csv文件
def find_original_csv(folder_path):
    """修正：匹配格式为 data_数字.csv 的文件"""
    # 新增对下划线的支持：^data_\d+\.csv$
    # ^data_ 表示以 "data_" 开头
    # \d+ 表示一个或多个数字（时间戳）
    # \.csv$ 表示以 .csv 结尾
    pattern = r'^data_\d+\.csv$'  # 关键修改：添加了下划线
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and re.match(pattern, file):
            return file_path
    
    # 如果未找到，可打印调试信息
    print(f"在 {folder_path} 中未找到匹配 {pattern} 的文件，目录下文件有：{os.listdir(folder_path)}")
    return None

def calculate_phase_percentages(pre02_a,heel_strikes, toe_offs, time_interval):
    """
    计算支撑相和摆动相占比
    :param heel_strikes: 足跟着地事件索引列表
    :param toe_offs: 足尖离地事件索引列表
    :param time_interval: 采样时间间隔
    :return: 平均支撑相占比, 平均摆动相占比
    """
    if not heel_strikes or not toe_offs:
        return 0.0, 0.0  # 如果没有检测到事件，返回0

    # 确保事件顺序正确
    heel_strikes = sorted(heel_strikes)
    toe_offs = sorted(toe_offs)

    # 移除不在合理范围内的事件
    heel_strikes = [hs for hs in heel_strikes if 0 <= hs < len(pre02_a)]
    toe_offs = [to for to in toe_offs if 0 <= to < len(pre02_a)]

    # 确保每个足跟着地事件后有对应的足尖离地事件
    valid_pairs = []
    for hs in heel_strikes:
        # 找到在hs之后的第一个to
        possible_toes = [to for to in toe_offs if to > hs]
        if possible_toes:
            to = min(possible_toes)
            valid_pairs.append((hs, to))
            # 移除已使用的toe_off事件
            toe_offs.remove(to)

    if not valid_pairs:
        return 0.0, 0.0

    # 计算每个周期的支撑相和摆动相
    support_times = []
    swing_times = []

    for i in range(len(valid_pairs) - 1):
        hs1, to1 = valid_pairs[i]
        hs2, to2 = valid_pairs[i + 1]

        # 支撑相时间 = 足跟着地到足尖离地
        support_time = (to1 - hs1) * time_interval

        # 摆动相时间 = 足尖离地到下一次足跟着地
        swing_time = (hs2 - to1) * time_interval

        # 整个周期时间
        cycle_time = (hs2 - hs1) * time_interval

        # 验证数据合理性
        if support_time > 0 and swing_time > 0 and cycle_time > 0 and \
                abs((support_time + swing_time) - cycle_time) < 0.1:  # 允许10%误差
            support_times.append(support_time)
            swing_times.append(swing_time)

    if not support_times or not swing_times:
        return 0.0, 0.0

    # 计算平均值
    avg_support_time = sum(support_times) / len(support_times)
    avg_swing_time = sum(swing_times) / len(swing_times)
    avg_cycle_time = avg_support_time + avg_swing_time

    # 计算百分比
    support_per = (avg_support_time / avg_cycle_time) * 100
    swing_per = (avg_swing_time / avg_cycle_time) * 100

    # 确保总和为100%
    total = support_per + swing_per
    if abs(total - 100) > 1:  # 如果误差超过1%，重新计算
        support_per = (avg_support_time / avg_cycle_time) * 100
        swing_per = 100 - support_per

    return support_per, swing_per

#获取脚跟着地和脚趾离地点
def detect_gait_events(df, fs=50, smooth_window=11, smooth_poly=3,hs_threshold=3, to_threshold=50, min_interval=0.3):
    """
    从足部IMU (imu4ang_x/imu7ang_x, imu4vel_x/imu7vel_x) 检测步态事件。


    Heel strike (HS): 足部角度局部最小值，并结合角速度负峰确认。
    Toe off (TO): 足部角度局部最大值，并结合角速度正峰确认。


    参数:
    df: DataFrame，至少包含列 'imu4ang_x','imu7ang_x','imu4vel_x','imu7vel_x'
    fs: 采样频率 Hz
    smooth_window: Savitzky-Golay 滤波窗口长度 (必须为奇数)
    smooth_poly: Savitzky-Golay 多项式阶数
    hs_threshold: heel strike 阈值 (角速度负峰需小于该值)
    to_threshold: toe off 阈值 (角速度正峰需大于该值)
    min_interval: 相邻事件的最小时间间隔 (s)


    返回:
    peaks_left, toe_off_left, peaks_right, toe_off_right
    (都是索引数组)
    """
    # 提取角度和角速度
    ang_left = df['imu4ang_x'].values
    ang_right = df['imu7ang_x'].values
    #vel_left = df['imu4vel_x'].values
    #vel_right = df['imu7vel_x'].values
    distance_use=10
    # === Heel Strike: 角度极小值 + 角速度负峰 ===
    ang_left_op=-ang_left
    ang_right_op=-ang_right
    cand_hs_left, _ = find_peaks(ang_left_op)
    cand_hs_right, _ = find_peaks(ang_right_op)

    #peaks_left = [i for i in cand_hs_left if vel_left[i] < hs_threshold]
    #peaks_right = [i for i in cand_hs_right if vel_right[i] < hs_threshold]
    peaks_left = [i for i in cand_hs_left if ang_left[i] < hs_threshold]
    peaks_right = [i for i in cand_hs_right if ang_right[i] < hs_threshold]

    # === Toe Off: 角度极大值 + 角速度正峰 ===
    cand_to_left, _ = find_peaks(ang_left, distance_use)
    cand_to_right, _ = find_peaks(ang_right, distance_use)

    #toe_off_left = [i for i in cand_to_left if vel_left[i] > to_threshold]
    #toe_off_right = [i for i in cand_to_right if vel_right[i] > to_threshold]
    toe_off_left = [i for i in cand_to_left if ang_left[i] > to_threshold]
    toe_off_right = [i for i in cand_to_right if ang_right[i] > to_threshold]
    
    return peaks_left, toe_off_left, peaks_right,toe_off_right
#低通滤波
def lowpass_6Hz_multichannel(data: np.ndarray,
                             fs_target: float = 50.0,
                             order: int = 4,fc = 6.0) -> np.ndarray:
    """
    对多通道信号做 6 Hz 零相位低通滤波。

    参数
    ----
    data : (N, M) ndarray
        第 0 列 = 时间（s），第 1…M-1 列 = 各通道信号
    fs_target : float, optional
        内部重采样目标频率，默认 50 Hz
    order : int, optional
        Butterworth 阶数，默认 4

    返回
    ----
    out : (N, M) ndarray
        时间列不变，所有信号列已 6 Hz 低通
    """
    t = data[:, 0]                 # 时间
    X = data[:, 1:].T              # 信号矩阵，(M-1, N)
    n_ch = X.shape[0]

    # 1. 重采样 → 等间隔
    n_new = int((t[-1] - t[0]) * fs_target) + 1
    t_reg = np.linspace(t[0], t[-1], n_new)
    X_reg = np.array([np.interp(t_reg, t, x) for x in X])  # (M-1, n_new)

    # 2. 设计 6 Hz（默认6Hz） 低通
    b, a = butter(order, fc / (fs_target / 2), btype='low')

    # 3. 零相位滤波（逐通道）
    X_filt_reg = np.array([filtfilt(b, a, x) for x in X_reg])

    # 4. 插值回原时间戳
    X_filt = np.array([np.interp(t, t_reg, x) for x in X_filt_reg]).T  # (N, M-1)

    # 5. 组装结果
    return np.column_stack([t, X_filt])

def calculate_gait(filename,thigh_length,shank_length,step,hs_threshold=3, to_threshold=50):
    # 第一步：数据归一化，保存为pre00_00,pre01
    result_dir = create_result_folder(filename)
    #data, headers = read_csv_file(filename)
    rawdata, rawheaders = read_csv_file(filename)
    data,headers=process_imu_joint_safe_ndarray(rawdata,rawheaders)
    pre00_path = os.path.join(result_dir, os.path.basename(filename).replace('.csv', '_pre00.csv'))
    save_csv_file(data,headers,pre00_path)
    pre01 = data_pre_analyze01(data, time_interval)
    pre01_path = os.path.join(result_dir, os.path.basename(filename).replace('.csv', '_pre01.csv'))
    save_csv_file(pre01, headers, pre01_path)

    # 第二步：寻找左髋和右髋的极大值点
    # 提取第一个极大值点到最后一个极大值点的数据内容，保存为pre02（这样可以提取中间的有效数据）
    peaks01, peaks02, pre02 = data_pre_analyze02(pre01, 18, 2, 45, 5, 0)
    pre02_path = os.path.join(result_dir, os.path.basename(filename).replace('.csv', '_pre02.csv'))
    save_csv_file(pre02, headers, pre02_path)
    pre02_b = lowpass_6Hz_multichannel(pre02,fc=4)
    pre02_a = cal_foot_acc(pre02_b)  # 中间过程，计算足部x方向加速度
    headers.extend(['左脚加速度x',
                    '右脚加速度x',
                    ])
    pre02a_path = os.path.join(result_dir, os.path.basename(filename).replace('.csv', '_pre02_a.csv'))
    save_csv_file(pre02_a, headers, pre02a_path)
    # 第三步：计算各个参数
    df_pre02 = pd.DataFrame(data=pre02_a, columns=headers)
    peaks_left, toe_off_left, peaks_right, toe_off_right=detect_gait_events(df_pre02,hs_threshold=hs_threshold, to_threshold=to_threshold)
    # 计算参数1：左右腿运动周期
    period_left = calculate_average_interval(peaks_left) * time_interval
    period_right = calculate_average_interval(peaks_right) * time_interval
    print("周期：左腿→%.2fs；右腿→%.2fs" % (period_left, period_right))

    # 计算参数2：左右腿支撑相摆动相占比
    stance_per_left, swing_per_left = calculate_phase_percentages(pre02_a,peaks_left, toe_off_left,time_interval)
    #stance_per_left, swing_per_left, heel_strike_left, toe_off_left = phase_rate(heel_strike_left, toe_off_left)
    print("左脚足跟着地点：", peaks_left)
    print("左脚脚趾离地点：", toe_off_left)
    print("左脚支撑相占比(%%)：%.2f；摆动相占比(%%)：%.2f" % (stance_per_left, swing_per_left))
    stance_per_right, swing_per_right = calculate_phase_percentages(pre02_a,peaks_right, toe_off_right,time_interval)
    #stance_per_right, swing_per_right, heel_strike_right, toe_off_right = phase_rate(heel_strike_right, toe_off_right)
    print("右脚足跟着地点：", peaks_right)
    print("右脚脚趾离地点：", toe_off_right)
    print("右脚支撑相占比(%%)：%.2f；摆动相占比(%%)：%.2f" % (stance_per_right, swing_per_right))

    # 计算参数3：左右腿平均步长
    step_length_left = cal_step_length(pre02_a, peaks_left,thigh_length,shank_length)
    print("左腿步长（默认大小腿长0.4m）：", step_length_left)
    if len(step_length_left) > 0:
        step_length_left_mean = sum(step_length_left) / len(step_length_left)
    else:
         step_length_left_mean =0

    print("左腿平均步长（默认大小腿长0.4m）：%.3f m" % step_length_left_mean)
    step_length_right = cal_step_length(pre02_a, peaks_right,thigh_length,shank_length)
    print("右腿步长（默认大小腿长0.4m）：", step_length_right)
    if len(step_length_right)>0:
        step_length_right_mean = sum(step_length_right) / len(step_length_right)
    else:
        step_length_right_mean =0
    print("右腿平均步长（默认大小腿长0.4m）：%.3f m" % step_length_right_mean)

    # 计算参数4：左右腿平均步速
    speed_left = step_length_left_mean / period_left
    print("左腿平均步速（默认大小腿长0.4m）：%.3f m/s" % speed_left)
    speed_right = step_length_right_mean / period_right
    print("右腿平均步速（默认大小腿长0.4m）：%.3f m/s" % speed_right)
    speed = 0.5 * (speed_left + speed_right)

    # 计算参数5：左右脚平均步宽
    widths = step_width_from_ndarray(
        pre02_a,
        peaks_left,
        peaks_right,
        L_thigh=thigh_length,
        L_shank=shank_length,
        offset_thigh_left=np.radians(2.1),
        offset_shank_left=np.radians(-1.3),
        offset_thigh_right=np.radians(1.8),
        offset_shank_right=np.radians(-0.9)
    )
    widths_mean = step+sum(widths) / len(widths)
    print("左右腿平均真实步宽（米）：%.3f m"%widths_mean)

    gait_parameters = [0] * 20
    #计算参数5：总平均周期、步长
    peak_all = peaks_left + peaks_right
    period_all = min(max(calculate_average_interval(peak_all) * time_interval if peak_all else 0, 0), 9.9)
    gait_parameters[15] = period_all
    step_all = step_length_left + step_length_right
    step_average = min(max(sum(step_all) / len(step_all) if step_all else 0, 0), 9.9)
    # 定义步态参数表头
    gait_params_headers = [
        "大腿长度(m)", "小腿长度(m)", "步宽(m)",
        "左腿摆动相占比(%)", "右腿摆动相占比(%)",
        "左腿支撑相占比(%)", "右腿支撑相占比(%)",
        "左腿运动周期(s)", "右腿运动周期(s)",
        "左腿平均步长(m)", "右腿平均步长(m)",
        "左腿平均步速(m/s)", "右腿平均步速(m/s)",
        "平均运动周期(s)", "平均步长(m)", "步速(m/s)",
        "步幅(m)", "步频(步/s)"
    ]
    gait_parameters[0] = thigh_length#大腿长
    gait_parameters[1] = shank_length#小腿长
    gait_parameters[2] = step#初始步宽
    gait_parameters[3] = swing_per_left#左腿摆动相占比
    gait_parameters[4] = swing_per_right#右腿摆动相占比
    gait_parameters[5] = stance_per_left#左腿支撑相
    gait_parameters[6] = stance_per_right#右腿支撑相
    gait_parameters[7] = period_left#左腿运动周期(s)
    gait_parameters[8] = period_right#右腿运动周期(s)
    gait_parameters[9]= step_length_left_mean#平均步长（左）
    gait_parameters[10]=step_length_right_mean#平均步长（右）
    gait_parameters[11] = widths_mean#平均步宽
    gait_parameters[13] = speed_left#左腿平均步速
    gait_parameters[14] = speed_right#右腿平均步速
    gait_parameters[15] = period_all#平均运动周期
    gait_parameters[16] = step_average#平均步长
    gait_parameters[17] = speed#平均步速
    stride_length = step_length_right_mean + step_length_left_mean
    gait_parameters[18] = stride_length#步幅
    gait_parameters[19] = 1 / period_all #步频

    return gait_params_headers,gait_parameters

if __name__ == "__main__":
    root_dir="E:/demo/python/Gait1016/datasets/rawZJH/gait01.csv"
    print("预处理完成")
    rawdata, rawheaders = read_csv_file(root_dir)
    data,headers=process_imu_joint_and_plantar_safe_ndarray(
        data=rawdata, headers=rawheaders, target_hz=50, time_col="时间",
        interp_method="pchip",         # "linear" | "pchip" | "cubic" | "akima" | "nearest"
        max_gap_seconds=1.5,          # >0.2 s 的缺口一律置 NaN（50Hz ≈ 10帧）
        allow_extrapolate=False,       # 不外推
        unwrap_angle_cols=None  # 对角度先 unwrap 再插
    )
    result_dir = create_result_folder(root_dir)
    pre00_path = os.path.join(result_dir, os.path.basename(root_dir).replace('.csv', '_pre00.csv'))
    save_csv_file(data,headers,pre00_path)