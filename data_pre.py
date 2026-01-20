import numpy as np
import csv
import os


def read_csv_file(filename, encoding='utf-8-sig'):
    """
    读取CSV文件。
    :param filename: str, 输入的CSV文件名
    :param encoding: str, 文件编码 (默认为utf-8-sig)
    :return: tuple, 包括数据数组和标题行
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    try:
        with open(filename, newline='', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # 获取标题行
            rows = list(reader)

        # 尝试将数据转换为浮点数数组
        try:
            data = np.array(rows, dtype=float)
        except ValueError:
            # 如果转换失败，可能是文件包含非数字数据
            # 尝试跳过第一行（可能是标题行）
            if len(rows) > 1:
                data = np.array(rows[1:], dtype=float)
            else:
                data = np.array([], dtype=float)

        return data, headers
    except Exception as e:
        raise RuntimeError(f"读取CSV文件失败: {str(e)}")


def save_csv_file(data, headers, filename, encoding='utf-8-sig'):
    """
    保存数据到CSV文件。
    :param data: np.ndarray, 数据数组
    :param headers: list, 标题行
    :param filename: str, 输出的文件名
    :param encoding: str, 文件编码 (默认为utf-8-sig)
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, mode='w', newline='', encoding=encoding) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data)
    except Exception as e:
        raise RuntimeError(f"保存CSV文件失败: {str(e)}")


def data_pre_analyze01(data, time_interval):
    """
    数据预处理01：时间列归一化并插值到固定时间间隔。
    :param data: np.ndarray, 输入数据
    :param time_interval: float, 时间间隔
    :return: np.ndarray, 插值后的数据
    """
    if data.size == 0:
        return np.array([])

    try:
        start_time = data[0, 0]
        data[:, 0] -= start_time  # 减去初始时间

        end_time = data[-1, 0]
        new_time = np.arange(0, end_time + time_interval, time_interval)

        new_data = np.zeros((len(new_time), data.shape[1]))
        for i in range(data.shape[1]):
            new_data[:, i] = np.interp(new_time, data[:, 0], data[:, i])

        new_data[:, 0] = new_time
        return new_data
    except Exception as e:
        raise RuntimeError(f"数据预处理01失败: {str(e)}")


def find_and_filter_peaks(data, col_a, col_b, col_c):
    """
    寻找极大值点并处理异常值。
    :param data: np.ndarray, 输入数据
    :param col_a: int, 数据导数所在列的列号
    :param col_b: int, 数据值所在列的列号
    :param col_c: int, 时间列的列号
    :return: list, 极大值点的行号（相对输入数据）
    """
    if data.size == 0:
        return []

    try:
        data_a = data[:, col_a]
        peaks = []

        for i in range(10, len(data_a)):
            # 检查当前点是否从负变为正，并且其前10个点均为负
            if data_a[i - 1] < 0 and data_a[i] >= 0 and all(data_a[i - 10:i] < 0):
                peaks.append(i)

        return peaks
    except Exception as e:
        raise RuntimeError(f"寻找极大值点失败: {str(e)}")


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
    if pre01.size == 0:
        return [], [], np.array([])

    try:
        filtered_peaks_01 = find_and_filter_peaks(pre01, col_a, col_b, col_time)  # 左髋极大值点
        filtered_peaks_02 = find_and_filter_peaks(pre01, col_c, col_d, col_time)  # 右髋极大值点

        # 如果找不到极大值点，返回空结果
        if not filtered_peaks_01 or not filtered_peaks_02:
            return [], [], np.array([])

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
    except Exception as e:
        raise RuntimeError(f"数据预处理02失败: {str(e)}")


def calculate_average_interval(data_list):
    """
    对列表排序，计算数据间隔，并返回其平均值。

    :param data_list: list, 包含浮点数的列表
    :return: float, 数据间隔的平均值
    """
    if len(data_list) < 2:
        return 0.0

    try:
        # 排序列表
        sorted_list = sorted(data_list)

        # 计算间隔
        intervals = [sorted_list[i] - sorted_list[i - 1] for i in range(1, len(sorted_list))]

        # 求间隔的平均值
        return sum(intervals) / len(intervals) if intervals else 0.0
    except Exception as e:
        raise RuntimeError(f"计算平均间隔失败: {str(e)}")


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
    if data_pre.size == 0:
        return np.array([])

    try:
        # IMU4 加速度计算
        imu4_angle_x_radians = np.radians(data_pre[:, col_imu4_angle_x])  # 将角度转换为弧度
        imu4_acc = - data_pre[:, col_imu4_acc_y] * np.cos(imu4_angle_x_radians) - \
                   data_pre[:, col_imu4_acc_z] * np.sin(-imu4_angle_x_radians)

        # IMU7 加速度计算
        imu7_angle_x_radians = np.radians(data_pre[:, col_imu7_angle_x])  # 将角度转换为弧度
        imu7_acc = - data_pre[:, col_imu7_acc_y] * np.cos(imu7_angle_x_radians) - \
                   data_pre[:, col_imu7_acc_z] * np.sin(-imu7_angle_x_radians)

        # 将新列添加到数据组中
        return np.column_stack((data_pre, imu4_acc, imu7_acc))
    except Exception as e:
        raise RuntimeError(f"计算足部加速度失败: {str(e)}")


def find_heel_strike(data, col, length=20, threshold=-2):
    """
    在数据中寻找波谷点（极小值点）。

    :param data: np.ndarray, 输入数据组（如 pre02_with_acc）
    :param col: int, 要寻找波谷的列号
    :param length: int, 滑动窗口的长度（默认为 20）
    :param threshold: float, 极小值点的值必须小于该阈值（默认为 -2）
    :return: list, 极小值点的位置（行号列表）
    """
    if data.size == 0:
        return []

    try:
        # 初始化极小值点的位置列表
        valley_positions = []

        # 滑动窗口范围必须足够大才能判断极小值
        if length < 10:
            length = 10

        i = length // 2
        # 遍历数据，确保滑动窗口不会超出边界
        while i < len(data) - length // 2:
            # 提取滑动窗口中的数据
            window = data[i - length // 2:i + length // 2, col]

            # 确保当前点在窗口中的位置为最小值，且处于窗口的中间部分
            if (
                    data[i, col] == min(window)  # 当前点是窗口中的最小值
                    and (length // 2 - 5 <= np.argmin(window) <= length // 2 + 5)  # 位于窗口中间部分
                    and data[i, col] < threshold  # 极小值点的值小于阈值
            ):
                valley_positions.append(i)  # 记录当前点的位置
                k = 1
                while i + k < len(data) and data[i + k, col] == data[valley_positions[-1], col]:
                    k = k + 1
                i = i + k
            else:
                i = i + 1

        return valley_positions
    except Exception as e:
        raise RuntimeError(f"寻找足跟着地点失败: {str(e)}")


def find_toe_off(data, col, num=20, threshold=1):
    """
    寻找数据中满足条件的点（Toe-Off时刻）。

    :param data: np.ndarray, 输入数据组（如 pre02_with_acc）
    :param col: int, 要处理的列号
    :param num: int, 判断当前点之前的点数（默认为 10）
    :param threshold: float, 阈值（默认为 1）
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
                toe_off_positions.append(i)

    return toe_off_positions


def phase_rate(heel_strike, toe_off):
    """
    计算站立相和摆动相的百分比。

    :param heel_strike: list, Heel Strike 的时间点列表（从小到大排列）
    :param toe_off: list, Toe Off 的时间点列表（从小到大排列）
    :return: tuple, (站立相百分比, 摆动相百分比)
    """
    # 确保 heel_strike 和 toe_off 都是从小到大排列
    heel_strike = sorted(heel_strike)
    toe_off = sorted(toe_off)

    # 删除 toe_off 中早于第一个 heel_strike 的值
    while toe_off and toe_off[0] < heel_strike[0]:
        toe_off.pop(0)

    # 确保两个列表长度一致
    length = min(len(heel_strike), len(toe_off))

    # 初始化 stance 和 swing
    stance = []
    swing = []

    # 计算 stance 和 swing
    for i in range(length - 1):
        stance.append(toe_off[i] - heel_strike[i])
        swing.append(heel_strike[i + 1] - toe_off[i])

    # 计算均值
    stance_mean = sum(stance) / len(stance) if stance else 0
    swing_mean = sum(swing) / len(swing) if swing else 0

    # 比较两个均值的百分比
    total = stance_mean + swing_mean

    # ==== 新增：检查分母是否为零 ====
    if total == 0:
        return 0.0, 0.0

    stance_percentage = (stance_mean / total) * 100
    swing_percentage = (swing_mean / total) * 100

    return stance_percentage, swing_percentage


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
    if data.size == 0 or not row_heel_strike:
        return []

    try:
        # 存储步长的列表
        step_lengths = []

        # 遍历足跟着地点的行号
        for row in row_heel_strike:
            # 确保行号在有效范围内
            if row < 0 or row >= len(data):
                continue

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
    except Exception as e:
        raise RuntimeError(f"计算步长失败: {str(e)}")


if __name__ == "__main__":
    try:
        filename = "data_20241223125235.csv"
        time_interval = 0.02

        # 第一步：数据归一化，保存为pre01
        data, headers = read_csv_file(filename)
        pre01 = data_pre_analyze01(data, time_interval)
        save_csv_file(pre01, headers, filename.split('.')[0] + "_pre01.csv")

        # 第二步：寻找左髋和右髋的极大值点
        # 提取第一个极大值点到最后一个极大值点的数据内容，保存为pre02（这样可以提取中间的有效数据）
        peaks01, peaks02, pre02 = data_pre_analyze02(pre01, 18, 2, 45, 5, 0)
        pre02_a = cal_foot_acc(pre02)  # 中间过程，计算足部x方向加速度
        headers.extend(['左脚加速度x', '右脚加速度x'])
        save_csv_file(pre02_a, headers, filename.split('.')[0] + "_pre02.csv")

        # 第三步：计算各个参数
        peaks_left = find_and_filter_peaks(pre02, 18, 2, 0)
        peaks_right = find_and_filter_peaks(pre02, 45, 5, 0)

        print("左髋极大值点的行号（相对pre02）：", peaks_left)
        print("右髋极大值点的行号（相对pre02）：", peaks_right)

        # 计算参数1：左右腿运动周期
        period_left = calculate_average_interval(peaks_left) * time_interval
        period_right = calculate_average_interval(peaks_right) * time_interval
        print("周期：左腿→%.2fs；右腿→%.2fs" % (period_left, period_right))

        # 计算参数2：左右腿支撑相摆动相占比
        heel_strike_left = find_heel_strike(pre02_a, 71)
        print("左脚足跟着地点：", heel_strike_left)
        toe_off_left = find_toe_off(pre02_a, 71)
        print("左脚脚趾离地点：", toe_off_left)
        stance_per_left, swing_per_left = phase_rate(heel_strike_left, toe_off_left)
        print("左脚支撑相占比(%%)：%.2f；摆动相占比(%%)：%.2f" % (stance_per_left, swing_per_left))
        heel_strike_right = find_heel_strike(pre02_a, 72)
        print("右脚足跟着地点：", heel_strike_right)
        toe_off_right = find_toe_off(pre02_a, 72)
        print("右脚脚趾离地点：", toe_off_right)
        stance_per_right, swing_per_right = phase_rate(heel_strike_right, toe_off_right)
        print("右脚支撑相占比(%%)：%.2f；摆动相占比(%%)：%.2f" % (stance_per_right, swing_per_right))

        # 计算参数3：左右腿平均步长
        step_length_left = cal_step_length(pre02_a, heel_strike_left)
        print("左腿步长（默认大小腿长0.4m）：", step_length_left)
        step_length_left_mean = sum(step_length_left) / len(step_length_left) if step_length_left else 0
        print("左腿平均步长（默认大小腿长0.4m）：%.3f m" % step_length_left_mean)
        step_length_right = cal_step_length(pre02_a, heel_strike_right)
        print("右腿步长（默认大小腿长0.4m）：", step_length_right)
        step_length_right_mean = sum(step_length_right) / len(step_length_right) if step_length_right else 0
        print("右腿平均步长（默认大小腿长0.4m）：%.3f m" % step_length_right_mean)

        # 计算参数4：左右腿平均步速
        speed_left = step_length_left_mean / period_left if period_left != 0 else 0
        print("左腿平均步速（默认大小腿长0.4m）：%.3f m/s" % speed_left)
        speed_right = step_length_right_mean / period_right if period_right != 0 else 0
        print("右腿平均步速（默认大小腿长0.4m）：%.3f m/s" % speed_right)
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")