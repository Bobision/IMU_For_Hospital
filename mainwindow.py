import csv
import os
import sys, io
import xjtu_rc
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import BlueTeethIMU
from ui import Ui_MainWindow
from button import connect_button_actions
import draw_pic
import datetime
import time
import numpy as np
from collections import deque
import threading
import traceback
from Postoper_Data_4 import read_csv_file,process_imu_joint_and_plantar_safe_ndarray,create_result_folder,save_csv_file
from preprocessing.preprocess_imu import PreprocessConfig
from scripts.TO_detect import calculate_gait_parameters
import pandas as pd
import queue
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.yourorg.gaitgui")
def resource_path(relative_path):
            """获取资源文件路径（兼容 PyInstaller 打包）"""
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.abspath(".")
            return os.path.join(base_path, relative_path) 
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 设置窗口图标
        # 获取当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建图标路径
        icon_path = os.path.join(base_dir, 'images', 'logo_transpar.ico')
        self.setWindowIcon(QtGui.QIcon(":/pic/images/logo_transpar.ico"))
        # 界面初始化
        self.setupUi(self)
        # 设置窗口标题为"下肢步态评估系统"
        self.setWindowTitle("下肢步态评估系统")
        # ==== 新增：初始化保存路径 ====
        # 获取当前程序所在目录
        self.default_save_path = os.path.abspath(os.path.dirname(__file__))
        # 设置初始保存路径为程序所在目录
        self.lineEdit_save_path.setText(self.default_save_path)
        # 连接选择路径按钮信号
        self.pushButton_select_path.clicked.connect(self.select_save_path)
        # 添加按钮函数(与线程无关的)
        connect_button_actions(self)
        # ==== 提前定义 upperbody_angle ====
        self.upperbody_angle = 0.0
        # 定义线程相关变量
        self.serial_port = None
        self.imu_thread = None
        self.start_time = time.perf_counter()
        self.now_time = time.perf_counter()
        self.time_count = 0
        self.pushButton_begin_read.clicked.connect(self.begin_read_thread_action)
        self.pushButton_end_read.clicked.connect(self.end_read_thread_action)
        self._writer_running = False
        # 每个设备的“最新三轴角度（°）”，用于后续关节角计算（避免从 QLabel 读回）
        self._latest_ang = {i: [0.0, 0.0, 0.0] for i in range(2, 9)}
        # 电量的 O(1) 指数滑动均值（避免每次 sum/len）
        self._bat_ema = {i: None for i in range(2, 9)}
        #保存数据相关的初始化
        self.record_queue = queue.Queue(maxsize=20000)  # 足够大避免堵塞
        self.writer_thread = None
        self.rec_dropped = 0          # 若队列满导致丢弃的计数（理想情况应=0）
        self.rec_started = False
        # 大腿长范围：0.01-2米
        self.doubleSpinBox_thigh_length_data.setMinimum(0.01)
        self.doubleSpinBox_thigh_length_data.setMaximum(2.0)
        self.doubleSpinBox_thigh_length_data.setDecimals(2)  # 3位小数

        # 小腿长范围：0.01-2米
        self.doubleSpinBox_shank_length_data.setMinimum(0.01)
        self.doubleSpinBox_shank_length_data.setMaximum(2.0)
        self.doubleSpinBox_shank_length_data.setDecimals(2)  # 3位小数

        #步宽范围
        self.doubleSpinBox_leg_width.setMinimum(0.01)
        self.doubleSpinBox_leg_width.setMaximum(2.0)
        self.doubleSpinBox_leg_width.setDecimals(2)
        # ==== 新增：连接标定按钮信号 ====
        self.pushButton_standard.clicked.connect(self.calibration_action)

        # 定义标签更新相关变量
        self.comp_data = [0.0] * 7

        # 定义绘图相关变量
        self.points = []
        self.scene, self.graphicsView, self.timer = draw_pic.init_graphics_view(self)
        # 在__init__中初始化时使用固定长度的deque
        max_points = 1000  # 限制每个关节最多存储1000个数据点
        self.joint_time_data = [deque(maxlen=max_points) for _ in range(6)]
        self.joint_angle_data = [deque(maxlen=max_points) for _ in range(6)]
        # ==== 为graphicsView_2初始化关节角度实时曲线图 ====
        self.joint_curves, self.joint_time_data, self.joint_angle_data,self.joint_timer= draw_pic.init_joint_angle_plot(
            self)

        # 配置定时器更新曲线
        self.joint_timer.timeout.connect(self.update_joint_angle_plot)
        self.joint_timer.start(100)  # 10Hz更新频率

        # ==== 新增：设备号到关节索引的映射 ====
        self.device_to_joint = {
            3: 0,  # 左髋角度
            4: 1,  # 左膝角度
            5: 2,  # 左踝角度
            6: 3,  # 右髋角度
            7: 4,  # 右膝角度
            8: 5  # 右踝角度
        }

        # 在 __init__ 方法中添加设备名称映射
        self.device_names = {
            2: "背部IMU模块",
            3: "左大腿IMU模块",
            4: "左小腿IMU模块",
            5: "左脚IMU模块",
            6: "右大腿IMU模块",
            7: "右小腿IMU模块",
            8: "右脚IMU模块"
        }
        # 定义数据记录相关变量(决定频率)
        self.refresh = 20
        # 创建定时器
        self.timer_record = QTimer()
        self.timer_record.setTimerType(QtCore.Qt.PreciseTimer)
        # 将定时器的超时信号连接到更新标签的槽函数
        self.timer_record.timeout.connect(self.record_data)

        self._rows_buffer = []

        self.csv_file = None
        self.csv_writer = None
        self.filename = ''
        self.fieldnames = ['时间',
                           '上身角度', '左髋角度', '左膝角度', '左踝角度', '右髋角度', '右膝角度', '右踝角度',
                           'imu1ang_x', 'imu1vel_x', 'imu1acc_x',
                           'imu1ang_y', 'imu1vel_y', 'imu1acc_y',
                           'imu1ang_z', 'imu1vel_z', 'imu1acc_z',
                           'imu2ang_x', 'imu2vel_x', 'imu2acc_x',
                           'imu2ang_y', 'imu2vel_y', 'imu2acc_y',
                           'imu2ang_z', 'imu2vel_z', 'imu2acc_z',
                           'imu3ang_x', 'imu3vel_x', 'imu3acc_x',
                           'imu3ang_y', 'imu3vel_y', 'imu3acc_y',
                           'imu3ang_z', 'imu3vel_z', 'imu3acc_z',
                           'imu4ang_x', 'imu4vel_x', 'imu4acc_x',
                           'imu4ang_y', 'imu4vel_y', 'imu4acc_y',
                           'imu4ang_z', 'imu4vel_z', 'imu4acc_z',
                           'imu5ang_x', 'imu5vel_x', 'imu5acc_x',
                           'imu5ang_y', 'imu5vel_y', 'imu5acc_y',
                           'imu5ang_z', 'imu5vel_z', 'imu5acc_z',
                           'imu6ang_x', 'imu6vel_x', 'imu6acc_x',
                           'imu6ang_y', 'imu6vel_y', 'imu6acc_y',
                           'imu6ang_z', 'imu6vel_z', 'imu6acc_z',
                           'imu7ang_x', 'imu7vel_x', 'imu7acc_x',
                           'imu7ang_y', 'imu7vel_y', 'imu7acc_y',
                           'imu7ang_z', 'imu7vel_z', 'imu7acc_z']
                        #    'index1',
                        #    'index2', 'index3', 'index4',
                        #    'index5', 'index6', 'index7']
        self.gait_paparemters = {}

        self.mapping1 = {'battery': 0, 'x': 1, 'y': 2, 'z': 3}
        self.mapping2 = {'bat': 0, 'ang': 1, 'vel': 2, 'acc': 3}
        self.pushButton_begin_record.clicked.connect(self.begin_record)
        self.pushButton_end_record.clicked.connect(self.stop_record_save)

        # ==== 新增部分：设备状态管理 ====
        # 设备状态字典：device_id (2-8) -> 最后接收时间
        self.device_last_seen = {i: 0 for i in range(2, 9)}
        # 初始化时所有设备都标记为未连接
        self.device_status = {i: "disconnected" for i in range(2, 9)}
        # 电池数据平滑处理：每个设备的电池值队列
        self.battery_history = {i: deque(maxlen=10000) for i in range(2, 9)}  # 保留最近值

        # 设备当前电池值（原始值，用于实时警告）
        self.current_battery = {i: 0 for i in range(2, 9)}
        self.last_displayed_battery = {i: 0 for i in range(2, 9)}  # 记录上次显示的电量值
        # 设备状态样式表
        self.status_styles = {
            "connected": "background-color: rgb(0, 255, 0);",  # 绿色
            "disconnected": "background-color: rgb(200, 200, 200);",  # 灰色
            "low_battery": "background-color: rgb(255, 0, 0);"  # 红色
        }

        # 状态检查定时器
        #self.status_timer = QTimer()
        #self.status_timer.timeout.connect(self.check_device_status)
        #self.status_timer.start(1000)  # 每秒检查一次
        self.data_save = [0.0] * (71)
        # 新建子线程绘图
        # self.plot_worker = PlotWorker(self.joint_time_data,
        #                               self.joint_angle_data,
        #                               parent=self)
        # # 信号槽连接：子线程 -> 主线程
        # self.plot_worker.plot_data_ready.connect(self.refresh_curves_slot)
        # self.plot_worker.start()

        # ==== 新增：标定相关变量 ====
        self.calibration_samples = {i: [] for i in range(2, 9)}  # 每个设备的标定样本
        self.calibration_data_timer = None  # 标定数据采集定时器
        self.calibration_end_timer = None  # 标定结束定时器

    def select_save_path(self):
        """选择数据保存路径"""
        # 弹出文件夹选择对话框
        selected_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择数据保存路径",
            self.lineEdit_save_path.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
        )

        if selected_path:  # 确保用户选择了路径而不是取消
            self.lineEdit_save_path.setText(selected_path)
# ==== 新增：足底压力线程相关方法 ====
# ==== 新增：左脚线程读取函数 ====
    # 其他原有方法保持不变...
    def calibration_action(self):
        """标定按钮点击事件处理"""
        # 检查是否已经开始读取数据
        if self.imu_thread is None or not self.imu_thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "无法标定",
                "请先开始读取IMU数据！\n\n操作步骤：\n1. 点击'开始读取'\n"
                "2. 等待设备连接并开始接收数据\n3. 点击'标定'按钮"
            )
            return

        # 重置标定样本存储
        self.calibration_samples = {i: [] for i in range(2, 9)}

        # 创建"正在标定中"对话框
        self.calibration_dialog = QtWidgets.QDialog(self)
        self.calibration_dialog.setWindowTitle("标定中")
        self.calibration_dialog.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint
        )

        # 添加标签
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("正在标定中，请保持姿势不动...\n（约3秒）")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        self.calibration_dialog.setLayout(layout)

        # 设置对话框大小
        self.calibration_dialog.resize(300, 100)

        # 启动数据采集定时器
        self.calibration_data_timer = QTimer()
        self.calibration_data_timer.timeout.connect(self.collect_calibration_data)
        self.calibration_data_timer.start(100)  # 每100ms采集一次数据

        # 设置3秒后结束标定
        self.calibration_end_timer = QTimer()
        self.calibration_end_timer.setSingleShot(True)
        self.calibration_end_timer.timeout.connect(self.finish_calibration)
        self.calibration_end_timer.start(3000)

        # 显示对话框
        self.calibration_dialog.exec_()

    def collect_calibration_data(self):
        """采集标定数据"""
        # 采集所有设备(2-8)的X轴角度数据
        for device_id in range(2, 9):
            label_name = f"label_origin_data_imu{device_id - 1}x_data"
            try:
                # 获取标签文本
                text = getattr(self, label_name).text()
                if text:  # 确保文本不为空
                    value = float(text)
                    self.calibration_samples[device_id].append(value)
            except (ValueError, AttributeError):
                # 处理可能的转换错误或属性不存在
                pass

    def finish_calibration(self):
        """完成标定并计算补偿值"""
        # 停止采集定时器
        if self.calibration_data_timer:
            self.calibration_data_timer.stop()

        # 关闭对话框
        if self.calibration_dialog:
            self.calibration_dialog.accept()
            self.calibration_dialog = None

        # 计算每个设备的补偿值
        for device_id, samples in self.calibration_samples.items():
            if samples:  # 确保有样本
                avg_value = sum(samples) / len(samples)
                # 存储补偿值（当前位置设为0度）
                self.comp_data[device_id - 2] = -avg_value

        # 显示完成提示
        QtWidgets.QMessageBox.information(
            self,
            "标定完成",
            "标定已完成",
            QtWidgets.QMessageBox.Ok
        )

        # 打印调试信息（可选）
        print("标定补偿值:")
        for i, val in enumerate(self.comp_data):
            print(f"设备 {i + 2}: {val:.3f}")

    def check_device_status(self):
        """检查设备连接状态，更新断开设备的状态"""
        current_time = time.time()
        for device_id in range(2, 9):  # 设备ID 2-8
            last_seen = self.device_last_seen.get(device_id, 0)

            # 如果超过10秒没有收到数据，认为设备已断开
            if current_time - last_seen > 10 and last_seen > 0:
                # 只有之前不是断开状态时才弹出提示
                if self.device_status[device_id] != "disconnected":
                    # 获取设备名称
                    device_name = self.device_names.get(device_id, f"IMU{device_id}")
                    # 弹出设备断开提示弹窗
                    QtWidgets.QMessageBox.warning(
                        self,
                        "设备连接中断",
                        f"{device_name}已断开连接！请检查设备供电和蓝牙连接。",
                        QtWidgets.QMessageBox.Ok
                    )

                # 更新设备状态
                self.update_device_status(device_id, "disconnected")

                # 清空历史电量数据并将电量显示置为0
                self.battery_history[device_id].clear()
                self.current_battery[device_id] = 0
                label_name = f"label_imu{device_id}_battery_data"
                getattr(self, label_name).setText("0")

    def update_device_status(self, device_id, status_type):
        """更新设备状态显示"""
        # 更新设备状态字典
        self.device_status[device_id] = status_type
        if status_type == "connected":
            status_text = "已连接"
            style = self.status_styles["connected"]
        elif status_type == "disconnected":
            status_text = "未连接"
            style = self.status_styles["disconnected"]
        else:  # low_battery
            status_text = "电量低"
            style = self.status_styles["low_battery"]

        # 更新状态文本
        label_name = f"label_imu{device_id}_status_text"
        getattr(self, label_name).setText(status_text)

        # 更新状态指示灯
        label_name = f"label_imu{device_id}_status_color"
        getattr(self, label_name).setStyleSheet(style)
    
    def begin_read_thread_action(self):
        """
        事件函数，打开数据读取的线程
        """

        # ==== 新增：串口状态检查 ====
        if self.serial_port is None:
            QtWidgets.QMessageBox.warning(
                self,
                "设备未连接",
                "请先搜索并打开蓝牙串口！\n\n操作步骤：\n1. 点击'搜索串口'\n"
                "2. 选择正确的COM端口\n3. 点击'打开'"
            )
            return
        self.imu_thread = imuThread(self.serial_port)
        self.imu_thread.update_label.connect(self.update_label_origin_angle_data_latest)

        # 在开始线程时重置所有设备状态
        for device_id in range(2, 9):
            self.device_status[device_id] = "disconnected"  # 重置状态
            self.update_device_status(device_id, "disconnected")
            self.battery_history[device_id].clear()  # 清空历史电量数据
            self.current_battery[device_id] = 0  # 重置当前电池值

        self.imu_thread.start()
        # ==== 新增：启动状态检查定时器 ====
        self.pushButton_begin_read.setEnabled(False)
        self.pushButton_imu_close.setEnabled(False)
        self.pushButton_end_read.setEnabled(True)
    # 其他方法保持不变...
    def end_read_thread_action(self):
        """
        事件函数，关闭数据读取的线程
        """
        # 停止IMU线程
        self.imu_thread.stop()

        # ==== 新增：重置计时器 ====
        self.start_time = time.perf_counter()  # 重置计时起点
        self.label_current_time_data.setText("0.0")  # 重置UI显示为0
        self.data_save[0] = 0.0  # 重置数据保存位置的时间值
        # 清空绘图数据
        if hasattr(self, 'joint_time_data'):
            for data in self.joint_time_data:
                data.clear()
        if hasattr(self, 'joint_angle_data'):
            for data in self.joint_angle_data:
                data.clear()
        if hasattr(self, 'plot_worker'):
            self.plot_worker.stop()

        # 立即将所有设备状态设为断开（灰色）
        for device_id in range(2, 9):
            self.device_status[device_id] = "disconnected"
            # 获取标签名
            status_text_label = f"label_imu{device_id}_status_text"
            status_color_label = f"label_imu{device_id}_status_color"
            battery_label = f"label_imu{device_id}_battery_data"

            # 更新为未连接状态（灰色）
            getattr(self, status_text_label).setText("未连接")
            getattr(self, status_color_label).setStyleSheet("background-color: rgb(200, 200, 200);")

            # 重置电池显示
            getattr(self, battery_label).setText("0")

            # 清空历史电量数据
            self.battery_history[device_id].clear()
            self.current_battery[device_id] = 0

        # 更新按钮状态
        self.pushButton_begin_read.setEnabled(True)
        self.pushButton_imu_close.setEnabled(True)
        self.pushButton_end_read.setEnabled(False)

    def update_label_joint_angle_data(self, data_type, device_number):
        """
        根据不同 IMU 的数据变化更新关节角度数据到 UI 标签
        :param data_type: 数据类型，可能是 'x', 'y', 'z' 或 'battery'
        :param device_number: IMU 设备编号
        """
        # 初始化 joint_data 变量
        joint_data = 0.0

        if data_type == 'x':
            if device_number in [2, 3, 6]:
                label_name1 = f"label_origin_data_imu{device_number - 1}{data_type}_data"
                imu_data1 = float(getattr(self, label_name1).text())
                joint_data = imu_data1 + self.comp_data[device_number - 2]
                if device_number == 2:
                    # self.label_angle_upperbody_data.setText(f"{joint_data:.3f}")  # 上身角度
                    self.upperbody_angle = joint_data  # 保存到属性
                    self.data_save[1] = joint_data
                elif device_number == 3:
                    self.label_angle_left_hip_data.setText(f"{-joint_data:.3f}")  # 左髋角度
                    self.data_save[2] = -joint_data
                elif device_number == 6:
                    self.label_angle_right_hip_data.setText(f"{-joint_data:.3f}")  # 右髋角度
                    self.data_save[5] = -joint_data

            elif device_number in [4, 5, 7, 8]:
                label_name1 = f"label_origin_data_imu{device_number - 2}{data_type}_data"
                imu_data1 = float(getattr(self, label_name1).text())
                label_name2 = f"label_origin_data_imu{device_number - 1}{data_type}_data"
                imu_data2 = float(getattr(self, label_name2).text())
                joint_data = -(imu_data1 + self.comp_data[device_number - 3]) + (
                        imu_data2 + self.comp_data[device_number - 2])
                if device_number == 4:
                    self.label_angle_left_knee_data.setText(f"{joint_data:.3f}")  # 左膝角度
                    self.data_save[3] = round(joint_data, 6)
                elif device_number == 5:
                    self.label_angle_left_ankle_data.setText(f"{joint_data:.3f}")  # 左踝角度
                    self.data_save[4] = round(joint_data, 6)
                elif device_number == 7:
                    self.label_angle_right_knee_data.setText(f"{joint_data:.3f}")  # 右膝角度
                    self.data_save[6] = round(joint_data, 6)
                elif device_number == 8:
                    self.label_angle_right_ankle_data.setText(f"{joint_data:.3f}")  # 右踝角度
                    self.data_save[7] = round(joint_data, 6)

        # 只处理x数据类型的设备2-8
        if data_type == 'x' and device_number in range(2, 9):
            self.update_joint_angle_data(device_number, joint_data)

    def update_label_origin_angle_data_latest(self, data_list):
        """
        批量处理IMU数据（更快版）：
        - 直接用内存缓存 _latest_ang 做关节角计算，避免从 QLabel.text() 读回
        - 电量用 EMA（O(1)）平滑
        - 其它：仍然更新原始数据标签 & data_save 索引
        """
        process_start = time.perf_counter()

        # ===== 1) 先把这批“每设备最新三轴角度”写进缓存 =====
        for data in data_list:
            d = data['device']
            ang = data['ang']  # [ax, ay, az]
            # 覆盖缓存
            if isinstance(ang, (list, tuple)):
                self._latest_ang[d] = [float(ang[0]), float(ang[1]), float(ang[2])]
            else:
                # 兼容 numpy 数组
                self._latest_ang[d] = [float(ang[0]), float(ang[1]), float(ang[2])]

        # ===== 2) 批量更新“原始三轴角/角速/加速度”的标签 & data_save =====
        # 说明：这里仍按你原有的 mapping 与 data_save 布局写入，确保其它功能不受影响
        for data in data_list:
            device_number = data['device']
            ang_data = data['ang']
            vel_data = data['vel']
            acc_data = data['acc']
            battery_value_raw = float(data['battery'])
            #seq=data["seq"]
            # if device_number==2:
            #     print(f"2号索引{seq}")
            # --- 原始三轴角度标签（一次性 3 个 setText；不要再在后面读回这些标签参与计算）
            for i, axis in enumerate(['x', 'y', 'z']):
                label_name = f"label_origin_data_imu{device_number - 1}{axis}_data"
                getattr(self, label_name).setText(f"{ang_data[i]:.3f}")

                # 同步写 data_save（保持你原先的索引规则）
                data_type = axis           # 'x'/'y'/'z'
                value_type = 'ang'         # 角度
                i_index = self.mapping1[data_type]   # 1/2/3
                j_index = self.mapping2[value_type]  # 1
                index = 9 * (device_number - 2) + 3 * (i_index - 1) + j_index + 7
                self.data_save[index] = round(float(ang_data[i]), 6)

            # --- 角速度
            for i, axis in enumerate(['x', 'y', 'z']):
                data_type = axis
                value_type = 'vel'
                i_index = self.mapping1[data_type]
                j_index = self.mapping2[value_type]  # 2
                index = 9 * (device_number - 2) + 3 * (i_index - 1) + j_index + 7
                self.data_save[index] = round(float(vel_data[i]), 6)

            # --- 加速度
            for i, axis in enumerate(['x', 'y', 'z']):
                data_type = axis
                value_type = 'acc'
                i_index = self.mapping1[data_type]
                j_index = self.mapping2[value_type]  # 3
                index = 9 * (device_number - 2) + 3 * (i_index - 1) + j_index + 7
                self.data_save[index] = round(float(acc_data[i]), 6)
            #self.data_save[69+device_number]=seq
            # ===== 3) 关节角：只在 X 轴时触发更新（调用快路径，见下一个改动点） =====
            # 说明：你原先就是只在 x 轴时调用，这里保持一致
            self.update_label_joint_angle_data_fast('x', device_number)

            # ===== 4) 电量：EMA 平滑 + 变化阈值 3% 才刷新 UI =====
            ema = self._bat_ema.get(device_number)
            alpha = 0.1  # 平滑系数，可 0.05~0.2
            ema = battery_value_raw if ema is None else (1 - alpha) * ema + alpha * battery_value_raw
            self._bat_ema[device_number] = ema

            disp = int(round(ema))
            if abs(disp - self.last_displayed_battery[device_number]) >= 3:
                getattr(self, f"label_imu{device_number}_battery_data").setText(f"{disp}%")
                self.last_displayed_battery[device_number] = disp

            # 低电量状态指示
            if disp < 20:
                self.update_device_status(device_number, "low_battery")
            else:
                self.update_device_status(device_number, "connected")

        # ===== 5) 批量处理结束后更新一次时间显示 =====
        time_data = round(time.perf_counter() - self.start_time, 3)
        self.label_current_time_data.setText(f"{time_data:.3f}")
        self.data_save[0] = time_data
        if self.rec_started:
            row = list(self.data_save)
            try:
                self.record_queue.put(row, timeout=0.05)
            except queue.Full:
                self.rec_dropped += 1
            # 调试：超过 10ms 打印
        cost_ms = (time.perf_counter() - process_start) * 1000
        if cost_ms > 10:
            print(f"[UI] batch={len(data_list)} cost={cost_ms:.2f}ms")

    def update_label_joint_angle_data_fast(self, data_type, device_number):
        """
        快路径：用 _latest_ang 缓存直接计算关节角，不从 QLabel 读回。
        仅处理 data_type='x' 的逻辑（和你现有一致）。
        """
        if data_type != 'x':
            return

        # 取缓存角（°）
        get_ang = lambda dev: self._latest_ang.get(dev, [0.0, 0.0, 0.0])[0]  # 只要 X 轴

        # 上身/左右髋（来自 2,3,6）
        if device_number in (2, 3, 6):
            imu_x = get_ang(device_number)
            joint = imu_x + self.comp_data[device_number - 2]
            if device_number == 2:
                # self.label_angle_upperbody_data.setText(f"{joint:.3f}")
                self.upperbody_angle = joint
                self.data_save[1] = joint
            elif device_number == 3:
                self.label_angle_left_hip_data.setText(f"{-joint:.3f}")
                self.data_save[2] = -joint
            elif device_number == 6:
                self.label_angle_right_hip_data.setText(f"{-joint:.3f}")
                self.data_save[5] = -joint

        # 膝/踝：4(=3+4)、5(=4+5)、7(=6+7)、8(=7+8) 这样的成对组合
        elif device_number in (4, 5, 7, 8):
            # 注意：这里沿用你原有的组合公式
            imu1 = get_ang(device_number - 1)  # 上一节
            imu2 = get_ang(device_number)      # 当前节
            joint = -(imu1 + self.comp_data[device_number - 3]) + (imu2 + self.comp_data[device_number - 2])

            if device_number == 4:
                self.label_angle_left_knee_data.setText(f"{joint:.3f}")
                self.data_save[3] = round(joint, 6)
            elif device_number == 5:
                self.label_angle_left_ankle_data.setText(f"{joint:.3f}")
                self.data_save[4] = round(joint, 6)
            elif device_number == 7:
                self.label_angle_right_knee_data.setText(f"{joint:.3f}")
                self.data_save[6] = round(joint, 6)
            elif device_number == 8:
                self.label_angle_right_ankle_data.setText(f"{joint:.3f}")
                self.data_save[7] = round(joint, 6)
        # 只处理x数据类型的设备2-8
        if data_type == 'x' and device_number in range(2, 9):
            self.update_joint_angle_data(device_number, joint)

    #绘图槽函数
    def refresh_curves_slot(self, packet):
        """
        packet: list of (times, angles, joint_index)
        运行在主线程，但只做 setData，不处理复杂逻辑
        """
        for times, angles, idx in packet:
            if idx < len(self.joint_curves):
                self.joint_curves[idx].setData(times, angles, _callSync='off')

        # 如需要自动滚动 X 轴，可在主线程简单处理
        if packet:
            last_time = packet[-1][0][-1] if packet[-1][0] else 0
            if last_time:
                for curve in self.joint_curves:
                    vb = curve.getViewBox()
                    if vb:
                        vb.setXRange(max(0, last_time - 8), last_time, padding=0)
    
    def update_joint_angle_data(self, device_number, angle_value):
        """
        更新特定关节的绘图数据
        """
        # 检查关节角度图是否初始化成功
        if not hasattr(self, 'joint_curves') or not hasattr(self, 'joint_time_data') or not hasattr(self,'joint_angle_data'):
            return

        # 映射设备号到关节索引
        joint_index = self.device_to_joint.get(device_number)
        if joint_index is None:
            return

        # 获取当前时间
        current_time = time.perf_counter() - self.start_time

        # 更新时间队列
        if joint_index < len(self.joint_time_data):
            self.joint_time_data[joint_index].append(current_time)
        else:
            # 如果索引超出范围，扩展列表
            while len(self.joint_time_data) <= joint_index:
                self.joint_time_data.append([])
            self.joint_time_data[joint_index].append(current_time)

        # 更新该关节的角度数据
        if joint_index < len(self.joint_angle_data):
            self.joint_angle_data[joint_index].append(angle_value)
        else:
            # 如果索引超出范围，扩展列表
            while len(self.joint_angle_data) <= joint_index:
                self.joint_angle_data.append([])
            self.joint_angle_data[joint_index].append(angle_value)

    def update_joint_angle_plot(self):
        """更新关节角度图 - 每个关节独立更新"""
        # 确保所有必要的属性都存在
        if not hasattr(self, 'joint_curves') or not hasattr(self, 'joint_time_data') or not hasattr(self,
                                                                                                    'joint_angle_data'):
            return

        # 遍历所有关节
        for joint_index in range(len(self.joint_curves)):
            # 确保索引在有效范围内
            if joint_index >= len(self.joint_time_data) or joint_index >= len(self.joint_angle_data):
                continue

            # 检查是否有实际数据
            has_real_data = bool(self.joint_time_data[joint_index] and self.joint_angle_data[joint_index])

            # 如果没有实际数据，跳过这个关节的更新
            if not has_real_data:
                continue

            try:
                # 使用实际数据
                time_points = list(self.joint_time_data[joint_index])
                angle_points = list(self.joint_angle_data[joint_index])
                if not time_points or not angle_points:
                    continue

                # 获取最近的时间点
                last_time = time_points[-1]

                # 更新曲线数据
                self.joint_curves[joint_index].setData(
                    time_points,
                    angle_points
                )

                # 设置视图范围（始终显示最后8秒）
                view_min = max(0, last_time - 8)
                view_max = last_time

                # 获取视图框
                view_box = self.joint_curves[joint_index].getViewBox()
                if view_box:
                    view_box.setXRange(view_min, view_max, padding=0)

                    # 如果数据量足够（大于8秒），保持固定8秒窗口
                    if view_min > 0:
                        view_box.setLimits(xMin=view_min, xMax=view_max, minXRange=8, maxXRange=8)

            except Exception as e:
                print(f"更新关节角度图错误 (关节 {joint_index}): {str(e)}")

    def begin_record(self):
        """
        启动数据记录的计时器
        """
        self.rec_started = False
        self.record_queue = queue.Queue(maxsize=20000)
        self.record_start_time = time.perf_counter()
        self.writer_thread = None
        self.rec_dropped = 0                     # 队列满的丢弃计数（监控用）
        self.record_start_time = 0.0             # 本次记录的起始时刻（perf_counter）
        self.record_start_seq = {i: -1 for i in range(2, 9)}
        # ==== 新增：获取保存路径 ====
        save_path = self.lineEdit_save_path.text().strip()
        if not save_path:
            # 使用默认路径
            save_path = self.default_save_path
        self._writer_running = True

        # 获取当前日期和时间
        # 使用时间戳生成文件名
        current_datetime = datetime.datetime.now()
        filename = f"data_{current_datetime.strftime('%Y%m%d_%H%M%S')}.csv"

        # 组合完整文件路径
        file_path = os.path.join(save_path, filename)

        # 确保路径存在
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "路径错误",
                    f"无法创建保存目录: {str(e)}",
                    QtWidgets.QMessageBox.Ok
                )
                return

        try:
            # 记录数据用
            self.csv_file = None
            self.csv_writer = None
            self.filename = filename  # 保存文件名
            self.save_path = save_path  # 保存路径
            # ... 你的文件名/路径生成逻辑（省略） ...
            # 清零统计
            self.rec_dropped = 0

            # 启动“单一写盘线程”，让它在线程内部打开/写文件
            self.writer_thread = CsvWriterThread(
                path=file_path,              # 你的 CSV 路径
                fieldnames=self.fieldnames,       # 或 None（若你不用 header）
                q=self.record_queue,
                batch_ms=200,                     # 200ms 批量写，可调为100~300
                dict_mode=False                   # data_save 是 list → False
            )
            self.writer_thread.start()
            self.rec_started = True
            # ==== 新增：保存路径属性 ====
            # self.csv_file = open(file_path, 'w', newline='', encoding='utf-8-sig')  # 添加utf-8编码
            # self.csv_writer = csv.writer(self.csv_file)

            # # 添加姓名登记行（单字段）
            # # self.csv_writer.writerow(["姓名："])
            # self.csv_writer.writerow(self.fieldnames)

            # # 启动定时器
            # self.timer_record.start(self.refresh)

            # 更新按钮状态
            self.pushButton_begin_record.setEnabled(False)
            self.pushButton_end_record.setEnabled(True)

        except Exception as e:
            print(f"开始记录失败: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "记录错误",
                f"无法开始记录数据: {str(e)}",
                QtWidgets.QMessageBox.Ok
            )
            # 确保按钮状态正确
            self.pushButton_begin_record.setEnabled(True)
            self.pushButton_end_record.setEnabled(False)

    def stop_record_save(self):
        """结束数据记录并保存"""
        try:
            self.rec_started = False
            # 请求线程停止并等待收尾
            if self.writer_thread:
                self.writer_thread.stop()
                self.writer_thread.join(timeout=3.0)

                # 审计信息（可写到 .audit.json）
                rows = self.writer_thread.rows_written
                hiwat = self.writer_thread.high_watermark
                print(f"[RECORD] rows_written={rows} queue_high_watermark={hiwat} dropped={self.rec_dropped}")

                self.writer_thread = None
            # # 1. 停止记录定时器
            # self.timer_record.stop()
            # # 添加短暂延迟确保所有数据写入
            # time.sleep(0.1)

            # # 2. 正确关闭文件
            # if self.csv_file:
            #     try:
            #         # 确保所有数据都写入磁盘
            #         self.csv_file.flush()
            #         # 使用系统调用强制将数据写入磁盘
            #         if hasattr(self.csv_file, 'fileno'):
            #             import os
            #             os.fsync(self.csv_file.fileno())
            #     finally:
            #         # 无论如何都尝试关闭文件
            #         self.csv_file.close()

            # # 3. 显示原始数据保存成功的提示
            # 获取保存路径
            save_path = self.lineEdit_save_path.text().strip()
            if not save_path:
                save_path = self.default_save_path

            #构建完整的文件路径
            file_path = os.path.join(save_path, self.filename)

            # 显示提示框
            QtWidgets.QMessageBox.information(
                self,
                "记录完成",
                f"已结束记录，数据已保存至：\n{file_path}",
                QtWidgets.QMessageBox.Ok
            )

            # 4. 设置固定的步态参数值doubleSpinBox_leg_width
            thigh_length = 100*self.doubleSpinBox_thigh_length_data.value()
            shank_length = 100*self.doubleSpinBox_shank_length_data.value()
            step_width = 100*self.doubleSpinBox_leg_width.value()
            height=self.doubleSpinBox_height.value()
            weight=self.lineEdit.text()
            OA_Knee=self.comboBox_OAKnee.currentText()
            OA_Time=self.comboBox_OATime.currentText()
            OA_Shape=self.comboBox_OAShape.currentText()
            # 7. 更新按钮状态
            self.pushButton_begin_record.setEnabled(True)
            self.pushButton_end_record.setEnabled(False)
            # 6. 步态参数计算
            time_interval = 0.02

            # 构建原始数据文件的完整路径
            file_path = os.path.join(save_path, self.filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"原始数据文件不存在: {file_path}")

            cfg = PreprocessConfig(
                fs=50.0,                 # 你的原始采样率
                time_col="时间",           # 若CSV没有时间列就用None；有的话写列名，如 "time"
                drop_initial_seconds=0,# 丢弃起步0.5s（可选）
                baseline_frames=20,      # 去基线
                interpolate_nan=True,    # 插值
                resample_hz=None,        # 是否统一重采样，通常保持None
                unwrap_angle=True,      # 角度是否去包裹
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
            rawdata, rawheaders = read_csv_file(file_path)
            #重复数据处理，并插值
            data,headers=process_imu_joint_and_plantar_safe_ndarray(rawdata,rawheaders,interp_method="cubic")
            #建立保存文件夹
            result_dir=create_result_folder(file_path)
            processed_data,processed_headers=process_imu_joint_and_plantar_safe_ndarray(data,headers,cfg.fs)
            pre00_path = os.path.join(result_dir, os.path.basename(file_path).replace('.csv', '_pre00.csv'))
            save_csv_file(data,headers,pre00_path)
            #保存插值文件
            raw_df=pd.read_csv(pre00_path,encoding="utf-8-sig")
            #计算步态参数
            self.gait_paparemters=calculate_gait_parameters(raw_df,cfg,thigh_length,shank_length,step_width,plot=False)
            self.gait_paparemters["大腿长(cm)"]=thigh_length
            self.gait_paparemters["小腿长(cm)"]=shank_length
            self.gait_paparemters["初始步宽(cm)"]=step_width
            self.gait_paparemters["身高(m)"]=height
            self.gait_paparemters["体重(kg)"]=weight
            self.gait_paparemters["膝关节炎位置"]=OA_Knee
            self.gait_paparemters["膝关节炎阶段"]=OA_Time
            self.gait_paparemters["膝关节力线"]=OA_Shape
            #左腿参数赋值
            self.label_left_swing_data.setText(str(round(self.gait_paparemters.get("左腿摆动相占比(%)",np.nan), 2)))
            self.label_left_support_data.setText(str(round(self.gait_paparemters.get("左腿支撑相占比(%)",np.nan), 2)))
            self.label_left_period_data.setText(str(round(self.gait_paparemters.get("左腿周期(s)",np.nan), 2)))
            self.label_left_step_length_data.setText(str(round(self.gait_paparemters.get("左腿步长(cm)",np.nan), 2)))
        
            #右腿参数赋值
            self.label_right_swing_data.setText(str(round(self.gait_paparemters.get("右腿摆动相占比(%)",np.nan), 2)))
            self.label_right_support_data.setText(str(round(self.gait_paparemters.get("右腿支撑相占比(%)",np.nan), 2)))
            self.label_right_period_data.setText(str(round(self.gait_paparemters.get("右腿周期(s)",np.nan), 2)))
            self.label_right_step_length_data.setText(str(round(self.gait_paparemters.get("右腿步长(cm)",np.nan), 2)))
           
           #总体参数
            self.label_left_step_width_data.setText(str(round(step_width, 2)))
            self.label_step_width_data.setText(str(round(self.gait_paparemters.get("步宽(cm)",np.nan), 2)))
            self.label_left_step_speed_data.setText(str(round(self.gait_paparemters.get("平均步速(cm/s)",np.nan), 2)))
            self.label_right_stride_length_data.setText(str(round(self.gait_paparemters.get("步幅(cm)",np.nan), 2)))
            self.label_average_period_data.setText(str(round(self.gait_paparemters.get("步频(步/s)",np.nan), 2)))
            avg_step_len = 0.5 * (
                self.gait_paparemters.get("左腿步长(cm)", np.nan)
                + self.gait_paparemters.get("右腿步长(cm)", np.nan)
            )
            self.label_step_length_data.setText(str(round(avg_step_len,2)))
           
            gait_params_filename = f"gait_params_output.csv"
            gait_params_path = os.path.join(result_dir, gait_params_filename)
            gait_params_df = pd.DataFrame([self.gait_paparemters])
            gait_params_df.to_csv(gait_params_path, encoding="utf-8-sig",index=False)

        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"结束记录过程中出错: {str(e)}")
            print(err_msg)
            QtWidgets.QMessageBox.warning(
                self,
                "记录错误",
                f"结束记录时出错: {str(e)}",
                QtWidgets.QMessageBox.Ok
            )

    def record_data(self):
        """定时记录当前数据到 CSV"""
        if not self.rec_started:
            return
        t_now = round(time.perf_counter() - self.start_time, 3)
        self.data_save[0] = t_now
        #落盘
        self._rows_buffer.append(self.data_save.copy())
    
    def flush_buffer(self):
        if not getattr(self, "csv_writer", None) or not self._rows_buffer:
            return
        if getattr(self, "_flushing", False):
            return
        self._flushing = True
        try:
            rows = self._rows_buffer
            self._rows_buffer = []  # ⚠️ 原地“换桶”避免与采集竞争
            self.csv_writer.writerows(rows)
            self.csv_file.flush()
        finally:
            self._flushing = False

class imuThread(QThread):
    """
    线程类，向主程序发送 IMU 的设备编号、轴名称、值类型、值
    """
    update_label = pyqtSignal(list)  # 设备编号、轴名称、值
    # 批量信号：一次发所有设备的“最新一帧”
    #frames_ready = QtCore.pyqtSignal(list)

    def __init__(self, port, parent=None):
        super().__init__(parent)
        self._running = True
        self.port = port

    def stop(self):
        self._running = False

    def run(self):
        imu = BlueTeethIMU.imu(self.port)

        latest_by_dev = {}               # device_id -> frame dict
        next_emit = QtCore.QElapsedTimer()
        next_emit.start()
        emit_dt_ms = 10                  # 20ms 刷新一次 UI（≈50Hz）

        while self._running:
            imu.read()                   # 仍然 100Hz+ 读取
            if imu.read_flag == 4:
                d = imu.device_number
                latest_by_dev[d] = {
                    "device": d,
                    "ang": imu.data_ang[d],   # [ax, ay, az]
                    "vel": imu.data_vel[d],
                    "acc": imu.data_acc[d],
                    "battery": imu.battery[d],
                    "t": QtCore.QDateTime.currentMSecsSinceEpoch()/1000.0
                    #"seq": imu.seq[d]
                }

            # 到节拍就“批量”发一次
            if next_emit.elapsed() >= emit_dt_ms:
                if latest_by_dev:
                    self.update_label.emit(list(latest_by_dev.values()))
                    latest_by_dev.clear()
                next_emit.restart()

class PlotWorker(QThread):
    # 把处理好的 6 条曲线数据一次性发回主线程
    plot_data_ready = pyqtSignal(list)   # list 内元素: (times, angles, idx)

    def __init__(self,
                 joint_time_deques,
                 joint_angle_deques,
                 parent=None):
        super().__init__(parent)
        self.time_deques = joint_time_deques
        self.angle_deques = joint_angle_deques
        self._running = True
        self.period_ms = 200          # 绘图刷新周期（可改）

    def run(self):
        """子线程死循环：准备数据 -> 发信号"""
        while self._running:
            t0 = time.perf_counter()
            packet = []
            MAX_POINTS = 500
            for idx in range(6):
                if not self.time_deques[idx]:
                    continue
                # 转成 list，只保留后 MAX_POINTS 个
                times  = list(self.time_deques[idx])[-MAX_POINTS:]
                angles = list(self.angle_deques[idx])[-MAX_POINTS:]
                packet.append((times, angles, idx))

            if packet:
                self.plot_data_ready.emit(packet)

            # 精确休眠到下一周期
            elapsed = (time.perf_counter() - t0) * 1000
            QThread.msleep(max(1, self.period_ms - int(elapsed)))

    def stop(self):
        self._running = False
        self.wait()

def my_random(start, end):
    # 使用时间戳和计数器确保每次调用结果不同
    current_time = time.time_ns()
    if not hasattr(my_random, "counter"):
        my_random.counter = 0
    my_random.counter += 1

    # 简单但有效的随机算法
    seed = (current_time * my_random.counter) % 0xFFFFFFFF
    seed = (seed * 1103515245 + 12345) % 0xFFFFFFFF

    # 映射到指定范围
    return start + seed % (end - start + 1)

class CsvWriterThread(threading.Thread):
    """
    独立写盘线程：从 queue 中批量取行，定期 writerows + flush。
    - 只在该线程里打开/关闭文件，避免跨线程共享 file 对象。
    - 支持 stop() 安全收尾（drain 队列后再关文件）。
    """
    def __init__(self, path, fieldnames, q: queue.Queue, batch_ms=200, dict_mode=False):
        super().__init__(daemon=True)
        self.path = path
        self.fieldnames = fieldnames
        self.q = q
        self.batch_ms = batch_ms
        self.dict_mode = dict_mode
        self.stop_event = threading.Event()

        # 统计
        self.rows_written = 0
        self.high_watermark = 0

    def run(self):
        # 1) 线程内打开文件 & writer
        with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
            if self.dict_mode:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            else:
                writer = csv.writer(f)
                if self.fieldnames:
                    writer.writerow(self.fieldnames)

            last_flush = time.perf_counter()
            buffer = []

            while not self.stop_event.is_set():
                # 2) 批量从队列取数据（非阻塞 drain + 短暂阻塞补一条）
                try:
                    # 先尝试立刻拿一条（阻塞 ≤ batch_ms）
                    item = self.q.get(timeout=self.batch_ms/1000.0)
                    buffer.append(item)
                except queue.Empty:
                    pass

                # 再把当前已有的都一次性倒出来（非阻塞）
                while True:
                    try:
                        buffer.append(self.q.get_nowait())
                    except queue.Empty:
                        break

                # 3) 到了节拍或收到了数据 → 批量写
                now = time.perf_counter()
                if buffer and (now - last_flush) >= (self.batch_ms/1000.0):
                    if self.dict_mode:
                        writer.writerows(buffer)
                    else:
                        writer.writerows(buffer)
                    f.flush()  # 让 OS 缓存刷到文件
                    self.rows_written += len(buffer)
                    buffer.clear()
                    last_flush = now

                # 记录队列最高水位
                self.high_watermark = max(self.high_watermark, self.q.qsize())

            # 4) 收尾：drain 剩余项并 fsync 落地
            tail = []
            while True:
                try:
                    tail.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if tail:
                if self.dict_mode:
                    writer.writerows(tail)
                else:
                    writer.writerows(tail)
                self.rows_written += len(tail)
                f.flush()
                if hasattr(f, "fileno"):
                    os.fsync(f.fileno())

    def stop(self):
        self.stop_event.set()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    icopath=resource_path("images/logo_transpar.ico")
    app.setWindowIcon(QtGui.QIcon(":/pic/images/logo_transpar.ico"))
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())