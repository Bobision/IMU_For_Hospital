import sys
import time
import threading
import serial
import pyqtgraph as pg
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt


class IMURealtimeMonitor(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle("IMU角度监测")
        self.setGeometry(100, 100, 1000, 700)

        # 设置全局样式 - 黑底白字
        self.setStyleSheet("""
            background-color: #000000; 
            color: #FFFFFF;
            font-family: Arial, sans-serif;
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        self.setLayout(main_layout)

        # === 标题区域 ===
        title_label = QLabel("IMU角度实时监测")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #FFFFFF; padding: 10px;")
        main_layout.addWidget(title_label)

        # === 状态信息区域 ===
        # 创建状态信息区域
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setSpacing(5)

        # 串口状态
        self.serial_status = QLabel("串口状态: 正在连接...")
        self.serial_status.setFont(QFont("Arial", 12))

        # 数据统计
        self.data_counter = QLabel("接收数据: 0 | 设备2包: 0 | 设备3包: 0")
        self.data_counter.setFont(QFont("Arial", 10))

        # 当前角度显示
        self.angle_display = QLabel("上身角度(2): --° | 大腿角度(3): --°")
        self.angle_display.setFont(QFont("Arial", 12))
        self.angle_display.setStyleSheet("background-color: #222222; padding: 8px;")

        status_layout.addWidget(self.serial_status)
        status_layout.addWidget(self.data_counter)
        status_layout.addWidget(self.angle_display)

        main_layout.addWidget(status_widget)

        # === 绘图区域 ===
        # 创建绘图控件
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("k")

        # 创建绘图对象
        self.plot = self.plot_widget.addPlot(title="角度变化曲线")
        self.plot.setLabel('left', '角度 (°)', color='w')
        self.plot.setLabel('bottom', '时间 (秒)', color='w')
        self.plot.setTitle("角度变化曲线", color="w", size="12pt")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setYRange(-100, 100)
        self.plot.getAxis('left').setPen('w')
        self.plot.getAxis('bottom').setPen('w')

        # 将枚举值转为整数传递
        dash_style_int = int(Qt.DashLine)  # 关键转换
        # 创建曲线（红色为设备2 - 上身，蓝色为设备3 - 左大腿）
        # 修改画笔定义代码
        pen_upper = pg.mkPen(
            color=(255, 50, 50, 200),
            width=2,
            style=Qt.PenStyle.DashLine  # 使用Qt原生枚举
        )

        pen_thigh = pg.mkPen(
            color=(50, 150, 255, 200),
            width=2,
            style=Qt.PenStyle.DashLine  # 用字符串表示虚线
        )

        self.curve_upper = self.plot.plot(pen=pen_upper, name='上身角度(设备2)')
        self.curve_thigh = self.plot.plot(pen=pen_thigh, name='左大腿角度(设备3)')

        # 添加图例
        self.plot.addLegend()

        # 将绘图控件添加到布局
        main_layout.addWidget(self.plot_widget, 1)  # 1表示扩展比例

        # 数据存储
        self.time_data = deque(maxlen=500)  # 时间数据队列
        self.upper_data = deque(maxlen=500)  # 上身角度数据队列
        self.thigh_data = deque(maxlen=500)  # 左大腿角度数据队列

        # 启动时间
        self.start_time = time.time()

        # 数据统计
        self.byte_count = 0
        self.packet_count = 0
        self.device2_count = 0
        self.device3_count = 0

        # 初始化串口
        self.init_serial()

        # 启动定时器更新UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)  # 50ms更新一次 (20Hz)

        # 自动添加一些模拟数据
        self.add_demo_data()

    def init_serial(self):
        """初始化串口连接"""
        self.serial_status.setText("串口状态: 正在连接...")

        # 使用单独的线程进行串口连接尝试
        threading.Thread(target=self.connect_serial_thread, daemon=True).start()

    def connect_serial_thread(self):
        """在后台线程中尝试串口连接"""
        try:
            # 尝试连接串口
            self.ser = serial.Serial(
                port='COM8',
                baudrate=460800,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )

            if self.ser.is_open:
                self.serial_status.setText("✅ 串口连接成功: COM8@460800")
                self.serial_status.setStyleSheet("color: #4CAF50;")

                # 启动串口数据读取线程
                threading.Thread(target=self.serial_read_thread, daemon=True).start()
            else:
                self.serial_status.setText("❌ 串口连接失败: 端口未打开")
                self.serial_status.setStyleSheet("color: #F44336;")

        except serial.SerialException as e:
            self.serial_status.setText(f"❌ 串口连接失败: {str(e)}")
            self.serial_status.setStyleSheet("color: #F44336;")

        except Exception as e:
            self.serial_status.setText(f"❌ 连接错误: {str(e)}")
            self.serial_status.setStyleSheet("color: #F44336;")

    def serial_read_thread(self):
        """串口数据读取线程"""
        # 状态机变量
        state = 0  # 0=初始状态, 1=设备号已读, 2=包头已读, 3=标志位已读
        device = 0
        data_list = []

        while getattr(self, 'ser', None) and self.ser.is_open:
            try:
                # 读取可用数据
                data = self.ser.read(self.ser.in_waiting or 1)
                if not data:
                    time.sleep(0.01)  # 避免高CPU占用
                    continue

                # 处理每个字节
                for byte in data:
                    self.byte_count += 1

                    # 状态机处理
                    if state == 0:  # 等待设备号
                        if byte == 0x02 or byte == 0x03:  # 设备2或3
                            device = byte
                            state = 1
                        continue

                    elif state == 1:  # 等待0x55
                        if byte == 0x55:
                            state = 2
                        else:
                            state = 0  # 重置状态
                        continue

                    elif state == 2:  # 等待0x61
                        if byte == 0x61:
                            state = 3
                            data_list = []  # 准备接收数据
                        else:
                            state = 0  # 重置状态
                        continue

                    elif state == 3:  # 接收26字节数据
                        data_list.append(byte)
                        if len(data_list) >= 26:  # 已接收完整包
                            # 处理完整数据包
                            self.process_packet(device, data_list)

                            # 更新统计
                            if device == 0x02:
                                self.device2_count += 1
                            elif device == 0x03:
                                self.device3_count += 1

                            self.packet_count += 1

                            # 重置状态
                            state = 0
                            data_list = []

            except Exception as e:
                print(f"读取错误: {e}")
                time.sleep(0.1)

    def process_packet(self, device, data):
        """处理接收到的数据包"""
        try:
            # 解析角度数据
            # 简化方法 - 实际中应该使用更精确的解析方法
            angle = int.from_bytes(data[18:20], byteorder='little', signed=True)
            angle_degrees = angle * (180.0 / 32768.0)

            # 获取当前时间（从启动开始计算）
            current_time = time.time() - self.start_time

            # 更新数据
            self.time_data.append(current_time)
            if device == 0x02:  # 设备2 - 上身角度
                self.upper_data.append(angle_degrees)
            elif device == 0x03:  # 设备3 - 左大腿角度
                self.thigh_data.append(angle_degrees)

            # 更新角度显示
            upper_value = self.upper_data[-1] if self.upper_data else 0
            thigh_value = self.thigh_data[-1] if self.thigh_data else 0

            # 应用颜色增强显示
            upper_color = "#FF5252" if abs(upper_value) > 30 else "#FFFFFF"
            thigh_color = "#42A5F5" if abs(thigh_value) > 30 else "#FFFFFF"

            self.angle_display.setText(
                f"上身角度(2): <span style='color:{upper_color}'>{upper_value:.1f}°</span> | "
                f"大腿角度(3): <span style='color:{thigh_color}'>{thigh_value:.1f}°</span>"
            )

        except Exception as e:
            print(f"处理数据包错误: {e}")

    def add_demo_data(self):
        """添加一些模拟数据用于演示（当没有真实数据时）"""
        for i in range(30):
            t = i * 0.05
            self.time_data.append(t)
            self.upper_data.append(30 * abs(t - 1.5) - 25)  # 上身角度（正弦波）
            self.thigh_data.append(40 * abs(t - 1.0) - 20)  # 大腿角度（余弦波）

    def update_ui(self):
        """更新UI显示"""
        # 更新曲线数据
        if self.time_data and self.upper_data and self.thigh_data:
            self.curve_upper.setData(list(self.time_data), list(self.upper_data))
            self.curve_thigh.setData(list(self.time_data), list(self.thigh_data))

            # 自动滚动视图（显示最近10秒数据）
            if self.time_data[-1] > 10:
                self.plot.setXRange(self.time_data[-1] - 10, self.time_data[-1])

        # 更新数据统计
        self.data_counter.setText(
            f"接收数据: {self.byte_count} | 设备2包: {self.device2_count} | 设备3包: {self.device3_count}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    window = IMURealtimeMonitor()
    window.show()
    sys.exit(app.exec_())