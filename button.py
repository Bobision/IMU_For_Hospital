import serial.tools.list_ports
import serial
# --------------------------------------------第一部分：基础函数-------------------------------------------
def scan_ports(window):
    """
    搜索串口，添加到指定下拉选框
    :param window：窗口对象
    """
    available_ports = serial.tools.list_ports.comports()
    window.comboBox_imu_list.clear()
    for port in available_ports:
        window.comboBox_imu_list.addItem(port.device)

def open_port(comboBox):
    """
    根据下拉选框指定的串口号打开串口
    :param comboBox: 下拉选框，内容是指定的串口号
    """
    selected_port = comboBox.currentText()
    port = serial.Serial(selected_port, baudrate=460800, timeout=1)
    return port


def close_port(port):
    """
    关闭指定的串口
    :param port: 指定的串口号
    """
    port.close()

# ---------------------------------------------第二部分：事件函数-------------------------------------------
def open_port_action(window):
    """
    按钮事件，打开串口
    :param window：窗口对象
    """
    if window.serial_port is None:
        window.serial_port = open_port(getattr(window, f"comboBox_imu_list"))
        getattr(window, f"pushButton_imu_open").setEnabled(False)
        getattr(window, f"pushButton_imu_close").setEnabled(True)
        for index in range(1, 8):
            getattr(window, f"label_imu{index + 1}_status_color").setStyleSheet("background-color: rgb(255, 255, 0);")
            getattr(window, f"label_imu{index + 1}_status_text").setText("未连接")

def close_port_action(window):
    """
    按钮事件，关闭串口
    :param window：窗口对象
    """
    if window.serial_port is not None:
        close_port(window.serial_port)
        window.serial_port = None
        getattr(window, f"pushButton_imu_open").setEnabled(True)
        getattr(window, f"pushButton_imu_close").setEnabled(False)
        for index in range(1, 8):
            getattr(window, f"label_imu{index + 1}_status_color").setStyleSheet("background-color: rgb(160, 160, 160);")
            getattr(window, f"label_imu{index + 1}_status_text").setText("未打开")

def standard_data_action(window):
    """
    按钮事件，标定
    :param window：窗口对象
    """
    for i in range(7):
        window.comp_data[i] = 0 - float(getattr(window, f"label_origin_data_imu{i + 1}x_data").text())
        #getattr(window, f"label_origin_data_imu{i + 1}_comp_data").setText(str(round(window.comp_data[i], 6)))


# ---------------------------------------------第三部分：关联函数-------------------------------------------
def connect_button_actions(window):
    """
    关联函数，将按钮事件与按钮对象关联
    :param window：窗口对象
    """
    window.pushButton_scan_port.clicked.connect(lambda: scan_ports(window))
    # 初始化串口对象列表
    window.serial_port = None
    window.comp_data = [0.0] * 7
    window.record_flag = [0, 0]

    # 连接每个按钮的点击事件
    window.pushButton_imu_open.clicked.connect(lambda: open_port_action(window))   
    window.pushButton_imu_close.clicked.connect(lambda: close_port_action(window))
    window.pushButton_standard.clicked.connect(lambda: standard_data_action(window))
