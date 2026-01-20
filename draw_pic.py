from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtGui import QPen, QColor, QBrush,QFont
from PyQt5.QtWidgets import QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsTextItem
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGraphicsLineItem
from PyQt5.QtCore import QPointF, Qt
from ui import Ui_MainWindow
from functools import partial
from math import cos, sin, radians
import pyqtgraph as pg
import numpy as np
from collections import deque



def init_graphics_view(main_window):
    # 获取 graphicsView 对象
    graphics_view = main_window.findChild(QGraphicsView, 'graphicsView')
    # 创建场景
    scene = QGraphicsScene(main_window)
    graphics_view.setScene(scene)
    # 设置视图的背景颜色
    graphics_view.setBackgroundBrush(Qt.white)
    # 启用抗锯齿渲染
    graphics_view.setRenderHint(QPainter.Antialiasing, True)
    # 在初始化时绘制坐标轴和刻度
    pre_axis(scene, main_window)
    # 创建定时器，每隔一段时间触发一次更新
    timer = QTimer(main_window)
    timer.timeout.connect(partial(pre_axis, scene, main_window))
    timer.start(50)  # 每50毫秒更新一次
    return scene, graphics_view, timer


def pre_axis(scene, main_window):
    # 清除之前的点和线
    for item in scene.items():
        if isinstance(item, (QGraphicsEllipseItem, QGraphicsLineItem)):
            scene.removeItem(item)

    zero_point = QPointF(0, 0)  # 初始位置

    # 添加一个黑圆
    add_circle_to_scene(scene, zero_point)

    # 初始位置
    start_point = QPointF(0, 0)  # 初始位置，髋关节角度
    # 各关节角度
    angle_upperbody = float(main_window.upperbody_angle) - 90 # 上肢角度
    angle_left_hip = - float(main_window.label_angle_left_hip_data.text()) + 90  # 左髋关节角度
    angle_left_knee = - float(main_window.label_angle_left_hip_data.text()) + \
                      float(main_window.label_angle_left_knee_data.text()) + 90  # 左膝关节角度
    angle_left_ankle = - float(main_window.label_angle_left_hip_data.text()) + \
                       float(main_window.label_angle_left_ankle_data.text()) + \
                       float(main_window.label_angle_left_knee_data.text())  # 左踝关节角度
    # angle_left_ankle = 0
    angle_right_hip = - float(main_window.label_angle_right_hip_data.text()) + 90  # 右髋关节角度
    angle_right_knee = float(main_window.label_angle_right_knee_data.text()) - \
                       float(main_window.label_angle_right_hip_data.text()) + 90  # 右膝关节角度
    angle_right_ankle = - float(main_window.label_angle_right_hip_data.text()) + \
                        float(main_window.label_angle_right_ankle_data.text()) + \
                        float(main_window.label_angle_right_knee_data.text())  # 右踝关节角度
    # 各部分长度
    length_upperbody = 80  # 上肢长度
    length_thigh = 70  # 大腿长度
    length_shank = 50  # 大腿长度
    length_foot = 20  # 大腿长度
    # 计算各关节点的位置
    point_upperbody = calculate_end_point(start_point, length_upperbody, angle_upperbody)  # 上肢末端位置
    point_left_knee = calculate_end_point(start_point, length_thigh, angle_left_hip)  # 左膝位置
    point_right_knee = calculate_end_point(start_point, length_thigh, angle_right_hip)  # 右膝位置
    point_left_ankle = calculate_end_point(point_left_knee, length_shank, angle_left_knee)  # 左踝位置
    point_right_ankle = calculate_end_point(point_right_knee, length_shank, angle_right_knee)  # 右踝位置
    point_left_foot = calculate_end_point(point_left_ankle, length_foot, angle_left_ankle)  # 左脚末端位置
    point_right_foot = calculate_end_point(point_right_ankle, length_foot, angle_right_ankle)  # 左脚末端位置
    # 画线
    add_line_to_scene(scene, start_point, point_upperbody, Qt.green)  # 画上肢
    add_line_to_scene(scene, start_point, point_left_knee, Qt.blue)  # 画左大腿
    add_line_to_scene(scene, start_point, point_right_knee, Qt.yellow)  # 画右大腿
    add_line_to_scene(scene, point_left_knee, point_left_ankle, Qt.blue)  # 画左小腿
    add_line_to_scene(scene, point_right_knee, point_right_ankle, Qt.yellow)  # 画右小腿
    add_line_to_scene(scene, point_left_ankle, point_left_foot, Qt.blue)  # 画左脚
    add_line_to_scene(scene, point_right_ankle, point_right_foot, Qt.yellow)  # 画右脚
    # 画关节圆
    add_circle_to_scene(scene, zero_point)
    add_circle_to_scene(scene, point_upperbody)
    add_circle_to_scene(scene, point_left_knee)
    add_circle_to_scene(scene, point_right_knee)
    add_circle_to_scene(scene, point_left_ankle)
    add_circle_to_scene(scene, point_right_ankle)
    add_circle_to_scene(scene, point_left_foot)
    add_circle_to_scene(scene, point_right_foot)


def calculate_end_point(start_point, length, angle):
    end_x = start_point.x() + length * cos(radians(angle))
    end_y = start_point.y() + length * sin(radians(angle))
    return QPointF(end_x, end_y)


def add_line_to_scene(scene, start_point, end_point, color=Qt.black, line_width=5):
    line = QGraphicsLineItem(start_point.x(), start_point.y(), end_point.x(), end_point.y())
    pen = QPen(color)
    pen.setWidth(line_width)
    line.setPen(pen)
    scene.addItem(line)


def add_circle_to_scene(scene, center_point, radius=10, color=Qt.black):
    circle = QGraphicsEllipseItem(center_point.x() - radius / 2, center_point.y() - radius / 2, radius, radius)
    circle.setBrush(QBrush(color))
    scene.addItem(circle)


def init_joint_angle_plot(main_window):
    """初始化关节角度实时曲线图（每个关节独立图表）"""
    # 定义关节配置
    joints_config = [
        {'name': '左腿髋关节', 'color': '#4CAF50', 'view': 'graphicsView_left_hip'},
        {'name': '左腿膝关节', 'color': '#2196F3', 'view': 'graphicsView_left_knee'},
        {'name': '左腿踝关节', 'color': '#FFEB3B', 'view': 'graphicsView_left_ankle'},
        {'name': '右腿髋关节', 'color': '#9C27B0', 'view': 'graphicsView_right_hip'},
        {'name': '右腿膝关节', 'color': '#00BCD4', 'view': 'graphicsView_right_knee'},
        {'name': '右腿踝关节', 'color': '#F44336', 'view': 'graphicsView_right_ankle'}
    ]

    curves = []
    time_data_list = []
    angle_data_list = []
    plot_widgets = []
    proxy_items = []  # 存储代理项，用于自适应调整大小
    zero_lines = []  # 存储0度参考线

    for joint in joints_config:
        # 获取对应的graphicsView对象
        graphics_view = getattr(main_window, joint['view'])
        if graphics_view is None:
            print(f"无法找到 {joint['view']}")
            continue

        # ==== 关键改进1：设置滚动条策略为始终关闭 ====
        graphics_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        graphics_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # ==== 关键改进2：设置视图尺寸策略为扩展 ====
        graphics_view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # 创建PyQtGraph PlotWidget
        plot_widget = pg.PlotWidget()

        # ==== 禁用鼠标交互 ====
        # 禁用所有鼠标交互（包括缩放和平移）
        plot_widget.setMouseEnabled(x=False, y=False)

        # 禁用滚轮缩放
        plot_widget.wheelEvent = lambda event: None
        # 设置背景为白色
        plot_widget.setBackground('w')

        # 获取PlotItem对象
        plot_item = plot_widget.getPlotItem()

        # ==== 关键改进3：设置图表尺寸策略为扩展 ====
        plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # ==== 添加0度参考线（粗横轴）====
        # 创建0度参考线（粗横轴）
        zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=1))
        plot_item.addItem(zero_line)
        zero_lines.append(zero_line)

        # 创建底部和左侧的AxisItem，确保它们在原点相交
        plot_item.setAxisItems({'bottom': pg.AxisItem(orientation='bottom'),
                                'left': pg.AxisItem(orientation='left')})

        # 设置坐标范围（根据图片）
        plot_item.setXRange(0, 5)  # 横轴范围 0-5秒
        plot_item.setYRange(-181, 181)  # 纵轴范围 -180~180度

        # 设置网格
        plot_item.showGrid(x=True, y=True, alpha=0.2)

        # 设置坐标轴标签
        plot_item.setLabel('left', '角度 (°)', **{'color': '#000000', 'size': '8pt'})
        plot_item.setLabel('bottom', '时间/s', **{'color': '#000000', 'size': '8pt'})

        # 设置坐标轴线样式
        plot_item.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot_item.getAxis('left').setPen(pg.mkPen(color='k', width=1))

        # 创建曲线
        pen = pg.mkPen(
            color=joint['color'],
            width=2
        )
        curve = plot_widget.plot(pen=pen, name=joint['name'])

        # 配置视图
        scene = QtWidgets.QGraphicsScene()
        proxy = scene.addWidget(plot_widget)

        # ==== 正确设置代理项位置和大小 ====
        # 获取视图大小
        view_width = graphics_view.width()
        view_height = graphics_view.height()

        # 设置代理项位置和大小（填满视图）
        proxy.setPos(0, 0)
        proxy.setGeometry(QtCore.QRectF(0, 0, view_width, view_height))

        # 设置视图场景
        graphics_view.setScene(scene)

        # 存储代理项
        proxy_items.append(proxy)

        # 初始化数据存储
        time_data_list.append(deque(maxlen=500))
        angle_data_list.append(deque(maxlen=500))

        # 存储曲线对象
        curves.append(curve)
        plot_widgets.append(plot_widget)

    # 创建定时器用于更新曲线
    timer = QtCore.QTimer()

    # 连接窗口大小变化信号
    def handle_resize_event(event):
        """处理窗口大小变化事件，调整图表大小"""
        # 调用父类方法
        QtWidgets.QMainWindow.resizeEvent(main_window, event)

        # 更新所有代理项的大小
        for i, joint in enumerate(joints_config):
            graphics_view = getattr(main_window, joint['view'])
            if graphics_view and i < len(proxy_items):
                # 获取视图大小
                view_width = max(100, graphics_view.width())
                view_height = max(100, graphics_view.height())

                # 更新代理项几何位置（填满视图）
                proxy_items[i].setGeometry(QtCore.QRectF(0, 0, view_width, view_height))

                # 强制重绘图表
                plot_widget = proxy_items[i].widget()
                if plot_widget:
                    plot_widget.update()

    main_window.resizeEvent = handle_resize_event

    # ==== 关键改进2：全局窗口大小变化监听 ====
    # 创建全局窗口大小变化监听器
    class GlobalResizeFilter(QtCore.QObject):
        resized = QtCore.pyqtSignal()

        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Resize:
                self.resized.emit()
            return super().eventFilter(obj, event)

    # 安装事件过滤器
    resize_filter = GlobalResizeFilter()
    main_window.installEventFilter(resize_filter)

    # 连接全局大小变化信号
    resize_filter.resized.connect(lambda: handle_resize_event(None))

    # ==== 关键改进3：标签页切换时强制调整 ====
    # 获取主窗口的tabWidget
    tab_widget = main_window.findChild(QtWidgets.QTabWidget, 'tabWidget')
    if tab_widget:
        # 连接标签页切换信号
        tab_widget.currentChanged.connect(
            lambda index: QtCore.QTimer.singleShot(50, lambda: handle_resize_event(None))
        )

    # ==== 关键改进4：立即调用一次自适应函数 ====
    # 在初始化后立即调用一次自适应函数
    QtCore.QTimer.singleShot(100, lambda: handle_resize_event(None))

    return curves, time_data_list, angle_data_list, timer