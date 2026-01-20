from PyQt5 import QtWidgets
from mainwindow import MainWindow  # 导入您的主界面类
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # 直接创建并显示主窗口
    main_window = MainWindow()
    main_window.show()

    # 运行应用程序
    sys.exit(app.exec_())