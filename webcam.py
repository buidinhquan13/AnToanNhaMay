# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1480 ,800)


        # Main widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)

        # Webcam frame
        self.webcam_frame = QtWidgets.QLabel()
        self.webcam_frame.setStyleSheet("background-color: black;")
        self.webcam_frame.setFixedSize(1280 , 720)

        # Info panel
        self.info_panel = QtWidgets.QWidget()
        self.info_panel_layout = QtWidgets.QVBoxLayout(self.info_panel)
        self.info_label = QtWidgets.QLabel("Thông tin hiển thị tại đây")
        self.info_panel_layout.addWidget(self.info_label)
        self.info_panel.setStyleSheet("background-color: #DDDDDD;")

        # Add widgets to layout
        self.main_layout.addWidget(self.webcam_frame, 3)
        self.main_layout.addWidget(self.info_panel, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        # OpenCV webcam setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.info_label.setText("Không thể mở webcam!")

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam_frame)
        self.timer.start(30)  # 30ms per frame

        # Window setup
        MainWindow.setWindowTitle("Webcam Viewer with Info Panel")

    def update_webcam_frame(self):
        """Capture and display webcam frame"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (1280 , 720))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.webcam_frame.setPixmap(pixmap)
        else:
            self.webcam_frame.setText("Không thể truy cập webcam.")

    def closeEvent(self, event):
        """Cleanup before closing"""
        self.timer.stop()
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
