
import cv2
import sys
import pyzed.sl as sl
import numpy as np
import math
import ogl_viewer.viewer as gl
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
import cv_viewer.tracking_viewer as cv_viewer

class ZEDApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_zed()
        

    def init_ui(self):
        self.setWindowTitle("ZED Tracking Viewer")
        self.setFixedSize(1480, 900)

        # Main widget
        self.centralwidget = QtWidgets.QWidget(self)
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)

        self.webcam_frame = QtWidgets.QLabel()
        self.webcam_frame.setStyleSheet("background-color: black;")
        self.webcam_frame.setFixedSize(1280, 720)
        # self.webcam_frame.resize(1280, 720)
        self.main_layout.addWidget(self.webcam_frame, 3)
        self.main_layout.addStretch(1)
        
        # self.webcam_frame.move(0, 0)
        # Thêm ô nhập bán kính và đặt nó tại (1380, 200)
        self.number_input_1 = QtWidgets.QLineEdit(self)
        self.number_input_1.setValidator(QtGui.QIntValidator())  # Chỉ nhập số nguyên
        self.number_input_1.move(1300, 130)  # Di chuyển ô nhập đến vị trí (1380, 200)
        # Set the label for radius
        self.radius_label = QtWidgets.QLabel("Nhập bán kính:", self)
        self.radius_label.move(1300, 100)
        

        self.select_box = QtWidgets.QComboBox(self.centralwidget)
        self.select_box.addItems(["Working", "Robot"])  # Add items to the combobox
        self.select_box.move(1300, 190)
        # Add widgets to layout
        
        #self.main_layout.addWidget(self.info_panel, 1)
        self.main_layout.addStretch(1)
        self.setCentralWidget(self.centralwidget)

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_zed_frame)
        self.timer.start(30)
    


    def init_zed(self):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if self.zed.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            self.info_label.setText("Failed to open ZED camera!")
            return
        
        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.body_format = sl.BODY_FORMAT.BODY_18
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        self.zed.enable_body_tracking(self.body_param)
        
        
        
        
        
        self.model = YOLO("models/best.pt")
        self.bodies = sl.Bodies()
        self.image = sl.Mat()
        
        self.nonlocal_variables = {
            'person_in_robot': 0,
            'person_in_working': 0,
            'vest_in_robot': 0,
            'vest_in_working': 0,
            'hel_in_robot': 0,
            'hel_in_working': 0,
            'is_safe_working': True,
            'is_safe_robot': True
        }

        # self.points_area_robot = np.array([[640, 370], [830, 370], [1150, 720], [640, 720]], dtype=np.int32)
        # self.points_area_working = np.array([[445, 370], [640, 370], [640, 720], [320, 720]], dtype=np.int32)
        self.points_area_robot = np.array([[640, 370], [830, 370], [1150, 720], [640, 720]], dtype=np.int32) 
        self.points_area_working = np.array([[445,370],[640,370],[640,720],[320,720]], dtype=np.int32) 
        
        # Tính toán các tham số cho khu vực làm việc (working area)
        self.a1_working = (self.points_area_working[0][1] - self.points_area_working[1][1]) / (self.points_area_working[0][0] - self.points_area_working[1][0])
        self.b1_working = self.points_area_working[0][1] - self.a1_working * self.points_area_working[0][0]

        self.a2_working = (self.points_area_working[2][1] - self.points_area_working[3][1]) / (self.points_area_working[2][0] - self.points_area_working[3][0])
        self.b2_working = self.points_area_working[2][1] - self.a2_working * self.points_area_working[2][0]

        # Tính toán các tham số cho khu vực robot (robot area)
        self.a1_robot = (self.points_area_robot[0][1] - self.points_area_robot[1][1]) / (self.points_area_robot[0][0] - self.points_area_robot[1][0])
        self.b1_robot = self.points_area_robot[0][1] - self.a1_robot * self.points_area_robot[0][0]

        self.a2_robot = (self.points_area_robot[2][1] - self.points_area_robot[3][1]) / (self.points_area_robot[2][0] - self.points_area_robot[3][0])
        self.b2_robot = self.points_area_robot[2][1] - self.a2_robot * self.points_area_robot[2][0]


    def calculate_line(self, points, indices):
        return (points[indices[0]][1] - points[indices[1]][1]) / (points[indices[0]][0] - points[indices[1]][0])

    def check_inside_working(self, h_p, w_p):
        return 1 if w_p > 350 and w_p < 640 else 0

    def update_zed_frame(self):
        camera_info = self.zed.get_camera_information()
    # 2D viewer utilities
        display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
        image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                    , display_resolution.height / camera_info.camera_configuration.resolution.height]
        
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_bodies(self.bodies, sl.BodyTrackingRuntimeParameters())

            image_left_ocv = self.image.get_data()
            results = self.model(image_left_ocv[:, :, :3], classes=[1, 2], conf=0.7)

            for result in results:
                boxes = result.boxes
                for box, class_id in zip(boxes.xyxy, boxes.cls):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = f"{['Person', 'Vest', 'Helmet'][int(class_id)]}"
                    cv2.putText(image_left_ocv, label, (x1 + 20, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 100), 2)
                    cv2.rectangle(image_left_ocv, (x1, y1), (x2, y2), (255, 50, 0), 2)

                    if class_id == 1 and self.check_inside_working(y2, x2):
                        self.nonlocal_variables['vest_in_working'] += 1
                    if class_id == 2 and self.check_inside_working(y2, x2):
                        self.nonlocal_variables['hel_in_working'] += 1

            # Update the call to render_2D with the missing argument `b2_robot`
            self.nonlocal_variables['person_in_working'], self.nonlocal_variables['person_in_robot'] = cv_viewer.render_2D(
                image_left_ocv,  # Image data to render
                # image_scale,  # img_scale (you can adjust this as needed)
                [1, 1],
                self.bodies.body_list,  # List of tracked bodies
                self.body_param.enable_tracking,  # Enable tracking flag
                self.body_param.body_format,  # Body tracking format
                self.a1_working, self.b1_working, self.a2_working, self.b2_working,  # Working area line equations
                self.a1_robot, self.b1_robot, self.a2_robot, self.b2_robot  # Robot area line equations
            )



            # Check safety conditions
            if (self.nonlocal_variables['person_in_working'] != self.nonlocal_variables['vest_in_working'] or
                self.nonlocal_variables['person_in_working'] != self.nonlocal_variables['hel_in_working']):
                self.nonlocal_variables['is_safe_working'] = False
            else:
                self.nonlocal_variables['is_safe_working'] = True

            if self.nonlocal_variables['person_in_robot'] > 0:
                self.nonlocal_variables['is_safe_robot'] = False

            # Draw safety areas
            image_left_ocv = cv_viewer.draw_area_working(image_left_ocv, self.points_area_working, self.nonlocal_variables['is_safe_working'])
            image_left_ocv = cv_viewer.draw_area_robot(image_left_ocv, self.points_area_robot, self.nonlocal_variables['is_safe_robot'])

            # Display the image
            image_left_ocv = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGR2RGB)
            cv2.imwrite('output_app.png',image_left_ocv)
            h, w, ch = image_left_ocv.shape
            bytes_per_line = ch * w
            qimg = QImage(image_left_ocv.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.webcam_frame.setPixmap(pixmap)

            # Reset counters for the next frame
            self.nonlocal_variables = {
                'person_in_robot': 0,
                'person_in_working': 0,
                'vest_in_robot': 0,
                'vest_in_working': 0,
                'hel_in_robot': 0,
                'hel_in_working': 0,
                'is_safe_working': True,
                'is_safe_robot': True
            }
            print(self.nonlocal_variables)

    def closeEvent(self, event):
        self.timer.stop()
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = ZEDApp()
    main_window.show()
    sys.exit(app.exec_())


