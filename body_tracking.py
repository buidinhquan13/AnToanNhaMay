########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
from ultralytics import YOLO
import threading
import math

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")



def main():

    model = YOLO("models/best.pt")

    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = False            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    #viewer = gl.GLViewer()
    #viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()



    image = sl.Mat()
    key_wait = 10 
    delay = 0
    name_classes = ['Person', 'Vest', 'Helmet']
    nonlocal_variables = {'person_in_robot':0,
                          'person_in_working':0,
                          'vest_in_robot': 0,
                          'vest_in_working':0,
                          'hel_in_robot': 0,
                          'hel_in_working': 0,
                          'is_safe_working': True,
                          'is_safe_robot':True}
    points_area_robot = np.array([[640, 370], [830, 370], [1150, 720], [640, 720]], dtype=np.int32) 
    points_area_working = np.array([[445,370],[640,370],[640,720],[320,720]], dtype=np.int32) 
    
    a1_working = (points_area_working[0][1]-points_area_working[3][1])/(points_area_working[0][0]-points_area_working[3][0])
    a1_working = 0 if math.isinf(a1_working) else a1_working
    b1_working = points_area_working[0][1] - a1_working*points_area_working[0][0]
    
    a2_working = (points_area_working[2][1]-points_area_working[1][1])/(points_area_working[2][0]-points_area_working[1][0])
    a2_working = 0 if math.isinf(a2_working) else a2_working
    b2_working = points_area_working[2][1] - a2_working*points_area_working[2][0]
    

    a1_robot = (points_area_robot[0][1]-points_area_robot[3][1])/(points_area_robot[0][0]-points_area_robot[3][0])
    a1_robot = 0 if math.isinf(a1_robot) else a1_robot
    b1_robot = points_area_robot[0][1] - a1_robot*points_area_robot[0][0]
    
    a2_robot = (points_area_robot[2][1]-points_area_robot[1][1])/(points_area_robot[2][0]-points_area_robot[1][0])
    a2_robot = 0 if math.isinf(a2_robot) else a2_robot
    b2_robot = points_area_robot[2][1] - a2_robot*points_area_robot[2][0]
    
    def check_inside_working(h_p,w_p ):
        # if h_p - (w_p*a1_working+b1_working) > 0 and h_p - (w_p*a2_working + b2_working) <0:
        #     #nonlocal_variables[class_] += 1
        #     return 1
        # return 0
        
        if w_p > 350 and w_p <640:
            return 1
        return 0

    while True: #viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            
            # Update GL view
            
            #viewer.update_view(image, bodies) 
            
            # Update OCV view
            image_left_ocv = image.get_data()
            
            results = model(image_left_ocv[:,:,:3],classes = [1,2], conf=0.7) 
            #cv2.line(image_left_ocv,(445,0),(445,800),(0,0,255),5)

   

            # Process each detection
            for result in results:
                #boxes = result.boxes.xyxy  # Get bounding boxes
                boxes = result.boxes
                for box,class_id in zip(boxes.xyxy, boxes.cls):
                    x1, y1, x2, y2 = map(int, box[:4])

                    label = f"{name_classes[int(class_id)]}"
                    cv2.putText(image_left_ocv,label,(x1+20,y1+50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,50,100),2)

                    cv2.rectangle(image_left_ocv, (x1, y1), (x2, y2), (255, 50, 0), 2)  

                    
                    #class_id = int(box.cls[0])
                    # Kiểm tra nếu đối tượng là class "1" và x2 <= 800
                    
                    if class_id == 1 and check_inside_working(y2,x2):
                        #print(y2,x2)
                        nonlocal_variables['vest_in_working'] += 1

                    # Kiểm tra nếu đối tượng là class "2" và x2 <= 800
                    
                    if class_id == 2 and check_inside_working(y2,x2):
                        
                        nonlocal_variables['hel_in_working'] += 1

            
            # In kết quả đếm
            #print(f"Số lượng vest: {nonlocal_variables['vest_in_working']}")
            #print(f"Số lượng mũ: {nonlocal_variables['hel_in_robot']}")


            nonlocal_variables['person_in_working'], nonlocal_variables['person_in_robot'] = cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format,
                                                                                                               a1_working,b1_working, a2_working, b2_working,a1_robot,b1_robot, a2_robot, b2_robot)
            #cv2.line(image_left_ocv,(800,00),(800,1000),(255,0,0),2)
            #print('so luong nguoi ', count_person_in)

            ###check safety
            if (nonlocal_variables['person_in_working'] != nonlocal_variables['vest_in_working'] or nonlocal_variables['person_in_working'] != nonlocal_variables['hel_in_working']):
                print('canh bao')
                #print(delay)
                delay += 1
                
                if delay > 3:
                    delay = 35
                    print("Warning")
                    nonlocal_variables['is_safe_working'] = False
            else:
                delay = 0

            if nonlocal_variables['person_in_robot']>0:
                nonlocal_variables['is_safe_robot'] = False

            image_left_ocv = cv_viewer.draw_area_working(image_left_ocv,points_area_working, nonlocal_variables['is_safe_working'])
            image_left_ocv = cv_viewer.draw_area_robot(image_left_ocv,points_area_robot, nonlocal_variables['is_safe_robot'])

            print(nonlocal_variables)
            nonlocal_variables = {'person_in_robot':0,
                          'person_in_working':0,
                          'vest_in_robot': 0,
                          'vest_in_working':0,
                          'hel_in_robot': 0,
                          'hel_in_working': 0,
                          'is_safe_working': True,
                          'is_safe_robot':True}
            
            #cv2.imwrite('out.png',image_left_ocv)
            cv2.imshow("ZED | 2D View", image_left_ocv)





            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 


    #viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 