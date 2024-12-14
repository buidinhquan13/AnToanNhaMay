import cv2
import numpy as np

from cv_viewer.utils import *
import pyzed.sl as sl
import math
#----------------------------------------------------------------------
#       2D VIEW
#----------------------------------------------------------------------
def cvt(pt, scale):
    '''
    Function that scales point coordinates
    '''
    out = [pt[0]*scale[0], pt[1]*scale[1]]
    return out

def render_sk(left_display, img_scale, obj, color, BODY_BONES,a1_working,b1_working, a2_working, b2_working,a1_robot,b1_robot, a2_robot, b2_robot):
    # Draw skeleton bones
    for part in BODY_BONES:
        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
        # Check that the keypoints are inside the image
        if(kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0] 
        and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
        and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0 ):
            cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), color, 1, cv2.LINE_AA)

    # Skeleton joints
    for kp in obj.keypoint_2d:
        cv_kp = cvt(kp, img_scale)
        if(cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]):
            cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)
    
    




    #### draw bounding box


    top_left = cvt(obj.bounding_box_2d[0],img_scale)
    bottom_right = cvt(obj.bounding_box_2d[2],img_scale)
    cv2.rectangle(left_display,(int(top_left[0]),int(top_left[1])),
                  (int(bottom_right[0]),int(bottom_right[1])),
                  (0,255,255),2
                  )
    
    label = f"{obj.id}"
    
    vx, vy, vz = obj.velocity
    vel = round(math.sqrt(vx*vx+vy*vy),2)
    label = f"ID: {obj.id}     Vel: {vel}m/s"
    cv2.putText(left_display,label,(int(top_left[0]),int(top_left[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,255,50),2)
    
    #cv2.line(left_display,(640,10),(640,800),(255,0,0),5)
    #cv2.line(left_display,(200,0),(200,800),(0,255,0),5)
    # if ((int(top_left[0]) +int(bottom_right[0]))//2 <= 680):
    #     print((int(top_left[0]) +int(bottom_right[0]))//2)
    #     return 1
    # else:
    #     return 0
    
    #print('vi tri nguoi')
    #print(bottom_right[1],bottom_right[0])
    if check_people_working(bottom_right[1],bottom_right[0],a1_working,b1_working, a2_working, b2_working):
        return 1,0
    elif check_people_robot(bottom_right[1],bottom_right[0],a1_robot,b1_robot, a2_robot, b2_robot):
        return 0,1
    else:
        return 0,0


    #### yolo
    



def draw_area_robot(img, points,is_safety = True):
    if is_safety:
        overlay_color = (0, 255, 255)  # Color in BGR format (orange)
    else:
        overlay_color = (0, 0, 255)  # Color in BGR format (orange)
    height,width,_  = img.shape
    
    overlay_alpha = 0.2  # Transparency level (0.0 to 1.0)
    overlay_poly = np.zeros(( height,width, 4), dtype=np.uint8)
    cv2.fillPoly(overlay_poly, [points], overlay_color)
    for i in range(4):
        cv2.circle(img,(points[i][0],points[i][1]),3,(255,0,0),-1)
    # Overlay the polygon on the frame
    frame = cv2.addWeighted(img, 0.8, overlay_poly, overlay_alpha, 0)
    return frame 

def draw_area_working(img, points, is_safety = True):
    if is_safety:
        overlay_color = (0, 255, 0)  
    else:
        overlay_color = (0, 0, 255)  
    height,width,_  = img.shape
     # (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    
    overlay_alpha = 0.2  # Transparency level (0.0 to 1.0)
    overlay_poly = np.zeros(( height,width, 4), dtype=np.uint8)
    cv2.fillPoly(overlay_poly, [points], overlay_color)
    for i in range(4):
        cv2.circle(img,(points[i][0],points[i][1]),3,(255,0,0),-1)
    # Overlay the polygon on the frame
    frame = cv2.addWeighted(img, 1 - overlay_alpha, overlay_poly, overlay_alpha, 0)
    
    
    
    #cv2.line(frame,(445,0),(445,800),(0,255,0),5)
    return frame 
def check_people_working(h_p,w_p,a1_working,b1_working, a2_working, b2_working ):
        # if h_p - (w_p*a1_working+b1_working) > 0 and h_p - (w_p*a2_working + b2_working) <0:
        #     return 1
        # return 0
        if  w_p >350 and w_p < 640:
            return 1
        return 0

def check_people_robot(h_p,w_p,a1_robot,b1_robot, a2_robot, b2_robot ):
        # if h_p - (w_p*a1_robot +b1_robot) > 0 and h_p - (w_p*a2_robot + b2_robot) <0:
        #     return 1
        # return 0
        if w_p > 640:
            return 1
        return 0

def render_2D(left_display, img_scale, objects, is_tracking_on, body_format,a1_working,b1_working, a2_working, b2_working,a1_robot,b1_robot, a2_robot, b2_robot):
    '''
    Parameters
        left_display (np.array): numpy array containing image data
        img_scale (list[float])
        objects (list[sl.ObjectData]) 
    '''
    overlay = left_display.copy()
    

    person_in_working = 0
    person_in_robot = 0
    # Render skeleton joints and bones
    for obj in objects:
        if render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                color = generate_color_id_u(obj.id)
                if body_format == sl.BODY_FORMAT.BODY_18:
                    p_working, p_robot = render_sk(left_display, img_scale, obj, color, sl.BODY_18_BONES,a1_working,b1_working, a2_working, b2_working,a1_robot,b1_robot, a2_robot, b2_robot)
                    if p_working:
                        person_in_working += 1
                    elif p_robot:
                        person_in_robot += 1 



                    ##### check safety
                    
                    ### check danger

                elif body_format == sl.BODY_FORMAT.BODY_34:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_34_BONES)
                elif body_format == sl.BODY_FORMAT.BODY_38:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES) 

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)
    return person_in_working, person_in_robot


    