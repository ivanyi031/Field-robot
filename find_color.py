from turtle import color
import cv2
import numpy as np

#img=cv2.imread('C:\\Users\\User\\Dropbox\\PC\\Desktop\\water_color.JPG')
#imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def ROI(image):
    height=image.shape[0]
    width=image.shape[1]
    mask = np.zeros_like(image)
    match_mask_color = (255,255,255)
    point=np.array([[width/4,height/4],[width/4,height*3/4],[width*3/4,height*3/4],[width*3/4,height/4]])
    cv2.fillPoly(mask, np.int32([point]), match_mask_color)
    masked_image = cv2.bitwise_and(image, mask) 
    return masked_image
def empty(object):
    pass
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    
    ret, frame = cap.read()
    dst = cv2.pyrMeanShiftFiltering(frame, 10, 50)#濾波
    ROI_img=ROI(dst)
    hsv=cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    ret,thresh2 = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh',thresh2)
    cv2.imshow('frame',ROI_img)
    

    # h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    # h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    # s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    # s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    # v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    # v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # lower = np.array([h_min, s_min, v_min])
    # upper = np.array([h_max, s_max, v_max])
    # mask = cv2.inRange(hsv, lower, upper)
    # imgResult = cv2.bitwise_and(frame, frame, mask=mask)

    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
    # cv2.imshow('Result',imgResult)

    
    lower_blue=np.array([38,147,0])
    upper_blue=np.array([176,255,255])
    mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)
    lower_red = np.array([0,180,170])
    upper_red = np.array([179,255,255])
    mask_red=cv2.inRange(hsv,lower_red,upper_red)
    lower_yellow=np.array([7,51,162])
    upper_yellow=np.array([121,255,255])
    mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
    Result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    Result_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    Result_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)
    cv2.imshow('blue',Result_blue)
    cv2.imshow('red',Result_red)
    cv2.imshow('yellow',Result_yellow)
    
    area_blue=[0]
    area_red=[0]
    area_yellow=[0]
    color=[]
    cntblue, hierarchy =cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in cntblue:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.01*peri, True)
        area_blue.append(cv2.contourArea(i))
    max_blue=max(area_blue)
    color.append(max_blue)
    cntbred, hierarchy =cv2.findContours(mask_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in cntbred:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.01*peri, True)
        area_red.append(cv2.contourArea(i))
    max_red=max(area_red)
    color.append(max_red)
    cntyellow, hierarchy =cv2.findContours(mask_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in cntyellow:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.01*peri, True)
        
        area_yellow.append(cv2.contourArea(i))
    max_yellow=max(area_yellow)
    color.append(max_yellow)
    actual_color=max(color)
    if actual_color==max_blue:
        cv2.putText(frame,'blue',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("result",frame)
    elif actual_color==max_red:
        cv2.putText(frame,'red',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("result",frame)
    elif actual_color==max_yellow:
        cv2.putText(frame,'yellow',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("result",frame)
    
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    


    


