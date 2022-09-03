import cv2
import numpy as np
import serial

# COM_PORT = 'COM3'  
# BAUD_RATES = 9600
# ser = serial.Serial(COM_PORT, BAUD_RATES)
def ROI(image):
    height=image.shape[0]
    width=image.shape[1]
    mask = np.zeros_like(image)
    match_mask_color = (255,255,255)
    point=np.array([[width/4,height/4],[width/4,height*3/4],[width*3/4,height*3/4],[width*3/4,height/4]])
    cv2.fillPoly(mask, np.int32([point]), match_mask_color)
    masked_image = cv2.bitwise_and(image, mask) 
    return masked_image
cap = cv2.VideoCapture(0)

c = 1
timeF = 10  # frame time

while(1):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    if (c % timeF == 0 or c % timeF == 5):# frame 限制
    
      dst = cv2.pyrMeanShiftFiltering(frame, 10, 50)#濾波
      roi=ROI(dst)
      hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
      gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)#灰度
      ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)#二值化
      lower_red1 = np.array([160,50,0])
      upper_red1 = np.array([179,255,255])
      lower_red2 = np.array([0,50,0])
      upper_red2 = np.array([10,255,255])
      mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
      mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
      mask = mask1 + mask2
      res = cv2.bitwise_and(frame, frame, mask=mask)
      
      cv2.imshow("ShiftFiltering", dst)
      cv2.imshow("threshold", thresh)
      cv2.imshow('res',res)
      cv2.imshow('mask',mask)
      #cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cnt2, hierarchy =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#加ROI?

      for i in cnt2: 
        
        #輪廓近似
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.01*peri, True)
        
        
        if len(approx) == 4 and peri >100:
          image=cv2.drawContours(frame,i, -1, (0, 255, 0), 2)
          text=str(peri)
          #ser.write(b'four\n')
          cv2.putText(image,text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
          cv2.imshow("result", image)
        elif len(approx) == 3 and peri >100:
          image=cv2.drawContours(frame,i, -1, (0, 255, 0), 2)
          text=str(peri)
          #ser.write(b'three\n')
          cv2.putText(image,text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
          cv2.imshow("result", image)
        elif len(approx)>10:
          #circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=100,maxRadius=200)
          image=frame.copy()
          image=cv2.drawContours(frame,i, -1, (0, 255, 0), 2)
          text=str(peri)
          #ser.write(b'circle\n')
          cv2.putText(image,text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
          cv2.imshow("result", image)
        # else:
        #       ser.write(b'one\n')
              
      
        
        
          
        
              

    c = c + 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        final = image
        break

# 釋放攝影機
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
cv2.imshow("final", final)
cv2.waitKey(0)