import cv2
import os
import  numpy as np
import face_detect as fr

test_img = cv2.imread("\\Users\\Purvika pandey\\AppData\\Local\\Programs\\Python\\Python37-32\\cs_project\\pic.jpg")
face_detected,gray_img = fr.facedetection(test_img)
print("face detected: ",face_detected)

for (x,y,w,h) in face_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness = 5)
    
resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow("face to be detected",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
