import cv2
import os
import numpy as np

def facedetection(test_img):
    gray_img= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('\\Users\\Purvika pandey\\AppData\\Local\\Programs\\Python\\Python37-32\\cs_project\\cascade_frontal.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img , scaleFactor = 1.3 , minNeighbors = 5)

    return faces,gray_img
