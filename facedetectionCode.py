#!/usr/bin/env python
# coding: utf-8

#         import numpy , matplotlib, cv2.
#         extract the opencv file and add to scripts of your python .
#         change the locations mentioned as per the comments and according to your PC 


import numpy as np
import cv2
import matplotlib.pyplot as plt


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces(cascade, test_image,scaleFactor =1.1):
    image_copy = test_image
    gray=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    faces_rects=haar_cascade_face.detectMultiScale(gray, scaleFactor = scaleFactor ,minNeighbors = 5)
    print('Faces found :', len(faces_rects))
    for (x,y,w,h) in faces_rects :
        cv2.rectangle(image_copy , (x,y), (x+w,y+h),(0,255,0),2)
    print('hi')
    return image_copy


#read the image
#location of faceimage
test_image=cv2.imread('C:/Users/arshika/Desktop/Sih/faceimage.jpg')
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
plt.imshow(test_image_gray,cmap='gray')
#location of data/haarcascades/haarcascade_frontalface_default.xml
haar_cascade_face= cv2.CascadeClassifier('C:/Python27/Lib/site-packages/OpenCV 4.0.0-alpha/opencv-opencv-e7c915a/data/haarcascades/haarcascade_frontalface_default.xml')
faces=detect_faces(haar_cascade_face, test_image)
plt.imshow(convertToRGB(faces))
print('check')    


# In[10]:


cam=cv2.VideoCapture(0)
while True:
    ret_val, img= cam.read()
    if ret_val== True:
        test_image=img
        faces=detect_faces(haar_cascade_face, test_image)
        #plt.imshow(convertToRGB(faces))
        cv2.imshow('Frame',convertToRGB(faces))
        # Press Q on keyboard to  exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
    

cam.release()
cv2.destroyAllWindows()






