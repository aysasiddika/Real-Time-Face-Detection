from PIL import Image
from importlib.resources import path
import face_recognition
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

path = 'D:\Lahm\JBL\FaceDetectionWithWebcam\Data'
path1 = 'D:\Lahm\JBL\FaceDetectionWithWebcam\Data1'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    
    im = Image.open(f'{path}/{cl}')
    classNames.append(os.path.splitext(cl)[0])
    Names =os.path.splitext(cl)[0]
    rgb_im = im.convert('RGB')
    rgb_im.save(f'D:\Lahm\JBL\FaceDetectionWithWebcam\DATA2\{Names}.jpeg')