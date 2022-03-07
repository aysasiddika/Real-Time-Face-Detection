from importlib.resources import path
import face_recognition
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json,load_model
from mtcnn import MTCNN
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

path = 'D:\Lahm\JBL\FaceDetectionWithWebcam\Data'
images = []
classNames = []
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
myList = os.listdir(path)
#print(myList)
MODEL_NAME = 'liveness_model.h5'
LABEL_NAME = 'label.pickle'
face_detector_mtcnn = MTCNN()
MODEL_PATH = os.path.abspath(
     './face_utils/cnn_models/liveness_model/' + MODEL_NAME)
LABEL_PATH = os.path.abspath(
     './face_utils/cnn_models/liveness_model/' + LABEL_NAME)

liveness_labels = pickle.loads(open(LABEL_PATH, "rb").read())
liveness_model = load_model(MODEL_PATH)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
   
#print(classNames)

def is_face_live(image):
     #img = base64_to_numpy(base64_image)
     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     img = crop_roi_mtcnn(img)
     # cv2.imshow('i2', img)
     # cv2.waitKey(0)
     img = preprocess_input(img)
     img = cv2.resize(img, (224, 224))
     img = np.expand_dims(img, axis=0)
     prediction = liveness_model.predict(img)
     print("Predictions: ", prediction)
     result_class = np.argmax(prediction)
     if result_class == liveness_labels['real']:
         print("Image is real")
         return True
     else:
         print("Image is fake")
         return False
def crop_roi_mtcnn(img):
     face = face_detector_mtcnn.detect_faces(img)
     if len(face) > 0:
         box = face[0]['box']
         #print("Box: ", box)
         startX = box[0]
         startY = box[1]
         endX = startX + box[2]
         endY = startY + box[3]

         roi_img_array = img[startY: endY, startX: endX]
         return roi_img_array
     else:
         return None
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        encode = face_recognition.face_encodings(img)[0]
        #print(classNames[len(encode)].upper())
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')
#print(len(encodeListKnown))


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #status = is_face_live(img)
    locations = face_recognition.face_locations(imgS)
    count = len(locations)
    if count!= 0:
     faceCurFrame = face_recognition.face_locations(imgS)
     faceCurEncode = face_recognition.face_encodings(imgS,faceCurFrame)
     #if faceCurEncode.size 
     if is_face_live(img) == True:
     	for encodeFace,faceLoc in zip(faceCurEncode,faceCurFrame):
     		matches = face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.475)
     		facedis =face_recognition.face_distance(encodeListKnown,encodeFace)
     		print(facedis)
     		matchIndex = np.argmin(facedis)
     		if matches[matchIndex]:
     			name = classNames[matchIndex].upper()
     			print(name)
     			y1,x2,y2,x1 = faceLoc
     			# y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
     			cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
     			cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
     			cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

     else:
     	for encodeFace,faceLoc in zip(faceCurEncode,faceCurFrame):
     		matches = face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.475)
     		facedis =face_recognition.face_distance(encodeListKnown,encodeFace)
     		print(facedis)   
     		matchIndex = np.argmin(facedis)
     		if matches[matchIndex]:
     			name = classNames[matchIndex].upper()
     			print(name)
     			y1,x2,y2,x1 = faceLoc
     			# y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
     			cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
     			cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
     			cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    else:
     print('No face')
    cv2.imshow('Webcan',img)
    cv2.waitKey(1)





