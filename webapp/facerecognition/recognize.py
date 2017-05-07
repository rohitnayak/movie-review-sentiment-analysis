import cv2
import numpy as np
from PIL import Image
import sys

def recognize(imageInPath, imageOutPath):
    recognizer = cv2.face.createLBPHFaceRecognizer()
    recognizer.load('facerecognition/train/train.yml')
    cascadePath = "facerecognition/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    users = {1: "rohit", 2: "akash", 3: "aditi"}
    im = cv2.imread(imageInPath)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),4)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        #print("Confidence level for id " + str(Id) + " is " + str(conf))
        if(conf<100):
            user = users[Id]
        else:
            user="Unknown"
        confPct = 100 - (100/100)*(conf-20)
        if (confPct < 0): confPct = 0
        if (confPct > 100): confPct = 99
        if user == "Unknown":
            s = user
        else:
            s = "%s (%d%%)" % (user, confPct)
        cv2.putText(im, s, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 3)
    cv2.imwrite(imageOutPath, im)
    return imageOutPath
