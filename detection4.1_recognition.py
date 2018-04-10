import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")
path= "dataSet"

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


font=cv2.FONT_HERSHEY_SIMPLEX

while(True):
    # Capture frame-by-frame
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceDetect.detectMultiScale(gray,1.3,5)

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])

        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img, str(profile[1]), (x, y + h + 30), font, 0.5, (255, 0, 0 ), 2)
            cv2.putText(img, str(profile[2]), (x, y + h + 60), font, 0.5, (255, 0, 0),2)
            cv2.putText(img, str(profile[3]), (x, y + h + 90), font, 0.5, (255, 0, 0), 2)


    cv2.imshow('Face', img)
    if (cv2.waitKey(1) == ord('q')):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
