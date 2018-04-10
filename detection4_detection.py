import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
    # Capture frame-by-frame
    ret, img = cam.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceDetect.detectMultiScale(gray,1.3,5)

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        cv2.putText(cv2.fromarray(img),str(id),(x,y+h),font,255)


    # Display the resulting frame
    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
