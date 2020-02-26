import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('//usr//share//opencv//haarcascades//haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FPS))

def FindAFace():
    while(True):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.02, 50)
        if len(faces):
            x, y, w, h = faces[0]
            global roio, roi
            roio = frame[y:y+h, x:x+w]
            roi = roio
            cv2.imshow('face', roi)
            print("Detected!")
            break


FindAFace()

while(True):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roig = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roiog = cv2.cvtColor(roio, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    templated = cv2.matchTemplate(gray, roig, cv2.TM_CCOEFF_NORMED)
    templatedo = cv2.matchTemplate(gray, roiog, cv2.TM_CCOEFF_NORMED)

    _, _, _, loc = cv2.minMaxLoc(templated)

    if templatedo[loc[1]][loc[0]] < 0.1:
        FindAFace()
        continue
        
    h, w = roi.shape[:2]

    cv2.rectangle(frame, loc, (loc[0] + w, loc[1] + h), (0,255,255), 2) 
    roi = frame[loc[1]:loc[1]+h, loc[0]:loc[0]+w]

    cv2.imshow('tracker', frame)
    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()
