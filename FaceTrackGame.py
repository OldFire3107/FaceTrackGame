import cv2
import numpy as np
from random import seed
from random import random

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

speed = 4
FindAFace()
seed(1)
rectYi = [0, 0, 0 ,0]
rectYf =  [0, 0, 0, 0]
safeZone = [0, 0, 0, 0]
safeZoneFlag = [True, False, True, True]
rectFlag = [True, False, False, False]

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

    # The game code begins from here. The above code was only for detection and tracking
    game = np.zeros(frame.shape, dtype=np.uint8)
    rectCount = 0

    centerX = int(loc[0] + w / 2)
    centerY = int(loc[1] + h / 2)

    for i in range(4):
        if safeZoneFlag[i]:
            safeZone[i] = int(frame.shape[1] * 15 / 16 * random() + 20)
            safeZoneFlag[i] = False

        if rectFlag[i]:
            cv2.rectangle(game, (0, rectYi[i]), (safeZone[i] - 60, rectYf[i]), (128, 128, 0), -1)
            cv2.rectangle(game, (safeZone[i] + 60, rectYi[i]), (frame.shape[1] - 1, rectYf[i]), (128, 128, 0), -1)

            if rectYf[i] >= 70:
                rectYi[i] += speed

            rectYf[i] += speed

            if rectYf[i] >= frame.shape[0]:
                rectYf[i] = frame.shape[0] - 1

            if rectYi[i] >= frame.shape[0]:
                rectYi[i] = 0
                rectYf[i] = 0
                rectFlag[i] = 0
                safeZoneFlag = 0

            if rectYi[i] >= 90:
                if i != 3:
                    rectFlag[i + 1] = True
                    safeZoneFlag[i + 1] = True
                else:
                    rectFlag[0] = True
                    safeZoneFlag[0] = True

    print(rectYf, rectYi)

    cv2.circle(game, (centerX, centerY), 10, (0, 140, 255), 11)

    cv2.imshow('tracker', frame)
    cv2.imshow('Game', game)
    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()
