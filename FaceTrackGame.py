import cv2
import numpy as np
import time
from random import seed
from random import random

face_cascade = cv2.CascadeClassifier('//usr//share//opencv//haarcascades//haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
gameOver = False

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

def CheckGameOver(x, y, game, radius):
    global gameOver
    for i in range(x - radius, x + radius + 1):
        if game[y-radius, i, 0]:
            gameOver = True
            return True
        if game[y+radius+1, i, 0]:
            gameOver = True
            return True

    for i in range(y - radius, y + radius+1):
        if game[i, x-radius, 0]:
            gameOver = True
            return True
        if game[i, x+radius+1, 0]:
            gameOver = True
            return True

    return False

speed = 2
score = 0
FindAFace()
seed(time.time())
rectYi = [0, 0, 0 ,0]
rectYf =  [0, 0, 0, 0]
safeZone = [100, 100, 100, 100]
safeZoneFlag = [True, False, False, False]
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

    if not gameOver:
        for i in range(4):
            if safeZoneFlag[i]:
                safeZone[i] = int(frame.shape[1] * 3.0 / 4.0 * random() + 80)
                safeZoneFlag[i] = False

            if rectFlag[i]:
                cv2.rectangle(game, (0, rectYi[i]), (safeZone[i] - 60, rectYf[i]), (128, 128, 0), -1)
                cv2.rectangle(game, (safeZone[i] + 60, rectYi[i]), (frame.shape[1] - 1, rectYf[i]), (128, 128, 0), -1)

                rectYf[i] += speed
                if rectYf[i] >= 70:
                    rectYi[i] += speed

                if rectYf[i] >= frame.shape[0]:
                    rectYf[i] = frame.shape[0] - 1

                if rectYi[i] >= frame.shape[0]:
                    rectYi[i] = 0
                    rectYf[i] = 0
                    rectFlag[i] = 0
                    safeZoneFlag[i] = 0

                if rectYi[i] >= 90:
                    if i < 3:
                        if not rectFlag[i + 1]:
                            safeZoneFlag[i + 1] = True
                            score += 1
                        rectFlag[i + 1] = True
                    else:
                        if not rectFlag[0]:
                            safeZoneFlag[0] = True
                            score += 1
                        rectFlag[0] = True

        print(rectYf, rectYi, safeZone, safeZoneFlag)

        CheckGameOver(centerX, centerY, game, 21)
        cv2.circle(game, (centerX, centerY), 11, (0, 140, 255), 21)
         
        print(gameOver)  

    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(game, "Game Over! Score: " + str(score) , (centerX - 180, centerY - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print(score)

    cv2.imshow('tracker', frame)
    cv2.imshow('Game', game)
    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()
