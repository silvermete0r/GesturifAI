################################
# Author: @silvermete0r ©2023  #
# Project: GesturifAI          #
# Event: JAS AI Hackathon      #
################################


######################################
# Importing Libraries & Dependencies #
######################################

import cv2
from Modules.HandTrackingModule import HandDetector
from Modules.ClassificationModule import Classifier
from Modules.FaceDetectionModule import FaceDetector
import keyboard
import numpy as np
import math
import time


###########################
# Main System Definitions #
###########################

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/gesturifai_keras_model.h5", "Models/labels.txt")
faceDetector = FaceDetector()

commands = {
    '0': "0",
    '1': "1",
    '2': "2",
    '3': "3",
    '4': "4",
    'Up': "up",
    'Down': "down",
    'Left': "left",
    'Right': "right",
    'Shift': "shift",
    'Space': "space",
    'Enter': "space",
    'Back': "backspace",
    'Special-A': "esc",
    'Special-B': "-",
    'Special-C': "+"
}

labels = ['0','1','2','3','4','Up','Down','Left','Right','Shift','Space','Enter','Back','Special-A','Special-B','Special-C']

offset = 20
imgSize = 300

########################
# Computer Vision Loop #
########################

while True:
    try:
        # Getting Video Frames
        success, img = cap.read()
        # If something went wrong just skip
        if not success:
            continue
        # Checking for faces and hands in the frame
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        imgOutput, bboxs = faceDetector.findFaces(imgOutput)

        # Check if face detected
        if bboxs:
            # Generating GesturifAI Computer Vision Zone
            l = 30
            x, y, w, h = bboxs[0][1]
            gx, gy, gw, gh = x-l-int(w*1.6), y-l, x-l, y+int(h*1.6)-l
            cv2.rectangle(imgOutput, (gx,gy), (gw,gh), (0,255,0), 2)
            cv2.putText(imgOutput,f"GesturifAI", (gx,gy-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            # Check if hand detected
            if hands:
                # Initializing box for Hand Tracking
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Checking for the right hand
                # if hand['type'] != 'Right':
                
                # Do commands only if hand box in GesturifAI Computer Vision Zone
                if x in range(int((gx+gw)/1.4)) and y in range(int((gy+gh)/1.4)):
                    imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
                    imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

                    # Continue if any issue with ImageCrop was occured
                    if imgCrop.size == 0:
                        continue

                    aspectRatio = h/w

                    # Checking hands position Horizontal aR<1 else Vertical
                    if aspectRatio > 1:
                        k = imgSize/h
                        wCal = math.ceil(k*w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize-wCal)/2)
                        imgWhite[:, wGap:wCal+wGap] = imgResize
                    else:
                        k = imgSize/w
                        hCal = math.ceil(k*h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize-hCal)/2)
                        imgWhite[hGap:hCal+hGap, :] = imgResize
                    
                    # Getting some predictions
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction[index],labels[index])

                    # keyboard.press_and_release(commands[labels[index]])
                    if index in (10,11):
                        keyboard.press_and_release('space')

                    cv2.putText(imgOutput,f'{labels[index]}, AC: {round(prediction[index]*100,2)}%',(x,y-30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255,0,0), 2)
                    cv2.imshow("HandTracking", imgWhite)
                    k = cv2.waitKey(1) & 0xff
        
            # Show Time
            cv2.imshow("GesturifAI", imgOutput)

            # Program termination by 'ESC' button
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    # Terminate the program if any unexpected errors occured
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Termination of video processing and closing windows
cap.release()
cv2.destroyAllWindows() 