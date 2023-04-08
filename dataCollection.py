# Import necessary libraries
import cv2
import numpy as np
import math
import time
from Modules.HandTrackingModule import HandDetector

# System Definitions 
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/Special-C"
counter = 0

# Start the video capture loop
while True:
    try:
        # Capture frame from camera
        success, img = cap.read()

        # Use HandDetector to find hands in the image
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create a white image with size imgSize x imgSize
            imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            aspectRatio = h/w

            # If the aspect ratio is greater than 1, scale the image by height
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize

            # If the aspect ratio is less than or equal to 1, scale the image by width
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize

            # Display the cropped and scaled image on a white background
            cv2.imshow("ImageWhite", imgWhite)

        # Display the original image with hand landmarks
        cv2.imshow("Image", img)

        # Wait for 's' key to save the image
        k = cv2.waitKey(1) & 0xff
        if k == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)

        # Wait for 'Esc' key to exit the loop
        elif k == 27:
            break

    # Catch and print any errors that occur during execution
    except Exception as e:
        print(f"An error occurred: {e}")
        break

