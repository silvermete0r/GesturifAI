###########################################
# Source: https://www.computervision.zone #
###########################################

import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.8, maxDetectionNum=1):
        self.minDetectionCon = minDetectionCon
        self.maxDetectionNum = maxDetectionNum
        #importing mediapipe
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon, self.maxDetectionNum)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bBoxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bBox = int(bBoxC.xmin * iw), int(bBoxC.ymin * ih), \
                    int(bBoxC.width * iw), int(bBoxC.height * ih)
                bboxs.append([id, bBox, detection.score])
                if draw:
                    self.fancyDraw(img, bBox)
                    cv2.putText(img,f"Face, AC: {int(detection.score[0]*100)}%", (bBox[0],bBox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=7, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        #Top Left
        cv2.line(img, (x,y), (x+l, y), (255,0,255), t)
        cv2.line(img, (x,y), (x, y+l), (255,0,255), t)
        #Top Right
        cv2.line(img, (x1,y), (x1-l, y), (255,0,255), t)
        cv2.line(img, (x1,y), (x1, y+l), (255,0,255), t)
        #Bottom Left
        cv2.line(img, (x,y1), (x+l, y1), (255,0,255), t)
        cv2.line(img, (x,y1), (x, y1-l), (255,0,255), t)
        #Bottom Right
        cv2.line(img, (x1,y1), (x1-l, y1), (255,0,255), t)
        cv2.line(img, (x1,y1), (x1, y1-l), (255,0,255), t)

        return img

def main():
    #Capturing video
    cap = cv2.VideoCapture(0)

    #Previous time value for FPS
    # pTime = 0

    #Initializing face detection class
    detector = FaceDetector()

    #Video processing for detecting faces -> Loop
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        #Frames per second counting
        # cTime = time.time()
        # fps = 1/(cTime-pTime)
        # pTime = cTime
        # cv2.putText(img,f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
            
        #Show resultant frames
        cv2.imshow("Image", img)
        
        #Close when ESC is pressed
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break



if __name__ == "__main__":
    main()