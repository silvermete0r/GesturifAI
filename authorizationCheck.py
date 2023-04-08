# Import necessary libraries
import cv2
import numpy as np
import face_recognition
import os

# Set path of the directory containing the images
path = 'Approved'

# Create empty lists to store images and corresponding class names
images = []
classNames = []

# Get a list of all files in the directory
myList = os.listdir(path)
print(myList)

# Loop through the files in the directory
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Define a function to encode the images
def findEncodings(images):
    # Create an empty list to store the encodings
    encodeList = []
    # Loop through the images
    for img in images:
        # Find the face encoding using the face_recognition library
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Call the function to encode the images
encodeListKnown = findEncodings(images)
print('Encoding Completed!')

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Loop indefinitely
while True:
    # Read a frame from the camera
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find the locations of all faces in the frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    # Loop through all face encodings in the frame
    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # If the face matches with one of the known encodings
        if matches[matchIndex]:
            # Get the name of the person from the class names list
            name = classNames[matchIndex].upper()
            # Calculate the distance between the face encoding and the best match
            IMndex = round(faceDis[matchIndex],2)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name + " " + str(IMndex),(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
        else:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,"Not Approved",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)

    # Display the window
    cv2.imshow('WebCam',img)
    
    # Wait for Esc key to stop 
    k = cv2.waitKey(1) & 0xff
    if k == 27: 
        break

# Close the window 
cap.release()

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 