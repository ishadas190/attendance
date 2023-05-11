import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
     with open('attendance.csv','a+') as f:
         myDataList = f.readlines()
         print(myDataList)
         nameList = []
         nameList = []
         nameList = []
         for line in myDataList:
             entry = line.split(',')
             nameList.append(entry[0])
         if name not in nameList:
             time_now = datetime.now()
             tStr = time_now.strftime('%H:%M:%S')
             f.writelines(f'\n{name},{tStr}')

     a = pd.read_csv("attendance.csv")

     a.to_html("Table.htm")

     html_file = a.to_html()


encodeListKnown  = findEncodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurframe = face_recognition.face_locations(imgS)
    encodesCurframe = face_recognition.face_encodings(imgS,facesCurframe)

    for encodeFace,faceloc in zip(encodesCurframe,facesCurframe):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Webcam', img)
            markAttendance(name)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release