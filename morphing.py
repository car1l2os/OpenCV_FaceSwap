import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    temp_positions = list();

    
    for (x,y,w,h) in faces:
        if(len(faces) == 2):
            temp_positions.append([x,y,w,h])
            #face_one = img[y:y+h, x:x+w]
            #img[0:h, 0:w] = face_one

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    if(len(faces) == 2):
        x1,y1,w1,h1 = temp_positions[0]
        face_one = img[y1:y1+h1, x1:x1+w1]
        x2,y2,w2,h2 = temp_positions[1]
        face_two = img[y2:y2+h2, x2:x2+w2]
        face_one = cv2.resize(face_one, (w2, h2))
        face_two = cv2.resize(face_two, (w1, h1))


        img[y1:y1+h1, x1:x1+w1] = face_two
        img[y2:y2+h2, x2:x2+w2] = face_one

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
