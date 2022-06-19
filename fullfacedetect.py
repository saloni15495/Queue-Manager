import cv2
import numpy as np
import pickle

capt=cv2.VideoCapture('queue.mp4')
fface=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
pface=cv2.CascadeClassifier("haarcascade_profileface.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
with open("labels.pickle", 'rb') as f:
	qlabels=pickle.load(f)
	labels= {v:k for k,v in qlabels.items()}

while True:
    ret,frame=capt.read()
    #img=cv2.imread("3.jpeg")
    cgray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(cgray)
    inv_frame=cv2.flip(frame,1)
    inv_gray=cv2.cvtColor(inv_frame,cv2.COLOR_BGR2GRAY)
    ffaces=fface.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=2)
    rfaces=pface.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=2)
    lfaces=pface.detectMultiScale(inv_gray,scaleFactor=1.3,minNeighbors=2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    for x,y,w,h in ffaces:
        roi=gray[y:y+h,x:x+w]
        _id,conf=recognizer.predict(roi)
        if conf<=90:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
            cv2.putText(frame,labels[_id],(x,y+20),font,1,(255,0,0),2)
        elif conf>=90 and conf<=110:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
            cv2.putText(frame,"Not a Person",(x,y+20),font,1,(255,0,0),2)
            print(conf)
        print(roi)
    for x,y,w,h in rfaces:
        roi=gray[y:y+h,x:x+w]
        _id,conf=recognizer.predict(roi)
        if conf<=100:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(frame,labels[_id],(x,y+20),font,1,(255,0,0),2)
        elif conf>=100 and conf<=110:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(frame,"Not a Person",(x,y+20),font,1,(255,0,0),2)
        print(conf)
    for x,y,w,h in lfaces:
        roi=inv_gray[y:y+h,x:x+w]
        print(roi)
        _id,conf=recognizer.predict(roi)
        if conf<=100:
            cv2.rectangle(frame,(x+40,y),(x+w+40,y+h),(255,255,0),2)
            cv2.putText(frame,labels[_id],(x,y+20),font,1,(255,0,0),2)
        elif conf>=100 and conf<=120:
            cv2.rectangle(frame,(x+40,y),(x+w+40,y+h),(255,255,0),2)
            cv2.putText(frame,"Not a Person",(x+40,y+20),font,1,(255,0,0),2)
            print(conf)
    cv2.imshow("frame",frame)
    if cv2.waitKey(20) & 0xFF==ord(" "):
        exit(0)


                           
