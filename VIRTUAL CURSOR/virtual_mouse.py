import cv2 as cv
import time as time
import numpy as np
import mediapipe as mp                                  #library provided by google for hand,face detection using direct APIs
import HandTrackingModule as htm
import pyautogui
import pickle

heightW,widthW = 480,640
frameRed=100                                            #frame reduction initializer
camera=cv.VideoCapture(0)
camera.set(3,widthW)
camera.set(4,heightW)
mphands = mp.solutions.hands                            #taking solution of hands for tracking
hands = mphands.Hands()                                 #taking hands 
mpdraw = mp.solutions.drawing_utils                     #taking solutions to draw connections and landmarks
PreviousTime=0                                          #past time,for frame calculation
CurrentTime=0                                           #current time,for frame calcultion

detector = htm.HandDetector(maxhands=1)                 #it refers that camera will detect only one hand if multiple hands are shown
widthS,heightS=pyautogui.size()
    
while True:
    #for accessing the camera of device
    result,image=camera.read()                          #the output will return result as boolean value
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)           #because RGB format only is supported by pyautogui
    image=cv.flip(image,1)                              #1: fliping the image on y-axis 
    
    
    #for generating the frame Rate
    CurrentTime=time.time()
    fps=1/(CurrentTime-PreviousTime)
    PreviousTime=CurrentTime
    Original=cv.putText(image,str(int(fps)),(20,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    #for plotting hand landmarks 
    image=detector.FindHands(image)
    lmlist=detector.FindPosition(image)
    
    if len(lmlist) != 0:
        #first finger 
        x1,y1=lmlist[8][1:]                            #8 : landmark assigned to index finger
        #middle finger
        x2,y2=lmlist[12][1:]                           #12 : landmark assigned to middle finger
        #print(x1,y1,x2,y2)
        
        #check which finger is pointing upward
        fingers=detector.fingersup()                   #creates detector as an object to find which finger is active/up
        #print(fingers)
        
        #check only index finger is upwar
        if fingers[1]==1 and fingers[2]==0 :
            #setting up region to capture proper gestures and proper functionality all around device screen
            cv.rectangle(image,(frameRed,frameRed),(widthW-frameRed,heightW-frameRed),(255,0,255),2)
            #converting coordinates
            x3=np.interp(x1,(frameRed,widthW-frameRed),(0,widthS))
            y3=np.interp(y1,(frameRed,heightW-frameRed),(0,heightS))
            
            #Mouse movement
            pyautogui.moveTo(x3,y3)    
            cv.circle(image,(x1,y1),15,(255,0,255),cv.FILLED)
            
        #both index finger and middle finger are upward: enable clicking mode
        if fingers[1]==1 and fingers[2]==1:
            #if both are up (index finger and middle finger),then calculate the distance
            #if distance is less --> clickinig mode on , if distance is more --> clicking mode off
            length,image, lineinfo =detector.findDistance(8,12,image)
            print(length)
            if length < 40 :
                #if distance is less then 40 : enables clicking , helps user with green indication
                cv.circle(image, (lineinfo[4], lineinfo[5]),15, (0, 255, 0), cv.FILLED)
                #for initializing clicking ability
                pyautogui.click()
            
     
    #showing output 
    cv.imshow('User Gestures',image)
    cv.waitKey(1)

    #for deployment , pickle library is used to create a model.pkl file for easy process
    #pickle.dump(open('virtual_mouse.pkl','wb'))
    


    
    



