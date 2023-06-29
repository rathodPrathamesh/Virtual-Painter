import cv2
import numpy as np
import os
import HandTrackingModule as htm

mylist = os.listdir("brushes")
mylist.sort()

overlayList = []
for imgpath in mylist:
    image = cv2.imread(f'brushes/{imgpath}')
    overlayList.append(image)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

imgCanvas = np.zeros((720,1280,3), np.uint8)
brushthickness = 10
erasersize = 100
drawcolor = (255,0,255)

header = overlayList[0]
detector = htm.handDetector(detection_confidence=0.9)

xp, yp = 0,0

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    #header image
    img[0:125,0:1280] = header

    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # tip of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

    #check fingers UP
        fingers = detector.fingersUP()

    #If selection made
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #checking for the click
            if y1<125:
                if 150<x1<350:
                    header=overlayList[1]
                    drawcolor = (255,0,255)
                if 400 < x1 < 550:
                    header = overlayList[2]
                    drawcolor = (255, 0, 100)
                if 700<x1<850:
                    header=overlayList[3]
                    drawcolor = (0, 255, 0)
                if 900<x1<1200:
                    header=overlayList[4]
                    drawcolor = (0, 0, 0)
            cv2.circle(img, ((x1 + x2) // 2, y1), 35, (0, 250, 0), cv2.FILLED)
        else:
            if drawcolor == (0, 0, 0):
                cv2.circle(img, (x1,y1), 50, drawcolor, cv2.FILLED)
            else:
                cv2.circle(img, (x1, y1), 20, drawcolor, cv2.FILLED)

            if xp==0 and xp==0:
                xp, yp = x1,y1

            if drawcolor == (0, 0, 0):
                cv2.line(img,(xp,yp),(x1,y1), drawcolor, erasersize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, erasersize)
                xp, yp = x1, y1
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
                xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)



    cv2.imshow("Drawing Space", img)
    cv2.waitKey(1)
