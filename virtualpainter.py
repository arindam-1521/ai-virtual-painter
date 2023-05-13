#first install these modules.
import cv2
import numpy as np
import time
import os
import handtrackmodule as htm


folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]
drawColor = (196, 200, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# address = "https://192.168.29.28:8080/video"
# cap.open(address)

detector = htm.handDetector(detectionCon=0.5)
xp,yp = 0, 0

imgCanvas = np.zeros((720,1280,3),np.uint8)
thickness = 15

while True:
    # 1.import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2.finding the hands landmark
    img = detector.findHands(img)
    lmList = detector.findPosition((img), draw=False)

    if len(lmList) != 0:

        # print(lmList)

        # Tip of index and middle finger
        # for the index finger
        x1, y1 = lmList[8][1:]
        # for the middle finger
        x2, y2 = lmList[12][1:]

        # 3.Checking when fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4.if selection mode -if two fingers are up,we haeve to select
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            # checking for the click
            #changing the colors according to the position.
            if y1 < 122:
                if 290 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (196, 200, 0)
                elif 450 < x1 < 610:
                    header = overlayList[1]
                    drawColor = (22,23, 253)
                elif 610 < x1 < 770:
                    header = overlayList[2]
                    drawColor = (192, 101, 253)
                elif 770 < x1 < 930:
                    header = overlayList[3]
                    drawColor = (53, 253, 102)
                elif 980 < x1 < 1200:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),drawColor, cv2.FILLED)

        # 5.if drawing mode is selected -when one finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15,drawColor, cv2.FILLED)
            # print("Drawing Mode")
            if(xp == 0 and yp == 0):
                xp = x1
                yp = y1
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,100)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,100)

            cv2.line(img,(xp,yp),(x1,y1),drawColor,8)
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,thickness)

            xp,yp = x1,y1


    imgGrey = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # overlaying the image
    img[0:122, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    # cv2.imshow("canvas",imgCanvas)
    cv2.waitKey(1)