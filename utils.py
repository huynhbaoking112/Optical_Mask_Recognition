import cv2
import numpy as np

def rectContour(contours):
    
    rectCon = []

    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True )
            # print(len(approx))
            if len(approx)==4:
                rectCon.append(i)
    
    rectCon = sorted(rectCon, key = cv2.contourArea, reverse=True)

    return rectCon


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02*peri, True )
    return approx


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)

    # print(myPoints)
    # print(add)
    # [[ 402  253]
    # [ 298  843]
    # [1030  927]
    # [1013  338]]
    # [ 655 1141 1957 1351]
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1 )
    # diff = [[253 - 402],  # Kết quả: [-149]
    #     [843 - 298],  # Kết quả: [545]
    #     [927 - 1030], # Kết quả: [-103]
    #     [338 - 1013]] # Kết quả: [-675]
    myPointsNew[1] = myPoints[np.argmin(diff)] #[w, 0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h, 0]

    return myPointsNew
    
    """
        =>  Sắp xếp lại các phần tử theo thứ tự x, x+w, y, y+h
        =>  Chuyển đổi (4, 1, 2) thành (4, 2) 
    """

def splitBoxes(img):
    rows = np.vsplit(img, 5)
    
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)

    return boxes

def showAnswers(img, myIndex, grading, ans,questions, choices):

    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)

    for x in range(0, questions):
        index = myIndex[x]
        first = index * secW 
        last = index * secW + secW
        firstY = x * secH
        lastY = x * secH + secH        
        xTam = int((first+last)/2)
        yTam = int((firstY+lastY)/2)
        # cv2.rectangle(img, (first, firstY), (last, lastY), (0, 0, 255), 2)
        cv2.circle(img, (xTam, yTam), 50, (0,0,255),-1)
   
    for x in range(0, questions):
        index = ans[x]
        first = index * secW 
        last = index * secW + secW
        firstY = x * secH
        lastY = x * secH + secH        
        xTam = int((first+last)/2)
        yTam = int((firstY+lastY)/2)
        # cv2.rectangle(img, (first, firstY), (last, lastY), (0, 0, 255), 2)
        cv2.circle(img, (xTam, yTam), 50, (0,255,0),-1)

    return img

def showAnswersGrade(img, score):

    secW = int(img.shape[1])
    secH = int(img.shape[0])

    tamX = int(secW/2)
    tamY = int(secH/2)

    cv2.putText(img, str(score), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 2)


    return img