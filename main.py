import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

######################
path = 'phieu2.jpg'
heightImg = 700
widthImg  = 700
question = 5 
choices = 5
ans = [1, 2, 0, 1, 4]
webcam = False
######################

cap = cv2.VideoCapture(0)
cap.set(10, 150)

while True:

    if webcam: success, img = cap.read()
    else:
        img = cv2.imread(path)


    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()



    # Tiền xử lí
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)


    # Áp dụng findContours tìm đường viền
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1,  (0,255,0), 2)
    #Find Rectangles
    rectCon = rectContour(contours)
    biggestContour = getCornerPoints(rectCon[0])
    gradePoints = getCornerPoints(rectCon[1])


    if biggestContour.size != 0 and gradePoints.size !=0:
        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 50)
        cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0),50)

        # Sắp xếp lại các điểm của khung tô đáp án
        biggestContour =  reorder(biggestContour)
        gradePoints =  reorder(gradePoints)


        # Thực hiện cắt khung tô đáp án
        pt1 = np.float32(biggestContour)
        pt2 = np.float32([[0,0],[widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        imgWarpColored = cv2.warpPerspective(img, matrix,  (widthImg, heightImg))

        # Thực hiện cắt khung tô điểm
        ptG1 = np.float32(gradePoints)
        ptG2 = np.float32([[0,0],[325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG,  (325, 150))

        # Áp dụng lọc ngưỡng để  nhận diện đáp án khoanh
        imgWrapGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWrapGray, 170, 255, cv2.THRESH_BINARY_INV)[1]


        # Cắt đáp án 
        boxes = splitBoxes(imgThresh)
        # cv2.imshow("Test", boxes[2])
        # print(cv2.countNonZero(boxes[2]))
        # print(cv2.countNonZero(boxes[1]))

        # Đưa đáp án khoanh vào một mảng
        myPixelVal = np.zeros((question, choices))
        countC = 0
        countR = 0
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if(countC == choices):countR += 1 ; countC = 0
        """
        [[ 4251. 10269.  2688.  2953.  3305.]
        [ 2876.  2302. 10670.  2249.  2624.]
        [ 9560.  2231.  2018.  2223.  2642.]
        [ 6422.  2167.  1947.  2199.  2676.]
        [ 3231.  2718.  2506.  2674.  6349.]]
        """
        # Tìm vị trí index của điểm đánh dấu
        myIndex = []
        for x in range (0, question):
            arr = myPixelVal[x]
            # myIndexVal = np.where(arr==np.amax(arr))
            # myIndex.append(myIndexVal[0][0])
            myIndex.append(int(np.argmax(arr)))
        #print(myIndex)

        # Chấm điểm
        grading = []
        for x in range (0, question):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        #print(grading)
        score = (sum(grading)/question) * 10 # Điểm cuối cùng
        # print(score)

        # Hiển thị kết quả
        # imgResult = imgWarpColored.copy()
        # imgResult =  showAnswers(imgResult, myIndex, grading, ans, question, choices)
        imRawDrawing = np.zeros_like(imgWarpColored)
        imRawDrawing = showAnswers(imRawDrawing,  myIndex, grading, ans, question, choices )
        invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
        imgInvWrap = cv2.warpPerspective(imRawDrawing, invMatrix,  (widthImg, heightImg))
        
        # imgResultGrade = imgGradeDisplay.copy()
        # imgResultGrade =  showAnswersGrade(imgResultGrade,score)
        imgResultGradeClone = np.zeros_like(imgGradeDisplay)
        imgResultGradeClone =  showAnswersGrade(imgResultGradeClone,score)
        matrixGrade2 = cv2.getPerspectiveTransform(ptG2, ptG1)
        imgGradeInvWrap = cv2.warpPerspective(imgResultGradeClone, matrixGrade2, (widthImg, heightImg))
        
        
        
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWrap, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgGradeInvWrap, 1, 0)

        cv2.imshow("asd", imgFinal)

        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite("FinalResult.jpg", imgFinal)

        
cv2.destroyAllWindows()
cap.release() 

# Danh sách các ảnh và tiêu đề tương ứng
imageArray = [img, imgGray, imgBlur, imgCanny, imgContours, imgBiggestContours, imgWarpColored, imgThresh , imRawDrawing, imgInvWrap, imgFinal]
titles = ['Original', 'Grayscale', 'Blurred', 'Canny', 'ImgContours', "imgBiggestContours","imgWarpColored", "imgThresh", "imRawDrawing", "imgInvWrap", "imgFinal"]
# Tạo figure và hiển thị các ảnh
plt.figure(figsize=(15, 5))  # Điều chỉnh kích thước của figure nếu cần
for i in range(len(imageArray)):
    plt.subplot(3, 4, i + 1)  # Sắp xếp theo lưới 3x4
    if len(imageArray[i].shape) == 2:  # Kiểm tra xem ảnh có phải là grayscale không
        plt.imshow(imageArray[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(imageArray[i], cv2.COLOR_BGR2RGB))  # Chuyển ảnh từ BGR sang RGB
    plt.title(titles[i])
    plt.axis('off')  # Tắt hiển thị trục tọa độ
plt.tight_layout()  # Để sắp xếp khoảng cách giữa các ảnh hợp lý
plt.show()


