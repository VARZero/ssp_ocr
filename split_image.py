import os
import cv2
import numpy as np

def tableGetPoint(img):
    table = cv2.inRange(img, (0, 0, 240), (0, 0, 255))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    table = cv2.bitwise_and(gray_img, gray_img, mask=table)
    ret, table = cv2.threshold(table, 5, 255, cv2.THRESH_BINARY)

    pointList = np.array(table)
    pointList = np.nonzero(pointList)

    boxStartPoint = [pointList[1][0], pointList[0][0]]
    boxEndPoint = [pointList[1][2], pointList[0][2]]
    rolMax = [pointList[1][1], pointList[0][1]]
    colMax = [pointList[1][3], pointList[0][3]]

    boxSize = [boxEndPoint[0] - boxStartPoint[0], boxEndPoint[1] - boxStartPoint[1]]

    rolMaxCount = int((rolMax[0] - boxStartPoint[0]) / boxSize[1])
    colMaxCount = int((colMax[1] - boxStartPoint[1]) / boxSize[1])

    return boxStartPoint, boxEndPoint, boxSize, rolMaxCount, colMaxCount

def txtimgGet(img):
    txtimg = cv2.inRange(img, (0, 0, 0), (32, 32, 32))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    txtimg = cv2.bitwise_not(gray_img, mask=txtimg)
    ret, txtimg = cv2.threshold(txtimg, 180, 255, cv2.THRESH_BINARY)

    return txtimg

def splitImg(textstart, txtimg, startpoint, endpoint, size, rolCount, colCount):
    os.makedirs('dataset',exist_ok=True)
    for d in range(colCount):
        os.makedirs('dataset/'+str(textstart),exist_ok=True)
        for f in range(rolCount):
            splitOne = txtimg[startpoint[1]+(size[1]*d):endpoint[1]+(size[1]*d), startpoint[0]+(size[0]*f):endpoint[0]+(size[0]*f)]
            rsSo = cv2.resize(splitOne, dsize=(28,28), interpolation=cv2.INTER_LINEAR)
            rsSo = cv2.rotate(rsSo, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rsSo = cv2.flip(rsSo, 0)
            cv2.imwrite('dataset/'+str(textstart)+'/'+str(f)+'.png', rsSo)
        textstart += 1

startTable = cv2.imread('predataset/eascii1.png')
boxSP, boxEP, size, rol, col = tableGetPoint(startTable)

for imgC in range(5):
    img = cv2.imread('predataset/ascii'+str(imgC+1)+'.png')
    timg = txtimgGet(img)
    splitImg(1+(col*imgC), timg, boxSP, boxEP, size, rol, col)

print('end')