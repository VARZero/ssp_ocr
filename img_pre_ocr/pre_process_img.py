import cv2
import numpy as np

txtimg = cv2.imread('exm1.png', cv2.IMREAD_UNCHANGED)
a = txtimg
txtimg = txtimg[:,:,3]

imgHalfHgt = int(txtimg.shape[0] / 2)

ret, thresh = cv2.threshold(txtimg, 1, 255, 0)
cnt, hiey = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

PreBoxArea = list()
BoxArea = list()

for b in cnt:
    x, y, w, h = cv2.boundingRect(b)
    PreBoxArea.append([x, 0, x+w, y+h])

print(PreBoxArea)

PreBoxArea.sort(key=lambda x:x[0])

print(PreBoxArea)

for pBA in PreBoxArea:
    BoxArea.append(pBA)
    for b in BoxArea[:-1]:
        if pBA[0] > b[2]: continue
        elif pBA[2] < b[0]: break

        BoxArea.pop()
        if pBA[0] <= b[0]: b[0] = pBA[0]
        elif pBA[2] >= b[2]: b[2] = pBA[2]

n = 1
for b in BoxArea:
    a = cv2.rectangle(a,(b[0],b[1]),(b[2],b[3]),(0,25*n,0), 3) # green
    cv2.imshow('e', a)
    n+=1

charIMG = list()
for b in BoxArea:
    cImg = txtimg[0:b[3], b[0]:b[2]]
    cv2.imshow("",cImg)
    cv2.waitKey(0)
cv2.destroyAllWindows()