import cv2
import numpy as np

txtimg = cv2.imread('exm1.png', cv2.IMREAD_UNCHANGED)
txtimg = txtimg[:,:,3]

imgHalfHgt = int(txtimg.shape[0] / 2)

ret, thresh = cv2.threshold(txtimg, 1, 255, 0)
cnt, hiey = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

PreBoxArea = list()
BoxArea = list()

for b in cnt:
    x, y, w, h = cv2.boundingRect(b)
    PreBoxArea.append([x, 0, x+w, y+h])

nowX, nowW = 0
for PbA in PreBoxArea:
    if len(BoxArea) == 0:
        BoxArea.append(PbA)
        nowX = PbA[0]
        nowW = PbA[2]
        continue
    for bA in BoxArea:
        if bA