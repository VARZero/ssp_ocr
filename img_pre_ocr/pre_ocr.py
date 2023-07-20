# %%
import cv2
import numpy as np
import matplotlib as plt

# %%
txtimg = cv2.imread('exm1.png', cv2.IMREAD_UNCHANGED)
a = txtimg
txtimg = txtimg[:,:,3]
print(txtimg.shape)
ret, thresh = cv2.threshold(txtimg,1,255,0)

cnt, hiey = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(cnt)

for b in cnt:
    x, y, w, h = cv2.boundingRect(b)
    a = cv2.rectangle(a,(x,y),(x+w, y+h),(0,255,0), 3) # green
    cv2.imshow('e', a)

#for c in cnt:
#    rect = cv2.minAreaRect(c)
#    box = cv2.boxPoints(rect)
#    box = np.intp(box)
#    a = cv2.drawContours(a, [box], 0, (0,0,255), 3)
#    cv2.imshow('e', a)

# %%
#cv2.imshow("e", a)
cv2.waitKey(0)
cv2.destroyAllWindows()


