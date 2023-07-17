# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# 

# %%
orig_img = cv2.imread("./predataset/eascii1.png", cv2.IMREAD_COLOR)
table = cv2.inRange(orig_img, (0, 0, 240), (0, 0, 255))
txtimg = cv2.inRange(orig_img, (0, 0, 0), (32, 32, 32))

print(table.max())

gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

table = cv2.bitwise_and(gray_img, gray_img, mask=table)
txtimg = cv2.bitwise_not(gray_img, mask=txtimg)

print(table.max())

txtimg_n = cv2.bitwise_not(txtimg)
#table = cv2.bitwise_and(table, txtimg_n)

ret, table = cv2.threshold(table, 5, 255, cv2.THRESH_BINARY)
ret, txtimg = cv2.threshold(txtimg, 180, 255, cv2.THRESH_BINARY)

cv2.imwrite("ee.png", table)

print(table.max())

pointList = np.array(table)
pointList = np.nonzero(pointList)

print(pointList)


# %%
plt.imshow(orig_img)
plt.show()
plt.imshow(table)
plt.show()
plt.imshow(txtimg)
plt.show()

plt.imshow(txtimg_n)
plt.show()

# %%
cv2.imwrite("cc.png", txtimg)


