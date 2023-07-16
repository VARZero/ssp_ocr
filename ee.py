import cv2

orig_img = cv2.imread("./predataset/ascii1.png", cv2.IMREAD_COLOR)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
table = cv2.inRange(orig_img, (16, 16, 16), (210, 210, 210))
txtimg = cv2.inRange(orig_img, (0, 0, 0), (16, 16, 16))

cv2.imshow("", orig_img)
cv2.imshow("", table)
cv2.imshow("", txtimg)
