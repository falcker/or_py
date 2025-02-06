import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
img1 = cv.imread(r"E:\AI\Operator Rounds\sift\data\1_TP6-TP-SE\DJI_20241118204703_0001_W_TP6-TP-SE.jpeg")
img2 = cv.imread(r"E:\AI\Operator Rounds\sift\data\1_TP6-TP-SE\DJI_20241118234158_0001_W_TP6-TP-SE.jpeg")

gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
 
sift = cv.SIFT_create()
kp1,des1 = sift.detectAndCompute(gray1,None)
kp2,des2 = sift.detectAndCompute(gray2,None)
 
# img=cv.drawKeypoints(gray,kp,img)
# cv.imwrite('sift_keypoints.jpg',img)
bf = cv.BFMatcher()

matches = bf.match(des1,des2)

matches = sorted(matches, key= lambda val: val.distance)

out = cv.drawMatches(img1,kp1,img2,kp2, matches[:50], None, flags=2)
plt.imshow(out), plt.show()
cv.imwrite('bfmatches.jpg', out)