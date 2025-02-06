import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread(r"E:\AI\Operator Rounds\sift\data\1_TP6-TP-SE\DJI_20241118204703_0001_W_TP6-TP-SE.jpeg",0)          # queryImage
img2 = cv2.imread(r"E:\AI\Operator Rounds\sift\data\1_TP6-TP-SE\DJI_20241118234158_0001_W_TP6-TP-SE.jpeg",0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

des1 = np.float32(des1)
des2 = np.float32(des2)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()

# store only good matches as before in good
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# extract coordinated for query and train image points
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
# use homography to get the M transformation matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
# here im1 is the original RGB (or BGR because of cv2) image
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

# use this transform to shift train image to query image
dst = cv2.perspectiveTransform(pts,M)
# for now I only displayed it using polylines, depending on what you need you can use these points to do something else
# I overwrite the original RGB (or BGR) image with a red rectangle where the smaller image should be
img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)
cv2.imwrite('image_overlap.png', img2) 