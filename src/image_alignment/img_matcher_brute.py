from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread(
    r"C:\Users\Gebruiker\Documents\GitHub\or_py\data\input\DJI_20241119192135_0004_Z_OR-107-96-0c165526932148249e6e62ec026dc476-[541-ST].jpeg"
)
img2 = cv.imread(
    r"C:\Users\Gebruiker\Documents\GitHub\or_py\data\input\DJI_20241121173814_0004_Z_OR-107-96-0c165526932148249e6e62ec026dc476-[541-ST].jpeg"
)
output_dir = Path(r"C:\Users\Gebruiker\Documents\GitHub\or_py\data\output\sift")

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# img=cv.drawKeypoints(gray,kp,img)
# cv.imwrite('sift_keypoints.jpg',img)
bf = cv.BFMatcher()

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda val: val.distance)

out = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
plt.imshow(out), plt.show()
cv.imwrite(output_dir / "bfmatches.jpg", out)
