# import the necessary packages
from imutils import paths
import imageio
import scipy

import numpy as np
from scipy.ndimage import correlate

# from scipy.linalg import norm
# from scipy.sparse import sum, average, to_grayscale

# https://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images

crop = 1
output = (
    r"C:\Users\Gebruiker\Documents\GitHub\or_py\data\output\cv2_stitch\stitched.jpg"
)
print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["images"])))
imagePaths = sorted(
    list(paths.list_images(r"C:\Users\Gebruiker\Documents\GitHub\or_py\data\input"))[
        0:2
    ]
)

img1 = imageio.imread(imagePaths[0])
img1_grey = to_grayscale(img1)
