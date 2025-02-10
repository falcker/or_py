from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import seaborn as sn
from pathlib import Path

import pandas as pd
from stitching.images import Images

from PIL import Image

from data_manager.data_manager import get_photo_stream_paths

from stitching.feature_detector import FeatureDetector

from stitching.feature_matcher import FeatureMatcher

from stitching.subsetter import Subsetter

from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector

from stitching.warper import Warper

from stitching.timelapser import Timelapser

from stitching.cropper import Cropper


# With the following block, we allow displaying resulting images within the notebook:
def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


# With the following block, we load the correct img paths to the used image sets:
def get_image_paths(img_set):
    return [str(path.relative_to(".")) for path in Path("imgs").rglob(f"{img_set}*")]


weir_imgs = [str(x) for x in get_photo_stream_paths()]

# Resize Images
# The first step is to resize the images to medium (and later to low) resolution. The class which can be used is the Images class. If the images should not be stitched on full resolution, this can be achieved by setting the final_megapix parameter to a number above 0.
# Images.of(images, medium_megapix=0.6, low_megapix=0.1, final_megapix=-1)

images = Images.of(weir_imgs)

medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
low_imgs = list(images.resize(Images.Resolution.LOW))
final_imgs = list(images.resize(Images.Resolution.FINAL))

original_size = images.sizes[0]
medium_size = images.get_image_size(medium_imgs[0])
low_size = images.get_image_size(low_imgs[0])
final_size = images.get_image_size(final_imgs[0])

print(
    f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px ~ 1 MP"
)
print(
    f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px ~ 0.6 MP"
)
print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px ~ 0.1 MP")
print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px ~ 1 MP")

""" 
Find Features

On the medium images, we now want to find features that can describe conspicuous elements within the images which might be found in other images as well. The class which can be used is the FeatureDetector class.

FeatureDetector(detector='orb', nfeatures=500)
"""
finder = FeatureDetector()
features = [finder.detect_features(img) for img in medium_imgs]
keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])

plot_image(keypoints_center_img, (15, 10))

""" 
Match Features

Now we can match the features of the pairwise images. The class which can be used is the FeatureMatcher class.

FeatureMatcher(matcher_type='homography', range_width=-1)
"""

matcher = FeatureMatcher()
matches = matcher.match_features(features)

""" 
We can look at the confidences, which are calculated by:

confidence = number of inliers / (8 + 0.3 * number of matches) (Lines 435-7 of this file)

The inliers are calculated using the random sample consensus (RANSAC) method, e.g. in this file in Line 425. We can plot the inliers which is shown later.
"""

conf_matrix = matcher.get_confidence_matrix(matches)
plt.imshow(conf_matrix)
df_cm = pd.DataFrame(
    conf_matrix, index=[i for i in "ABCDEFGHIJKL"], columns=[i for i in "ABCDEFGHIJKL"]
)
sn.heatmap(df_cm, annot=True)

""" 
With a confidence_threshold, which is introduced in detail in the next step, we can plot the relevant matches with the inliers: 
"""

all_relevant_matches = matcher.draw_matches_matrix(
    medium_imgs, features, matches, conf_thresh=1, inliers=True, matchColor=(0, 255, 0)
)

for idx1, idx2, img in all_relevant_matches:
    print(f"Matches Image {idx1+1} to Image {idx2+1}")
    plot_image(img, (20, 10))

""" 
Subset

Above we saw that the noise image has no connection to the other images which are part of the panorama. We now want to create a subset with only the relevant images. The class which can be used is the Subsetter class. We can specify the confidence_threshold from when a match is regarded as good match. We saw that in our case 1 is sufficient. For the parameter matches_graph_dot_file a file name can be passed, in which a matches graph in dot notation is saved.

Subsetter(confidence_threshold=1, matches_graph_dot_file=None)
"""

subsetter = Subsetter()
dot_notation = subsetter.get_matches_graph(images.names, matches)
print(dot_notation)

""" 
We now want to subset all variables we've created till here, incl. the attributes img_names and img_sizes of the ImageHandler
"""
indices = subsetter.get_indices_to_keep(features, matches)

medium_imgs = subsetter.subset_list(medium_imgs, indices)
low_imgs = subsetter.subset_list(low_imgs, indices)
final_imgs = subsetter.subset_list(final_imgs, indices)
features = subsetter.subset_list(features, indices)
matches = subsetter.subset_matches(matches, indices)

images.subset(indices)

print(images.names)
print(matcher.get_confidence_matrix(matches))

""" 
Camera Estimation, Adjustion and Correction

With the features and matches we now want to calibrate cameras which can be used to warp the images so they can be composed correctly. The classes which can be used are CameraEstimator, CameraAdjuster and WaveCorrector:

CameraEstimator(estimator='homography')
CameraAdjuster(adjuster='ray', refinement_mask='xxxxx')
WaveCorrector(wave_correct_kind='horiz')
"""


camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster()
wave_corrector = WaveCorrector()

cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)

""" 
Warp Images

With the obtained cameras we now want to warp the images itself into the final plane. The class which can be used is the Warper class:

Warper(warper_type='spherical', scale=1)
"""

warper = Warper()
# At first, we set the the medium focal length of the cameras as scale:
warper.set_scale(cameras)
# Warp low resolution images
low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
camera_aspect = images.get_ratio(
    Images.Resolution.MEDIUM, Images.Resolution.LOW
)  # since cameras were obtained on medium imgs

warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
# Warp final resolution images
final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
warped_final_masks = list(
    warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
)
final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
# We can plot the results. Not much scaling and rotating is needed to align the images. Thus, the images are only slightly adjusted in this example
plot_images(warped_low_imgs, (10, 10))
plot_images(warped_low_masks, (10, 10))
print(final_corners)
print(final_sizes)
""" 
Excursion: Timelapser

The Timelapser functionality is a nice way to grasp how the images are warped into a final plane. The class which can be used is the Timelapser class:

Timelapser(timelapse='no')

THIS IS THE OUTPUT!?!?!?!??!!?!?

"""

timelapser = Timelapser("as_is")
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(warped_final_imgs, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    plot_image(frame, (10, 10))
""" 
Crop

We can see that none of the images have the full height of the final plane. To get a panorama without black borders we can now estimate the largest joint interior rectangle and crop the single images accordingly. The class which can be used is the Cropper class:

Cropper(crop=True)
 """

cropper = Cropper()

# We can estimate a panorama mask of the potential final panorama (using a Blender which will be introduced later)
mask = cropper.estimate_panorama_mask(
    warped_low_imgs, warped_low_masks, low_corners, low_sizes
)
plot_image(mask, (5, 5))

# he estimation of the largest interior rectangle is not yet implemented in OpenCV, but a Numba Implementation by my own. You check out the details here. Compiling the Code takes a bit (only once, the compiled code is then cached for future function calls)
lir = cropper.estimate_largest_interior_rectangle(mask)
# After compilation the estimation is really fast:
lir = cropper.estimate_largest_interior_rectangle(mask)
print(lir)
plot = lir.draw_on(mask, size=2)
plot_image(plot, (5, 5))
""" By zero centering the the warped corners, the rectangle of the images within the final plane can be determined. Here the center image is shown: """
low_corners = cropper.get_zero_center_corners(low_corners)
rectangles = cropper.get_rectangles(low_corners, low_sizes)

plot = rectangles[1].draw_on(plot, (0, 255, 0), 2)  # The rectangle of the center img
plot_image(plot, (5, 5))

# Using the overlap new corners and sizes can be determined:
overlap = cropper.get_overlap(rectangles[1], lir)
plot = overlap.draw_on(plot, (255, 0, 0), 2)
plot_image(plot, (5, 5))

# With the blue Rectangle in the coordinate system of the original image (green) we are able to crop it
intersection = cropper.get_intersection(rectangles[1], overlap)
plot = intersection.draw_on(warped_low_masks[1], (255, 0, 0), 2)
plot_image(plot, (2.5, 2.5))

# Using all this information we can crop the images and masks and obtain new coreners and sizes
cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

cropped_low_masks = list(cropper.crop_images(warped_low_masks))
cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

lir_aspect = images.get_ratio(
    Images.Resolution.LOW, Images.Resolution.FINAL
)  # since lir was obtained on low imgs
cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

# Redo the timelapse with cropped Images:
timelapser = Timelapser("as_is")
timelapser.initialize(final_corners, final_sizes)

# export images
output_dir = Path(r"C:\Users\Gebruiker\Documents\GitHub\or_py\data\output")
for idx, (img, corner) in enumerate(zip(cropped_final_imgs, final_corners)):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    # print(frame, type(frame))
    im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    im.save(output_dir / f"{idx+1}.jpg")
    # plot_image(frame, (10, 10)) (for timelapse)


# https://github.com/OpenStitching/stitching_tutorial/blob/master/Stitching%20Tutorial.ipynb
