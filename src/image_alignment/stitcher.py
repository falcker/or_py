from image_alignment.stitcher_module import (
    estimate_cameras,
    feature_matcher,
    resize_images,
    feature_finder,
    process_timelapse,
    subset_images,
    feature_matcher,
    warp_images,
)
from data_manager.data_manager import get_photo_stream_paths
from stitching.feature_matcher import FeatureMatcher
import cv2 as cv


image_paths = [str(x) for x in get_photo_stream_paths()]

images_container = resize_images(image_paths)

features = feature_finder(images_container.medium_quality_images)
matcher = FeatureMatcher()

matches = feature_matcher(images_container.medium_quality_images, features, matcher)

images_container = subset_images(images_container, matcher, matches, features)

cameras = estimate_cameras(features, matches)

warped_images = warp_images(images_container, cameras)

process_timelapse(
    warped_images["warped_final"],
    images_container.original_images.sizes,
    images_container.original_images.sizes,
)
