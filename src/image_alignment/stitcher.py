from image_alignment.stitcher_module import (
    camera_corrector,
    feature_matcher,
    images_prepare_resolutions,
    feature_finder,
    process_timelapse,
    subset,
    feature_matcher,
    warp_images,
)
from data_manager.data_manager import get_photo_stream_paths
from stitching.feature_matcher import FeatureMatcher
import cv2 as cv


image_paths = [str(x) for x in get_photo_stream_paths()]

images_container = images_prepare_resolutions(image_paths)

features = feature_finder(images_container.medium_quality_images)
matcher = FeatureMatcher()

matches = feature_matcher(images_container.medium_quality_images, features, matcher)

images_container = subset(images_container, matcher, matches, features)

cameras = camera_corrector(features, matches)

warped_images = warp_images(images_container, cameras)

process_timelapse(
    warped_images["warped_final"],
    images_container.original_images.sizes,
    images_container.original_images.sizes,
)
