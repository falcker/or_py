from pathlib import Path
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
from config import PACKAGE_ROOT
from stitching.images import Images
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


def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def get_image_paths(img_set):
    return [str(path.relative_to(".")) for path in Path("imgs").rglob(f"{img_set}*")]


def load_images():
    weir_imgs = [str(x) for x in get_photo_stream_paths()]
    return Images.of(weir_imgs)


def resize_images(images):
    return {
        "medium": list(images.resize(Images.Resolution.MEDIUM)),
        "low": list(images.resize(Images.Resolution.LOW)),
        "final": list(images.resize(Images.Resolution.FINAL)),
    }


def detect_features(medium_imgs):
    finder = FeatureDetector()
    return [finder.detect_features(img) for img in medium_imgs]


def match_features(features):
    matcher = FeatureMatcher()
    return matcher.match_features(features)


def subset_images(images, features, matches):
    subsetter = Subsetter()
    indices = subsetter.get_indices_to_keep(features, matches)
    images.subset(indices)
    return images, indices


def estimate_cameras(features, matches):
    camera_estimator = CameraEstimator()
    camera_adjuster = CameraAdjuster()
    wave_corrector = WaveCorrector()
    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    return wave_corrector.correct(cameras)


def warp_images(images, cameras):
    warper = Warper()
    warper.set_scale(cameras)
    return {
        "warped_low": list(warper.warp_images(images["low"], cameras, 1)),
        "warped_final": list(warper.warp_images(images["final"], cameras, 1)),
    }


def process_timelapse(warped_final_imgs, final_corners, final_sizes):
    timelapser = Timelapser("as_is")
    timelapser.initialize(final_corners, final_sizes)
    for img, corner in zip(warped_final_imgs, final_corners):
        timelapser.process_frame(img, corner)


def crop_images(warped_low_imgs, warped_low_masks, low_corners, low_sizes):
    cropper = Cropper()
    mask = cropper.estimate_panorama_mask(
        warped_low_imgs, warped_low_masks, low_corners, low_sizes
    )
    lir = cropper.estimate_largest_interior_rectangle(mask)
    lir.draw_on(mask, size=2)
    low_corners = cropper.get_zero_center_corners(low_corners)
    rectangles = cropper.get_rectangles(low_corners, low_sizes)
    overlap = cropper.get_overlap(rectangles[1], lir)
    intersection = cropper.get_intersection(rectangles[1], overlap)

    cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
    return {
        "cropped_low_imgs": list(cropper.crop_images(warped_low_imgs)),
        "cropped_low_masks": list(cropper.crop_images(warped_low_masks)),
    }


def export_images(images, output_dir):
    output_dir = PACKAGE_ROOT / "data/output/stitched2"
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images):
        im = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        im.save(output_dir / f"{idx+1}.jpg")


def main():
    images = load_images()
    resized_images = resize_images(images)
    features = detect_features(resized_images["medium"])
    matches = match_features(features)
    images, indices = subset_images(images, features, matches)
    cameras = estimate_cameras(features, matches)
    warped_images = warp_images(resized_images, cameras)
    process_timelapse(warped_images["warped_final"], images.sizes, images.sizes)
    cropped_images = crop_images(warped_images["warped_low"], [], [], [])
    export_images(
        cropped_images["cropped_low_imgs"], PACKAGE_ROOT / "data/output/stitching4"
    )


if __name__ == "__main__":
    main()
