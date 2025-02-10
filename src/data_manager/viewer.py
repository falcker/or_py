import numpy as np
import skimage
import napari
from pathlib import Path

image1 = skimage.io.imread(Path(r"C:\Users\Milo\Documents\Falcker\AI\datasets\TP6\TP6 2024-11-18 13_36_29 (UTC+01)\DJI_20241118204703_0001_W_TP6-TP-SE.jpeg"))
image2 = skimage.io.imread(Path(r"C:\Users\Milo\Documents\Falcker\AI\datasets\TP6\TP6 2024-11-18 16_31_38 (UTC+01)\DJI_20241118234158_0001_W_TP6-TP-SE.jpeg"))
image3 = skimage.io.imread(Path(r"C:\Users\Milo\Documents\Falcker\AI\datasets\TP6\TP6 2024-11-19 12_08_14 (UTC+01)\DJI_20241119192036_0001_W_OR-107-96-744b0082fd9b4f7d8edbf8434d67fe5d-[TP6-TP-SE].jpeg"))
image4 = skimage.io.imread(Path(r"C:\Users\Milo\Documents\Falcker\AI\datasets\TP6\TP6 2024-11-21 10_28_37 (UTC+01)\DJI_20241121173719_0001_W_OR-107-96-744b0082fd9b4f7d8edbf8434d67fe5d-[TP6-TP-SE].jpeg"))


multiscale = [
    image1,
    image2,
    ]

# add image multiscale
viewer = napari.Viewer(ndisplay=2)
image_layer = viewer.add_image(image1)#, multiscale=True)
image_layer = viewer.add_image(image2)#, multiscale=True)

if __name__ == '__main__':
    napari.run()