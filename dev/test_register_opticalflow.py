import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
# Attempting to remake the script to see if I can improve the registration result
from oic_toolkit import display, register

#ROI = [  #[top left height width]

# target = sk.io.imread("../data/2026-06-08 Downsampled/30239_Ki67_267466.tif")
# moving = sk.io.imread("../data/2026-06-08 Downsampled/30239_H3K9me3_267412.tif")

target = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_Ki67_267466.ome.tif")
moving = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_H3K9me3_267412.ome.tif")

# target = sk.color.rgb2gray(target)
# target = sk.util.invert(target)

# moving = sk.color.rgb2gray(moving)
# moving = sk.util.invert(moving)

target = sk.color.rgb2gray(target)
moving = sk.color.rgb2gray(moving)
target, moving = register.pad_images(target, moving)

v, u = sk.registration.optical_flow_ilk(target, moving, radius=10, prefilter=True)

nr, nc = target.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

image1_warp = sk.transform.warp(moving, np.array([row_coords + v, col_coords + u]), mode='edge')

merge = display.merge_images(target, image1_warp)

plt.imshow(merge)
plt.show()

