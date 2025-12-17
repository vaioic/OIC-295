# Try registering smaller slices
import skimage
import numpy as np
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

imageA = rgb2gray(skimage.io.imread('./export/259270_1.tif'))
imageB = rgb2gray(skimage.io.imread('./export/261082_1.tif'))

# Condition the images to have the same shape


v, u = skimage.registration.optical_flow_ilk(imageB, imageA)

nr, nc = imageA.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

image1_warp = skimage.tranform.warp(imageB, np.array([row_coords + v, col_coords + u]), mode='edge')

# build an RGB image with the registered sequence
reg_im = np.zeros((nr, nc, 3))
reg_im[..., 0] = image1_warp
reg_im[..., 1] = imageA
reg_im[..., 2] = imageA

plt.imshow(reg_im)
plt.show()