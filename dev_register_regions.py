# Try registering smaller slices
import skimage
import numpy as np
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from tifffile import imwrite

imageA = rgb2gray(skimage.io.imread('./export/259270_1.tif'))
imageB = rgb2gray(skimage.io.imread('./export/261082_1.tif'))

# Condition the images to have the same shape
height_A, width_A = imageA.shape
height_B, width_B = imageB.shape

target_height = max(height_A, height_B)
target_width = max(width_A, width_B)

if height_A < target_height:
    diff = target_height - height_A
    imageA = np.pad(imageA, ((0, diff), (0, 0)), 'constant')

if width_A < target_width:
    diff = target_width - width_A
    imageA = np.pad(imageA, ((0, 0), (0, diff)), 'constant')

if height_B < target_height:
    diff = target_height - height_B
    imageB = np.pad(imageB, ((0, diff), (0, 0)), 'constant')

if width_B < target_width:
    diff = target_width - width_B
    imageB = np.pad(imageB, ((0, 0), (0, diff)), 'constant')

# min_height = min(height_A, height_B)
# min_width = min(width_A, width_B)

# print(f"Min dimensions: {(min_height, min_width)}")

# if height_A > min_height:
#     imageA = imageA[:min_height, :]

# if width_A > min_width:
#     imageA = imageA[:, :min_width]

# if height_B > min_height:
#     imageB = imageB[:min_height]

# if width_B > min_width:
#     imageB = imageB[:, :min_width]

print(imageA.shape)
print(imageB.shape)

imageA = imageA[5000:10000, 5000:10000]
imageB = imageB[5000:10000, 5000:10000]

shift, _, _ = skimage.registration.phase_cross_correlation(imageA, imageB)

tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))

corrected = skimage.transform.warp(imageB, tform)

# # Try optical flow registration
# # v, u = skimage.registration.optical_flow_ilk(imageB, imageA)

# # nr, nc = imageA.shape

# # row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

# # image1_warp = skimage.transform.warp(imageB, np.array([row_coords + v, col_coords + u]), mode='edge')

# build an RGB image with the registered sequence
reg_im = np.zeros((imageA.shape[0], imageA.shape[1], 3))
reg_im[..., 0] = corrected
reg_im[..., 1] = imageA
reg_im[..., 2] = imageA

imwrite('./export/registered_phasexcorr_region.tif', reg_im)