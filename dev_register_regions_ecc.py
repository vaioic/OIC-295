# Try registering smaller slices
import skimage
import numpy as np
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from tifffile import imwrite
import cv2 as cv

#imageA = rgb2gray(skimage.io.imread('./export/259270_1.tif'))
imageA = cv.imread('./export/259270_1.tif', cv.IMREAD_GRAYSCALE)
imageB = cv.imread('./export/261082_1.tif', cv.IMREAD_GRAYSCALE)

print(imageA.dtype)

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

imageA = imageA[5000:10000, 5000:10000]
imageB = imageB[5000:10000, 5000:10000]

# Initializing the warp matrix as identity
warp_matrix = np.eye(3, 3, dtype=np.float32) # Use 2x3 for MOTION_AFFINE/EUCLIDEAN/TRANSLATION

# 3. Define motion model (e.g., Affine)
warp_mode = cv.MOTION_HOMOGRAPHY

# 4. Define termination criteria
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 2500, 1e-6)

# 5. Compute the transform
(cc, warp_matrix) = cv.findTransformECC(imageA, imageB, warp_matrix, warp_mode, criteria)
# try:
#     (cc, warp_matrix) = cv.findTransformECC(np.float32(imageA), np.float32(imageB), warp_matrix, warp_mode, criteria)
# except cv.error as e:
#     print(f"ECC failed: {e}")

height, width = imageA.shape

# corrected = cv.warpAffine(imageA, warp_matrix, (width, height), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

corrected = cv.warpPerspective(imageB, warp_matrix, (width, height), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

# Use cv2.warpPerspective if using MOTION_HOMOGRAPHY
# # Try optical flow registration
# # v, u = skimage.registration.optical_flow_ilk(imageB, imageA)

# # nr, nc = imageA.shape

# # row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

# # image1_warp = skimage.transform.warp(imageB, np.array([row_coords + v, col_coords + u]), mode='edge')

# build an RGB image with the registered sequence
reg_im = np.zeros((height, width, 3))
reg_im[..., 0] = corrected
reg_im[..., 1] = imageA
reg_im[..., 2] = imageA

imwrite('./export/registered_ecc_region_homography.tif', reg_im)