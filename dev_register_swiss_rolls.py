import core_functions
import skimage
from matplotlib import pyplot as plt
import scipy
import numpy as np
import cv2
import tifffile
from scipy.signal.windows import hann

def apply_window(image):
    win_2d = np.outer(hann(image.shape[0]), hann(image.shape[1]))
    return image * win_2d


target = skimage.io.imread("target_ds.tiff")
moving = skimage.io.imread("moving_ds.tiff")

target, moving = core_functions.match_image_size(target, moving)

# Convert to just blue
target_hed = skimage.color.rgb2hed(target)
moving_hed = skimage.color.rgb2hed(moving)

# Renormalize the blue stain to show the tissue shape
target_h_norm = target_hed[:, :, 0].copy()

h_min = np.percentile(target_h_norm, 25)
h_max = np.percentile(target_h_norm, 50)

target_h_norm = (target_h_norm - h_min) / (h_max - h_min)
target_h_norm = np.clip(target_h_norm, 0, 1)

# Renormalize the blue stain to show the tissue shape
moving_h_norm = moving_hed[:, :, 0].copy()

h_min = np.percentile(moving_h_norm, 25)
h_max = np.percentile(moving_h_norm, 50)

moving_h_norm = (moving_h_norm - h_min) / (h_max - h_min)
moving_h_norm = np.clip(moving_h_norm, 0, 1)

# plt.imshow(moving_h_norm)
# plt.show()

# Perform a coarse alignment

# shift, error, diffphase = skimage.registration.phase_cross_correlation(target_h_norm, moving_h_norm, upsample_factor=1)
# print(shift)

# corrected = core_functions.translate_image(moving, shift)

# overlay = core_functions.merge_images(target, corrected)

# plt.imshow(overlay)
# plt.show()

# exit()


# shift = core_functions.get_shift(apply_window(target_h_norm), apply_window(moving_h_norm), downsample_factor=None)

# print(shift)

# corrected = core_functions.translate_image(moving_h_norm, shift)

# overlay = core_functions.merge_images(target_h_norm, corrected)

# plt.imshow(overlay)
# plt.show()

# exit()


# 2. Pre-process: Normalize and Blur
# This removes the "cell-level" noise and focuses on tissue architecture
t_smooth = skimage.filters.gaussian(skimage.exposure.rescale_intensity(target), sigma=3)
m_smooth = skimage.filters.gaussian(skimage.exposure.rescale_intensity(moving), sigma=3)

# 3. Create Edge Maps (Sobel)
# This is the secret sauce for histology—it makes staining differences irrelevant
t_edges = skimage.filters.sobel(t_smooth)
m_edges = skimage.filters.sobel(m_smooth)

dX, dY, dX_c, dY_c = core_functions.get_fine_shift(m_edges, t_edges, 15, 200, debug_plot=True)

# Might want to apply some kind of smoothing filter

# Interpolate the displacements to the whole image
interp_dX = scipy.interpolate.RegularGridInterpolator((dX_c, dY_c), dX.T, bounds_error=False, fill_value=None)
interp_dY = scipy.interpolate.RegularGridInterpolator((dX_c, dY_c), dY.T, bounds_error=False, fill_value=None)

# Generate image coordinates
iX = np.arange(0, moving.shape[1])
iY = np.arange(0, moving.shape[0])

iXX, iYY = np.meshgrid(iX, iY, indexing='xy')

# Upsample the displacement field
dX_upsampled = interp_dX((iXX, iYY))
dY_upsampled = interp_dY((iXX, iYY))

# plt.subplot(2, 2, 1)
# plt.imshow(dX)
# plt.colorbar()
# plt.subplot(2, 2, 2)
# plt.imshow(dX_upsampled)
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.imshow(dY)
# plt.colorbar()
# plt.subplot(2, 2, 4)
# plt.imshow(dY_upsampled)
# plt.colorbar()
# plt.show()

corrected_final = core_functions.remap_image(moving, iXX, iYY, dX_upsampled, dY_upsampled)

corrected_final = skimage.util.img_as_float(corrected_final)
crop_target = skimage.util.img_as_float(target)

merged = core_functions.merge_images(crop_target, corrected_final)
merged_original = core_functions.merge_images(crop_target, moving)

# print(corrected_final.dtype)
# print(crop_target.dtype)

# plt.subplot(1, 2, 1)
# plt.imshow(merged_original)
# plt.subplot(1, 2, 2)
# plt.imshow(merged)
# plt.show()

# Generate merged image
alpha = 0.5

composite = np.zeros(corrected_final.shape, dtype=corrected_final.dtype)
for iC in range(3):
    composite[:, :, iC] = alpha * corrected_final[:, :, iC] + (1 - alpha) * crop_target[:, :, iC]

# plt.imshow(composite)
# plt.show()

# Save output images
tifffile.imwrite("../processed/merged_crop.tiff", composite, compression="lzw")