import core_functions
import skimage
from matplotlib import pyplot as plt
import scipy
import numpy as np
import cv2
import tifffile

target = skimage.io.imread("../data/2026-04-21 Cropped for testing/259591_Ki67.ome.tif")
moving = skimage.io.imread("../data/2026-04-21 Cropped for testing/259590_H3K9me3.ome.tif")

# Downsample the images to speed up testing
target = skimage.transform.rescale(target, 0.25, channel_axis=2)
moving = skimage.transform.rescale(moving, 0.25, channel_axis=2)

plt.imshow(target)
plt.show()
plt.close()

# Match image sizes
target, moving = core_functions.match_image_size(target, moving)

# Perform a coarse alignment
shift = core_functions.get_shift(target, moving)

print(shift)

corrected = core_functions.translate_image(moving, shift)

overlay = core_functions.merge_images(target, corrected)

plt.imshow(overlay)
plt.show()

exit()



crop_moving, crop_target = core_functions.match_translated_images(corrected, target, shift)

merged = core_functions.merge_images(crop_target, crop_moving)
# plt.imshow(merged)
# plt.show()
# exit()


# # For development, crop the resulting image further for speed of testing
# crop_moving = crop_moving[1500:3000, 1500:3000, :]
# crop_target = crop_target[1500:3000, 1500:3000, :]

# merge = core_functions.merge_images(crop_moving, crop_target)
# plt.imshow(merge)
# plt.show()

# Perform fine-tuned alignments
dX, dY, dX_c, dY_c = core_functions.get_fine_shift(crop_moving, crop_target, 60, 400)

# Might want to apply some kind of smoothing filter

# Interpolate the displacements to the whole image
interp_dX = scipy.interpolate.RegularGridInterpolator((dX_c, dY_c), dX.T, bounds_error=False, fill_value=None)
interp_dY = scipy.interpolate.RegularGridInterpolator((dX_c, dY_c), dY.T, bounds_error=False, fill_value=None)

# Generate image coordinates
iX = np.arange(0, crop_moving.shape[1])
iY = np.arange(0, crop_moving.shape[0])

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

corrected_final = core_functions.remap_image(crop_moving, iXX, iYY, dX_upsampled, dY_upsampled)

corrected_final = skimage.util.img_as_float(corrected_final)
crop_target = skimage.util.img_as_float(crop_target)

merged = core_functions.merge_images(crop_target, corrected_final)
merged_original = core_functions.merge_images(crop_target, crop_moving)

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

# skimage.io.imsave("../processed/merged_full_60x60.png", skimage.util.img_as_ubyte(composite))


# composite_original = np.zeros(target.shape, dtype=target.dtype)
# for iC in range(3):
#     composite_original[:, :, iC] = alpha * moving[:, :, iC] + (1 - alpha) * target[:, :, iC]

# skimage.io.imsave("../processed/merged_crop_noreg.png", skimage.util.img_as_ubyte(composite_original))

# # Make sure these are floats
# crop_target = skimage.util.img_as_float32(crop_target)
# crop_moving = skimage.util.img_as_float32(crop_moving)

# composite_onlytranslate = np.zeros(crop_target.shape, dtype=crop_target.dtype)
# for iC in range(3):
#     composite_onlytranslate[:, :, iC] = alpha * crop_moving[:, :, iC] + (1 - alpha) * crop_target[:, :, iC]

# skimage.io.imsave("../processed/merged_crop_translateonly.png", skimage.util.img_as_ubyte(composite_onlytranslate))

# Not currently working
# # Get the target coordinates
# target_XX = iXX - (dX_upsampled)
# target_YY = iYY - (dY_upsampled)

# coords_to_sample = np.array([target_XX, target_YY])

# print(coords_to_sample.shape)

# final_corrected_image = np.zeros(crop_moving.shape, dtype=crop_moving.dtype)
# for iC in range(moving.shape[2]):
#      final_corrected_image[:, :, iC] = scipy.ndimage.map_coordinates(crop_moving[:, :, 0], coords_to_sample, 
#                                           order=3, mode='constant')

# merged_original = core_functions.merge_images(crop_target, crop_moving)
# merged_corrected = core_functions.merge_images(crop_target, final_corrected_image)

# dX_c_grid, dY_c_grid = np.meshgrid(dX_c, dY_c)

# plt.subplot(1, 2, 1)
# plt.imshow(merged_original)
# plt.subplot(1, 2, 2)
# plt.imshow(merged_corrected)
# #plt.quiver(dX_c, dY_c, dX, dY)
# plt.quiver(iXX[::50, ::50], iYY[::50, ::50], dX_upsampled[::50, ::50], dY_upsampled[::50, ::50])
# # plt.plot(target_XX[::10, ::10], target_YY[::10, ::10], 'rx')
# # plt.plot(iXX[::10, ::10], iYY[::10, ::10], 'bo')
# plt.show()