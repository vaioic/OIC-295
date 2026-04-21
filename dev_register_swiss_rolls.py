import core_functions
import skimage
from matplotlib import pyplot as plt
import scipy
import numpy as np
import cv2
import tifffile

# target = skimage.io.imread("../data/2026-04-21 Cropped for testing/259591_Ki67.ome.tif")
# moving = skimage.io.imread("../data/2026-04-21 Cropped for testing/259590_H3K9me3.ome.tif")

# # Downsample the images to speed up testing
# target = skimage.transform.rescale(target, 0.25, channel_axis=2)
# moving = skimage.transform.rescale(moving, 0.25, channel_axis=2)

# skimage.io.imsave("target_ds.tiff", target)
# skimage.io.imsave("moving_ds.tiff", moving)

target = skimage.io.imread("target_ds.tiff")
moving = skimage.io.imread("moving_ds.tiff")

# plt.imshow(target)
# plt.show()

target_hed = skimage.color.rgb2hed(target)
moving_hed = skimage.color.rgb2hed(moving)

patch = moving_hed[1370:1870, 1370:1870, 0]

# plt.imshow(patch)
# plt.show()

# exit()

image = target_hed[:, :, 0]

result = skimage.feature.match_template(image, patch)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

ax1.imshow(patch, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hcoin, wcoin = patch.shape
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()

print(x, y)


# # Create an RGB image for each of the stains
# null = np.zeros_like(ihc_hed[:, :, 0])
# ihc_h = skimage.color.hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
# ihc_e = skimage.color.hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
# ihc_d = skimage.color.hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

# # Display
# fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(target)
# ax[0].set_title("Original image")

# ax[1].imshow(ihc_h)
# ax[1].set_title("Hematoxylin")

# ax[2].imshow(ihc_e)
# ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

# ax[3].imshow(ihc_d)
# ax[3].set_title("DAB")

# for a in ax.ravel():
#     a.axis('off')

# fig.tight_layout()

# plt.show()

# Make a mask of the blue stain
# fig, ax = skimage.filters.try_all_threshold(ihc_hed[:, :, 0], figsize=(10,8), verbose=True)
# plt.show()

# threshold = skimage.filters.threshold_minimum(target_hed[:, :, 0])
# mask_target = target_hed[:, :, 0] > threshold

# threshold2 = skimage.filters.threshold_minimum(moving_hed[:, :, 0])
# mask_moving = moving_hed[:, :, 0] > threshold

# # Try to register the masks

# mask_target, mask_moving = core_functions.match_image_size(mask_target, mask_moving)

# shift, _, _ = skimage.registration.phase_cross_correlation(mask_target, mask_moving)

# plt.imshow(mask_moving)
# plt.show()



