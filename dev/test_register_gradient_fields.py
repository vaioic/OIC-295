import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from oic_toolkit import display, register
from scipy import ndimage

# Trying to see if using normalized gradients will help with the registration
target = sk.io.imread("../data/2026-06-08 Downsampled/30239_Ki67_267466.tif")
target = sk.color.rgb2hed(target)

moving = sk.io.imread("../data/2026-06-08 Downsampled/30239_H3K9me3_267412.tif")
moving = sk.color.rgb2hed(moving)

# # --- Plot originals ---#
# fig = plt.figure()

# ax1 = plt.subplot(1, 3, 1)
# ax1.imshow(target[..., 0])

# ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
# ax2.imshow(moving[..., 0])

# ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
# merged = display.merge_images(target[..., 0], moving[..., 0])
# ax3.imshow(merged)

# plt.show()

# Try both rotation and translation alignment
reference_image = target[..., 0]
reference_image = sk.util.img_as_float32(reference_image)

moving_image = moving[..., 0]
moving_image = sk.util.img_as_float32(moving_image)

h0, w0 = reference_image.shape
hm, wm = moving_image.shape

# Match the moving to the target shape
moving_final = np.zeros_like(reference_image)

# Determine the overlapping bounds
slice_y = min(h0, hm)
slice_x = min(w0, wm)

moving_final[:slice_y, :slice_x] = moving_image[:slice_y, :slice_x]

moving_image = moving_final


reference_image_filt = sk.filters.sobel(reference_image)
moving_image_filt = sk.filters.sobel(moving_image)


results, _ = register.register_phasexcorr(reference_image_filt, moving_image_filt)

corrected = ndimage.shift(
    moving_image,
    shift=results["shift"]
)

plt.imshow(reference_image_filt)
plt.show()



# fig = plt.figure()

# ax1 = plt.subplot(1, 3, 1)
# ax1.imshow(reference_image)

# ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
# ax2.imshow(sk.exposure.rescale_intensity(corrected, out_range=(0.0, 1.0)))

# ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
# merged = display.merge_images(reference_image, corrected)
# ax3.imshow(merged)

# plt.show()
exit()



rotation, scale = register.log_polar_phasecorr(reference_image, moving_image)

print(f"Rotation: {rotation}")
print(f"Scale: {scale}")

# Correct the rotation
unrot_tform = sk.transform.EuclideanTransform(rotation=(rotation/180)*np.pi)
unrotated_image = sk.transform.warp(moving_image, unrot_tform.inverse)

# --- Plot rotation correction ---#
fig = plt.figure()

ax1 = plt.subplot(1, 3, 1)
ax1.imshow(reference_image)

ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(sk.exposure.rescale_intensity(unrotated_image, out_range=(0.0, 1.0)))

ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
merged = display.merge_images(reference_image, unrotated_image)
ax3.imshow(merged)

plt.show()

# Correct translation
_, corrected = register.register_phasexcorr(target, unrotated_image)

fig = plt.figure()

ax1 = plt.subplot(1, 3, 1)
ax1.imshow(reference_image)

ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(sk.exposure.rescale_intensity(corrected, out_range=(0.0, 1.0)))

ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
merged = display.merge_images(reference_image, corrected)
ax3.imshow(merged)

plt.show()


# #Radius
# radius = np.max([h0, w0, hm, wm])//2

# image_polar = sk.transform.warp_polar(reference_image, radius=radius)
# moving_polar = sk.transform.warp_polar(moving_image, radius=radius)

# # Calculate the shift in the polar domain
# shifts_polar, error, phasediff = sk.registration.phase_cross_correlation(image_polar, moving_polar, upsample_factor=10)

# # Extract rotation angle from the vertical shift
# # Assuming standard radius mapping in warp_polar
# radius = image_polar.shape[0] / np.log(max(image_polar.shape[0], image_polar.shape[1]))
# angle_diff = shifts_polar[0] / radius * (2 * np.pi) 

# # 3. Correct the rotation using Euclidean Transform
# unrot_tform = sk.transform.EuclideanTransform(rotation=angle_diff)
# unrotated_image = sk.transform.warp(moving_image, unrot_tform.inverse)

# print(unrotated_image.dtype)
# print(np.max(unrotated_image), np.min(unrotated_image))

# # --- Plot rotation correction ---#
# fig = plt.figure()

# ax1 = plt.subplot(1, 3, 1)
# ax1.imshow(reference_image)

# ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
# ax2.imshow(sk.exposure.rescale_intensity(unrotated_image, out_range=(0.0, 1.0)))

# ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
# merged = display.merge_images(reference_image, unrotated_image)
# ax3.imshow(merged)

# plt.show()



# # 4. Find the remaining (X, Y) translation shift
# shifts_xy, error, phasediff = sk.registration.phase_cross_correlation(reference_image, unrotated_image, upsample_factor=10)

# # 5. Build and apply the final rigid alignment transformation
# final_tform = sk.transform.EuclideanTransform(rotation=angle_diff, translation=(shifts_xy[1], shifts_xy[0]))
# aligned_image = sk.transform.warp(moving_image, final_tform.inverse)

# print(f"Rotation:{angle_diff}")
# print(f"XY shift:{shifts_xy}")

# # --- Plot originals ---#
# fig = plt.figure()

# ax1 = plt.subplot(1, 3, 1)
# ax1.imshow(reference_image)

# ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
# ax2.imshow(aligned_image)

# ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
# merged = display.merge_images(reference_image, aligned_image)
# ax3.imshow(merged)

# plt.show()
