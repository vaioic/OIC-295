import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
# Attempting to remake the script to see if I can improve the registration result
from oic_toolkit import display, register, util
from scipy import ndimage as ndi

#ROI = [  #[top left height width]

# target = sk.io.imread("../data/2026-06-08 Downsampled/30239_Ki67_267466.tif")
# target = sk.color.rgb2gray(target)

# moving = sk.io.imread("../data/2026-06-08 Downsampled/30239_H3K9me3_267412.tif")
# moving = sk.color.rgb2gray(moving)

target = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_Ki67_267466.ome.tif")
target = sk.color.rgb2gray(target)
target = target[::2, ::2]

moving = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_H3K9me3_267412.ome.tif")
moving = sk.color.rgb2gray(moving)
moving = moving[::2, ::2]


# Try rotation and scale correction
# _, _, corrected = register.rotation_scale_phasecorr(target, moving)

rotation, scale, corrected_logPolar = register.log_polar_phasecorr(target, moving)

# Try optical flow directly
# u, v, corrected_flow = register.optical_flow_tvl1(target, moving)

# Try pre-registration before optical flow
u, v, corrected_pre_flow = register.optical_flow_tvl1(target, corrected_logPolar)

# fig = plt.figure()

# # ax1 = fig.add_subplot(2, 2, 1)
# # merge1 = display.merge_images(target, moving)
# # ax1.imshow(merge1)
# # ax1.set_title("Original (merged)")

# # ax2 = fig.add_subplot(2, 2, 2)
# # merge2 = display.merge_images(target, corrected)
# # ax2.imshow(merge2)
# # ax2.set_title("Rotation and scale only (merged)")

# # ax3 = fig.add_subplot(2, 2, 3)
# # merge3 = display.merge_images(target, corrected_logPolar)
# # ax3.imshow(merge3)
# # ax3.set_title("Log-polar (merged)")

# ax3 = fig.add_subplot(2, 2, 3)
# merge3 = display.merge_images(target, corrected_flow)
# ax3.imshow(merge3)
# ax3.set_title("Optical flow")

# ax4 = fig.add_subplot(2, 2, 4)
merge4 = display.merge_images(target, corrected_pre_flow)
# ax4.imshow(merge4)
# ax4.set_title("Log-polar then optical flow (merged)")

# plt.show()

plt.imshow(merge4)
plt.show()

# exit()
# Now try upscaling to the full image

# Upscale to full size

# target_full = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_Ki67_267466.ome.tif")
# target_full = sk.color.rgb2gray(target_full)
# target_full = target_full[::2, ::2]

# moving_full = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_H3K9me3_267412.ome.tif")
# moving_full = sk.color.rgb2gray(moving_full)
# moving_full = moving_full[::2, ::2]

# target_full, moving_full = register.pad_images(target_full, moving_full)

# moving_full_corrected = sk.transform.rotate(moving_full, -rotation)
# moving_full_corrected = sk.transform.rescale(moving_full_corrected, 1/scale)



# u, v, corrected = register.optical_flow_tvl1(target_full, moving_full_corrected)

# --- Step 1: Define your upscaling factor ---
# If your low-res image was created via rescale(image, 0.25), your factor is 4.0
# UPSCALE_FACTOR = target_full.shape[0] // target.shape[0]
# print(UPSCALE_FACTOR)
# UPSCALE_FACTOR = 4.0

# # --- Step 2: Get your high-resolution target dimensions ---
# h_high, w_high = moving_full_corrected.shape[:2]

# # --- Step 3: Resize and scale the flow fields ---
# # Resize the grid layout to match the full-res dimensions (using bilinear order=1)
# v_high = sk.transform.resize(v, (h_high, w_high), order=1, preserve_range=True)
# u_high = sk.transform.resize(u, (h_high, w_high), order=1, preserve_range=True)

# # CRITICAL: Scale the actual displacement values by the upscale factor
# v_high = v_high * UPSCALE_FACTOR
# u_high = u_high * UPSCALE_FACTOR

# # --- Step 4: Generate pixel coordinate grid for high-res warping ---
# row_coords, col_coords = np.meshgrid(
#     np.arange(h_high), 
#     np.arange(w_high), 
#     indexing='ij'
# )

# # --- Step 5: Correct the full-resolution image distortion ---
# # In optical flow, warp coordinates are: original_grid + displacement_vectors
# high_res_coords = np.array([row_coords + v_high, col_coords + u_high])

# corrected_full_res_image = sk.transform.warp(
#     moving_full_corrected, 
#     high_res_coords, 
#     order=1,               # Bilinear interpolation
#     mode='edge'            # Handles out-of-bound edge pixels nicely
# )

# merge_full = display.merge_images(target_full, corrected_full_res_image)


# plt.imshow(merge_full)
# plt.show()
