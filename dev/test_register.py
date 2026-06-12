import skimage as sk
from matplotlib import pyplot as plt
# Attempting to remake the script to see if I can improve the registration result
from oic_toolkit import display, register

#ROI = [  #[top left height width]

target = sk.io.imread("../data/2026-06-08 Downsampled/30239_Ki67_267466.tif")
moving = sk.io.imread("../data/2026-06-08 Downsampled/30239_H3K9me3_267412.tif")

# target = sk.color.rgb2gray(target)
# target = sk.util.invert(target)

# moving = sk.color.rgb2gray(moving)
# moving = sk.util.invert(moving)

target = sk.color.rgb2hed(target)
target = target[..., 0]

moving = sk.color.rgb2hed(moving)
moving = moving[..., 0]

target_filt = sk.filters.difference_of_gaussians(target, 5, 20)
target_filt = sk.filters.sobel(target_filt)
plt.imshow(target_filt)
plt.show()

# results, corrected = register.register_phasexcorr(target, moving)
# merged = display.merge_images(target, corrected)

# plt.imshow(merged)
# plt.show()

# merge_original = display.merge_images(target, moving)

# plt.imshow(merge_original)
# plt.show()

src, dst = register.calculate_displacement_field(target, moving, num_grid=(40,40), template_size=200, search_window=150, show_plots=True)

_, src_full, dst_full = register.estimate_tform(src, dst, target.shape, mesh_grid=(80, 80))

corrected = register.fast_warp(moving, target.shape, src_full, dst_full, mesh_grid=(80, 80))

merge_final = display.merge_images(target, corrected)

plt.imshow(merge_final)
plt.show()
