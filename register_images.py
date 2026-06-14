from pathlib import Path

import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from oic_toolkit import display, register, util
from scipy import ndimage as ndi

from core import core_funcs

output_path = Path("../processed/2026-06-14")

output_path.mkdir(parents=True, exist_ok=True)

fn = "30239_Ki67_267466.tif"

target = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_Ki67_267466.ome.tif")
# target = sk.io.imread("../data/2026-06-08 Downsampled/259591_Ki67.tif")
target = target[::2, ::2, ...]

moving = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_H3K9me3_267412.ome.tif")
# moving = sk.io.imread("../data/2026-06-08 Downsampled/259590_H3K9me3.tif")
moving = moving[::2, ::2, ...]

moving = register.match_size(target, moving)

# print(f"Target, Moving: {target.shape},{moving.shape}")

target_gray = sk.color.rgb2gray(target)
moving_gray = sk.color.rgb2gray(moving)

rotation, scale, corrected_logPolar = register.log_polar_phasecorr(target_gray, moving_gray)

u, v = register.optical_flow_tvl1(target_gray, corrected_logPolar)

# Correct the moving image
moving_rotscalecorrected = sk.transform.rotate(moving, -np.deg2rad(rotation))
moving_rotscalecorrected = sk.transform.rescale(moving_rotscalecorrected, scale)

moving_rotscalecorrected = register.match_size(target, moving_rotscalecorrected)

corrected = register.correct_optical_flow(moving_rotscalecorrected, u, v)

print(f"Corrected dtype: {corrected.dtype}")
print(np.max(corrected), np.min(corrected))

plt.imshow(corrected)
plt.show()

corrected = sk.util.img_as_ubyte(corrected)
# corrected = sk.util.img_as_ubyte(corrected / 255.0)

core_funcs.export_stacked_tiff(target, corrected, (output_path/fn))

merge = display.merge_images(target, corrected)
sk.io.imsave(output_path / (fn + "_merged.tif"), merge)

# plt.imshow(merge)
# plt.show()
