import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, exposure, transform, util, color
from scipy.interpolate import griddata
import tifffile

target = io.imread("target_output.tif")
corrected = io.imread("corrected_output.tif")

# Split into HED channels
target_HED = color.rgb2hed(target)
target_HED = util.img_as_ubyte(target_HED)

corrected_HED = color.rgb2hed(corrected)
corrected_HED = util.img_as_ubyte(corrected_HED)

# Write all four channels -- the shape should be (channel, Y, X)

stacked = np.stack(
    [target_HED[..., 0],
    target_HED[..., 2],
    corrected_HED[..., 0],
    corrected_HED[..., 2]],
    axis=0)

print(stacked.shape)

channel_names = ['Ki67_H', 'Ki67_D', 'H3K9me3_H', 'H3K9me3_D']

# Save as OME-TIFF
tifffile.imwrite(
    "multichannel_image.ome.tif",
    stacked,
    metadata={'axes': 'CYX', 'Channel': {'Name': channel_names}},
    ome=True
)