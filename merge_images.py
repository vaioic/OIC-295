import os
import skimage
import openslide
import numpy as np
from matplotlib import pyplot as plt
from pybioimageutils import visualize
from tifffile import imwrite

# image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

# image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261081.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259267.svs"

# ---Begin processing---

# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

# Calculate image registration based on the second downsample level
ii = 2 #Image downsample index

img_size = slide_A.level_dimensions[ii]  # Note: Assumes slide A is always larger

image_A = slide_A.read_region((0,0), ii, slide_A.level_dimensions[ii])
image_B = slide_B.read_region((0,0), ii, img_size)

# Convert images from RGBA to RGB, then into numpy arrays
image_A = np.array(image_A.convert('RGB'))
image_B = np.array(image_B.convert('RGB'))

# Adjust the contrast of Image A, otherwise it is hard to see
image_A = skimage.exposure.equalize_adapthist(image_A)
image_A = (image_A * 255).astype(np.uint8)

# # Debugging statements
# print(f"Image A dtype: {image_A.dtype}")
# print(f"Image A shape: {image_A.shape}")
# print(f"Image A Max: {np.max(image_A)}")
# print(f"Image A Min: {np.min(image_A)}")

# print(f"Image B dtype: {image_B.dtype}")
# print(f"Image B shape: {image_B.shape}")

# skimage.io.imsave('ImageA.tif', image_A)
# skimage.io.imsave('ImageB.tif', image_B)

# Convert images to grayscale prior to registration
image_A_gray = skimage.color.rgb2gray(image_A)
image_B_gray = skimage.color.rgb2gray(image_B)

shift, _, _= skimage.registration.phase_cross_correlation(image_A_gray, image_B_gray)

print(f"Pixel shift: {shift[0], shift[1]}")

output_ds = 1

# Calculate the transformation matrix
scale_factor = slide_A.level_downsamples[ii] / slide_A.level_downsamples[output_ds]
shift_factored = shift * scale_factor

tform = skimage.transform.SimilarityTransform(translation=(-shift_factored[1], -shift_factored[0]))

image_A_full = slide_A.read_region((0,0), output_ds, slide_A.level_dimensions[output_ds])
image_A_full = np.array(image_A_full.convert('RGB'))

image_B_full = slide_B.read_region((0,0), output_ds, slide_A.level_dimensions[output_ds])
image_B_full = np.array(image_B_full.convert('RGB'))

image_B_corrected = np.zeros(image_B_full.shape)

for c in range(3):
    image_B_corrected[:, :, c] = skimage.transform.warp(image_B_full[:, :, c], tform)

image_B_corrected = (image_B_corrected * 255).astype(np.uint8)

merged_images = visualize.composite(image_A_full, image_B_corrected, normalize_images=True)
merged_images = (merged_images * 255).astype(np.uint8)

imwrite('test.tif', merged_images)

slide_A.close()
slide_B.close()