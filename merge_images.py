import os
import skimage
import openslide
import numpy as np
from matplotlib import pyplot as plt
from pybioimageutils import visualize

# image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

# image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261081.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259267.svs"

ii = 2 #Image downsample index

slide_A = openslide.OpenSlide(image_A_path)
img_size = slide_A.level_dimensions[ii]
image_A = slide_A.read_region((0,0), ii, slide_A.level_dimensions[ii])
slide_A.close()

slide_B = openslide.OpenSlide(image_B_path)
image_B = slide_B.read_region((0,0), ii, img_size)
slide_B.close()

image_A = np.array(image_A.convert('RGB'))
image_B = np.array(image_B.convert('RGB'))

# Try adjusting the contrast
image_A = skimage.exposure.equalize_adapthist(image_A)
image_A = (image_A * 255).astype(np.uint8)
# image_A[:, :, 0] = image_A[:, :, 0] * 1.2
# image_A[:, :, 2] = image_A[:, :, 2] * 1.2

print(f"Image A dtype: {image_A.dtype}")
print(f"Image A shape: {image_A.shape}")
print(f"Image A Max: {np.max(image_A)}")
print(f"Image A Min: {np.min(image_A)}")

print(f"Image B dtype: {image_B.dtype}")
print(f"Image B shape: {image_B.shape}")

skimage.io.imsave('ImageA.tif', image_A)
skimage.io.imsave('ImageB.tif', image_B)

image_A_gray = skimage.color.rgb2gray(image_A)
image_B_gray = skimage.color.rgb2gray(image_B)

shift, _, _= skimage.registration.phase_cross_correlation(image_A_gray, image_B_gray)

print(f"Pixel shift: {shift[0], shift[1]}")

# Calculate the transformation matrix
tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))

warped_rgb = np.zeros(image_A.shape, np.float64)

for c in range(3):
    warped_rgb[:, :, c] = skimage.transform.warp(image_B[:, :, c], tform)

warped_rgb = (warped_rgb * 255).astype(np.uint8)

print(f"Transform matrix: {tform}")

print(f"Warped image shape: {warped_rgb.shape}")
print(f"Warped image dtype: {warped_rgb.dtype}")
print(f"Warped image Max Value: {np.max(warped_rgb)}")
print(f"Warped image Min Value: {np.min(warped_rgb)}")

skimage.io.imsave('warped.tif', warped_rgb)

ovImg = visualize.composite(image_A, warped_rgb, normalize_images=True)
ovImg = (ovImg * 255).astype(np.uint8)

print(f"Overlay image shape: {ovImg.shape}")
print(f"Overlay image dtype: {ovImg.dtype}")
print(f"Overlay image Max Value: {np.max(ovImg)}")
print(f"Overlay image Min Value: {np.min(ovImg)}")

skimage.io.imsave('merged_images.tif', ovImg)
