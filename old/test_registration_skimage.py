import os
import skimage
import openslide
import numpy as np
from tifffile import imwrite, TiffWriter
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import slidetools
import cv2

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

moving = slide_B.read_region((10000, 10000), 0, (5000, 3000))

moving_gray = np.array(moving.convert('L'))
moving = np.array(moving.convert('RGB'))

reference = slide_A.read_region((10000, 10000), 0, (5000, 3000))

reference_gray = np.array(reference.convert('L'))
reference = np.array(reference.convert('RGB'))

print(reference_gray.shape)

tform = skimage.transform.estimate_transform('affine', reference_gray, moving_gray)

corrected = np.zeros(moving.shape)

for c in range(3):
    corrected[:, :, c] = skimage.transform.warp(moving[:, :, c], tform)

corrected = (corrected * 255).astype(np.uint8)

merged = visualize.composite(reference, corrected, 
normalize_images=False)

imwrite('merged2.tif', merged, photometric='rgb')
# imwrite('moving.tif', moving, photometric='rgb')
# imwrite('corrected.tif', corrected, photometric='rgb')
# imwrite('refrence.tif', reference, photometric='rgb')

# plt.imshow(merged)
# plt.show()

# plt.imshow(corrected)
# plt.show()


# shift, ds_index = slidetools.register_region(slide_B, slide_A, topleft=()
