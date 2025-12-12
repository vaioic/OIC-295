import os
import skimage
import openslide
import numpy as np
from matplotlib import pyplot as plt
from pybioimageutils import visualize

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

slide_A = openslide.OpenSlide(image_A_path)
print(slide_A.level_dimensions)
image_A = np.array(slide_A.read_region((0,0), 2, (3000,3000)))
slide_A.close()

slide_B = openslide.OpenSlide(image_B_path)
image_B = np.array(slide_B.read_region((0,0), 2, (3000,3000)))
slide_B.close()

shift, _, _= skimage.registration.phase_cross_correlation(image_A[:, :, :3], image_B[:, :, :3])
print(shift)

corrected_image_b = np.roll(image_B, shift)

ovImg = visualize.composite(image_A[:, :, 0], image_B[:, :, 0], color_A=(1, 0, 1), color_B=(0, 1, 0))

skimage.io.imsave('test.tiff', ovImg)

print(image_A.shape)
print(image_A.dtype)

plt.figure()
plt.imshow(ovImg)
plt.show()