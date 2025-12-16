import os
import skimage
import openslide
import numpy as np
from tifffile import imwrite, TiffWriter
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import slidetools

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

slide = openslide.OpenSlide(image_A_path)

image = slide.read_region((0,0), 1, slide.level_dimensions[1])
image = np.array(image.convert('RGB'))

print(image.dtype)

image_adj = slidetools.adjust_images(image)

plt.imshow(image_adj)
plt.show()

