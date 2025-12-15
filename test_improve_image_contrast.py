import os
import skimage
import openslide
import numpy as np
from tifffile import imwrite
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import slidetools

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

slide_A = openslide.OpenSlide(image_A_path)

image_A = slide_A.read_region((0,0), 2, slide_A.level_dimensions[2])

image_A = np.array(image_A.convert('RGB'))


image_A = (image_A.astype(np.float32) * 10) - 100
#image_A = image_A.astype(np.float32)
#image_A = skimage.exposure.adjust_sigmoid(image_A, cutoff=0.1)


#image_A = (image_A - np.min(image_A)) / (np.max(image_A) - np.min(image_A))
image_A = (image_A * 255).astype(np.uint8)
skimage.io.imsave('ImageA_corrected.tif', image_A)


#plt.imshow(image_A)
#plt.show()