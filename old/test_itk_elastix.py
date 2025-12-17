import openslide
import numpy as np
from tifffile import imwrite
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import itk

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

# slide_A = openslide.OpenSlide(image_A_path)
# slide_B = openslide.OpenSlide(image_B_path)

# moving = slide_B.read_region((10000, 10000), 0, (5000, 3000))
# moving = np.array(moving.convert('RGB'))

# reference = slide_A.read_region((10000, 10000), 0, (5000, 3000))
# reference = np.array(reference.convert('RGB'))

moving = itk.imread('moving.tif')
reference = itk.imread('reference.tif')

registered, params = itk.elastix_registration_method(reference, moving)

