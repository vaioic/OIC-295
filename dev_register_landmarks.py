# Perform coarse registration

import openslide
import slides
from tifffile import imwrite
import numpy as np
from pybioimageutils import visualize
import skimage
from matplotlib import pyplot as plt

# image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

# image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

shift = slides.calculate_shift(slide_B, slide_A, ds_level=2)

ref, moving = slides.get_registered_regions(slide_B, slide_A, shift, (5000, 5000), topleft=(10000,5000), ds_level=0)

# Get regions
ref_gray = skimage.color.rgb2gray(ref)
moving_gray = skimage.color.rgb2gray(moving)




# imwrite('./export/ref_registered_gray.tif', ref_gray)
# imwrite('./export/moving_registered_gray.tif', moving_gray)

# output = visualize.composite(ref, moving, normalize_images=False)
# # output = (output * 255).astype(np.uint8)
# output = output.astype(np.uint8)
# imwrite('./export/merged_registered.tif', output)
