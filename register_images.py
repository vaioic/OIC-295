import os
import skimage
import openslide
import numpy as np
from tifffile import imwrite
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import slides


image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

shift = slides.calculate_shift(slide_B, slide_A, ds_level=2)

output_size = slide_A.level_dimensions[0]

sz_tile = 128

imwrite('./export/merged_259270_261082_level0.tif',
        slides.register_tiled_image(slide_B, slide_A, shift, ds_level=0, tile_size=sz_tile),
        tile=(sz_tile, sz_tile),
        shape=(output_size[1], output_size[0], 3),
        dtype=np.uint8,
        photometric='rgb',
        )
