import os
import skimage
import openslide
import numpy as np
from tifffile import imwrite
from matplotlib import pyplot as plt
from pybioimageutils import visualize
import slidetools

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

shift, ds_index = slidetools.calculate_shift(slide_B, slide_A, downsample_index=2)

output_size = slide_A.level_dimensions[0]

imwrite('merged_259270_261082.tif',
        slidetools.correct_image_tiled(slide_B, slide_A, shift, shift_level=ds_index, output_level=0),
        tile=(128, 128),
        shape=(output_size[1], output_size[0], 3),
        dtype=np.uint8,
        photometric='rgb',
        )

output_size = slide_A.level_dimensions[1]

imwrite('merged_259270_261082_level1.tif',
        slidetools.correct_image_tiled(slide_B, slide_A, shift, shift_level=ds_index, output_level=1),
        tile=(128, 128),
        shape=(output_size[1], output_size[0], 3),
        dtype=np.uint8,
        photometric='rgb',
        )



# ds_index = 1
# output_size = slide_A.level_dimensions[ds_index]

# imwrite('tester.tif', 
#         slidetools.read_tiles(slide_A, level=ds_index),
#         tile=(128, 128),
#         shape=(output_size[1], output_size[0], 3),
#         dtype=np.uint8,
#         photometric='rgb',
#         )

# tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))

# warped_rgb = np.zeros(image_A.shape, np.float64)

# for c in range(3):
#     warped_rgb[:, :, c] = skimage.transform.warp(image_B[:, :, c], tform)

# warped_rgb = (warped_rgb * 255).astype(np.uint8)