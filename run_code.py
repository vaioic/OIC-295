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

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

shift, ds_index = slidetools.calculate_shift(slide_B, slide_A, downsample_index=1)

output_size = slide_A.level_dimensions[0]

szTile = 256
subresolutions = 2

# with TiffWriter('temp_2.tiff', bigtiff=True) as tif:
    
#     x_resolution = float(slide_A.properties[openslide.PROPERTY_NAME_MPP_X])
#     y_resolution = float(slide_A.properties[openslide.PROPERTY_NAME_MPP_Y])
    
#     metadata = {
#         'PhysicalSizeX': x_resolution,
#         'PhysicalSizeXUnit': 'µm',
#         'PhysicalSizeY': y_resolution,
#         'PhysicalSizeYUnit': 'µm',
#     }

#     options = dict(
#         photometric='rgb',
#         tile=(szTile,szTile),
#         compression='jpeg',
#         resolutionunit='CENTIMETER',
#         maxworkers=2,
#     )

#     tif.write(
#         slidetools.correct_image_tiled(slide_B, slide_A, shift, shift_level=ds_index, output_level=0, tile_size=szTile),
#         subifds=subresolutions,
#         resolution=(1e4 / x_resolution, 1e4 / y_resolution),
#         shape=(slide_A.dimensions + (3,)),
#         dtype=np.uint8,
#         **options,
#     )

#     #Write pyramid levels
#     for level in range(subresolutions):
#         tif.write(
#             slidetools.correct_image_tiled(slide_B, slide_A, shift, shift_level=ds_index, output_level=(level + 1), tile_size=szTile),
#             subfiletype=1,
#             resolution=(
#                 1e4 / (slide_A.level_downsamples[level + 1] * x_resolution), 1e4 / (slide_A.level_downsamples[level + 1] * y_resolution),
#                 ),
#             shape=(slide_A.level_dimensions[level + 1] + (3,)),
#             dtype=np.uint8,
#             **options,            
#         )


szTile = 256
output_size = slide_A.level_dimensions[0]

imwrite('merged_259270_261082_01.tif',
        slidetools.correct_image_tiled(slide_B, slide_A, shift, shift_level=ds_index, output_level=0, tile_size=szTile),
        bigtiff=True,
        tile=(szTile, szTile),
        shape=(output_size[1], output_size[0], 3),
        dtype=np.uint8,
        photometric='rgb',
        )

# output_size = slide_A.level_dimensions[1]

# imwrite('merged_259270_261082_level1.tif',
#         slidetools.correct_image_tiled(slide_B, slide_A, shift, shift_level=ds_index, output_level=1),
#         tile=(128, 128),
#         shape=(output_size[1], output_size[0], 3),
#         dtype=np.uint8,
#         photometric='rgb',
#         )



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