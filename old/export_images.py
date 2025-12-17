
import openslide
import numpy as np
from tifffile import imwrite
import slidetools

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

shift = (-91, 584 + 7)
ds_index = 0


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
