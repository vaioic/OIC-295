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

image = slide_A.read_region((0,0), 1, slide_A.level_dimensions[1])
image = np.array(image.convert('RGB'))
ref = slide_B.read_region((0,0), 1, slide_B.level_dimensions[1])

ref = np.array(ref.convert('RGB'))

imwrite('reference.tif', image, photometric='rgb')
imwrite('moving.tif', ref, photometric='rgb')

# imwrite('test_merge.tif',
#         slidetools.correct_image_tiled_2(slide_B, slide_A, output_level=2, tile_size=128),
#         tile=(128, 128),
#         dtype=np.uint8,
#         shape=(slide_A.level_dimensions[2] + (3,)),
#         photometric='rgb')


# merged = slidetools.register_region_2(slide_B, slide_A, topleft=(10000, 10000), level=1)

# imwrite('merged2.tif', merged, photometric='rgb')
# imwrite('moving.tif', moving, photometric='rgb')
# imwrite('corrected.tif', corrected, photometric='rgb')
# imwrite('refrence.tif', reference, photometric='rgb')

# plt.imshow(merged)
# plt.show()

# plt.imshow(corrected)
# plt.show()


# shift, ds_index = slidetools.register_region(slide_B, slide_A, topleft=()
