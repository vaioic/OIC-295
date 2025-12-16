import skimage
import openslide
import numpy as np
from pybioimageutils import visualize
from tifffile import imwrite

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

# image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261081.svs"

# image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259267.svs"

# ---Begin processing---

# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

image_A = slide_A.


tform = skimage.transform.SimilarityTransform(translation=(-shift_factored[1], -shift_factored[0]))

image_A_full = slide_A.read_region((0,0), output_ds, slide_A.level_dimensions[output_ds])
image_A_full = np.array(image_A_full.convert('RGB'))

image_B_full = slide_B.read_region((0,0), output_ds, slide_A.level_dimensions[output_ds])
image_B_full = np.array(image_B_full.convert('RGB'))

image_B_corrected = np.zeros(image_B_full.shape)

tform = skimage.transform.SimilarityTransform(translation=(-shift_factored[1], -shift_factored[0]))

for c in range(3):
    image_B_corrected[:, :, c] = skimage.transform.warp(image_B_full[:, :, c], tform)

image_B_corrected = (image_B_corrected * 255).astype(np.uint8)

merged_images = visualize.composite(image_A_full, image_B_corrected, normalize_images=True)
merged_images = (merged_images * 255).astype(np.uint8)

imwrite('test.tif', merged_images)

slide_A.close()
slide_B.close()