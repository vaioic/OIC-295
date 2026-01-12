import core_functions
import skimage
from matplotlib import pyplot as plt
from numpy import pad, zeros, max, uint8, floor, ceil

target = skimage.data.eagle()
target = skimage.util.img_as_float(target)

template = target[455:610, 770:1000]

shift = core_functions.fast_template_match(template, target)

# There are two ways to correct the image
# THe returned value is the top left corner of the image (and is also equivalent to the pixel shift)

# Pad the template to match the target size
template_H, template_W = template.shape[:2]
target_H, target_W = target.shape[:2]   

# Calculate the number of pixels to pad
template_padded = pad(template, [(0, target_H - template_H), (0, target_W - template_W)], mode='constant', constant_values=0)

#shift_final = shift - (template_W, template_H)

# print(f"Shift: {shift}")

#tform = skimage.transform.SimilarityTransform(translation=(-(shift[0] - template_H/2), -(shift[1] - template_W/2)))
tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
template_corrected = skimage.transform.warp(template_padded, tform)

# template_corrected = zeros((target.shape[0], target.shape[1]))
# template_corrected[shift[0]:(shift[0] + template_H), shift[1]:(shift[1] + template_W)] = template
# template_corrected = skimage.util.img_as_ubyte(template_corrected)

# Generate image for checking
corrected = zeros((target.shape[0], target.shape[1], 3), dtype=float)
corrected[..., 0] = target
corrected[..., 1] = template_corrected
corrected[..., 2] = target

plt.imshow(corrected)
plt.show()