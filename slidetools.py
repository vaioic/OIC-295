import openslide
import numpy as np
import skimage
from matplotlib import pyplot as plt
from pybioimageutils import visualize
from tqdm import tqdm
import cv2

def register_region(slide, reference_slide, topleft=(0,0), region_size=(100,100), level=2):

    moving = slide.read_region(topleft, level, region_size)
    moving = np.array(moving.convert('L'))

    reference = reference_slide.read_region(topleft, level, region_size)
    reference = np.array(reference.convert('L'))

    shift, _, _= skimage.registration.phase_cross_correlation(reference, moving)
    
    return shift


def calculate_shift(slide, reference_slide, downsample_index=2):
    #read_region size: (width, height)

    image = slide.read_region((0,0), downsample_index, reference_slide.level_dimensions[downsample_index])
    image = np.array(image.convert('L'))

    # image_A = skimage.exposure.equalize_adapthist(image_A)
    # # image_A = (image_A * 255).astype(np.uint8)
    # plt.imshow(image)
    # plt.show()

    reference_image = reference_slide.read_region((0,0), downsample_index, reference_slide.level_dimensions[downsample_index])
    reference_image = np.array(reference_image.convert('L'))
    
    shift, _, _= skimage.registration.phase_cross_correlation(reference_image, image)

    return shift, downsample_index

def correct_image_tiled(slide, reference_slide, shift, shift_level=2, output_level=0, tile_size=128):

    # Calculate number of tiles
    width, height = reference_slide.level_dimensions[output_level]    
    downsample_factor = reference_slide.level_downsamples[output_level]
    
    num_cols = int(np.ceil(width / tile_size))
    num_rows = int(np.ceil(height / tile_size))

    shift_factor = reference_slide.level_downsamples[shift_level]/reference_slide.level_downsamples[0]

    print(f"Shift: {shift}")
    print(f"Shift factor: {shift_factor}")
    
    shift_corrected = (int(np.round((shift[0] + 1) * shift_factor)),
                       int(np.round((shift[1] + 1) * shift_factor)))
    
    print(f"Shift corrected: {shift_corrected}")
    
    # # Get the lowest level image to equalize the histogram later
    # image_lvl2 = np.array(reference_slide.read_region((0,0), 2, reference_slide.level_dimensions[2]).convert('RGB'))

    for row in tqdm(range(num_rows)):
        for col in range(num_cols):
            
            x0 = int(col * tile_size * downsample_factor)
            y0 = int(row * tile_size * downsample_factor)

            tile_width = min(tile_size, width - col * tile_size)
            tile_height = min(tile_size, height - row * tile_size)
                
            #Read the reference slide
            ref_image = reference_slide.read_region(
                    (x0, y0),
                    output_level,
                    size=(tile_width, tile_height)
            )
            ref_image = np.array(ref_image.convert('RGB'))
            #ref_image = adjust_images(ref_image)

            # Can't normalize a tile... the histograms are all wrong
            # ref_image = skimage.exposure.match_histograms(ref_image, image_lvl2)
            # ref_image = (ref_image * 255).astype(np.uint8)
            
            # Read the corrected image
            image = slide.read_region(
                (x0 - shift_corrected[1], y0 - shift_corrected[0]),
                output_level,
                size=(tile_width, tile_height)
            )
            image = np.array(image.convert('RGB'))
            #image = adjust_images(image)
            # print(image.height)
            # print(image.width)

            output = visualize.composite(ref_image, image, normalize_images=False)

            # print(np.max(output))
            # print(np.min(output))

            #output = visualize.normalize(image)

            # output = (output * 255).astype(np.uint8)

            yield output.astype(np.uint8)

def read_tiles(slide, level=0, tile_size=128, start=(0, 0)):

    # Calculate number of tiles
    width, height = slide.level_dimensions[level]    
    downsample_factor = slide.level_downsamples[level]
    
    num_cols = int(np.ceil(width / tile_size))
    num_rows = int(np.ceil(height / tile_size))

    for row in tqdm(range(num_rows)):
        for col in range(num_cols):

            x0 = int(col * tile_size * downsample_factor)
            y0 = int(row * tile_size * downsample_factor)

            tile_width = min(tile_size, width - col * tile_size)
            tile_height = min(tile_size, height - row * tile_size)
                
            #Get current slice
            image = slide.read_region(
                (x0, y0),
                level,
                size=(tile_width, tile_height)
            )
            # print(image.height)
            # print(image.width)
          
            yield np.array(image.convert('RGB'))

def adjust_images(image, gain=1.0):

    print(image.dtype)
    image = image.astype(np.float32)
    print(image.dtype)
    image = ((image - 160)/(255 - 160)) * gain
    image[image < 0] = 0
    image[image > 1] = 1.0
    image = (image * 255).astype(np.uint8)

    return image
