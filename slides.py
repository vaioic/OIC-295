'''
Contains functions used to work with and register slide images
'''

import openslide
from  matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

def get_region(slide, ds_level=2):

    # Initialize variables to hold the final selection
    topleft = (0,0)
    bottomright = (0,0)

    # Define callback function for rectangle selection
    def onselect_callback(eclick, erelease):
        nonlocal topleft, bottomright

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        topleft = (min(x1,x2), min(y1,y2))
        bottomright = (max(x1,x2), max(y1,y2))

    # Read in the image at the current downsample level
    img = slide.read_region((0,0), ds_level, slide.level_dimensions[ds_level])

    fig, ax = plt.subplots()
    ax.imshow(img)
    selector = RectangleSelector(
        ax, onselect_callback,
        useblit=True,
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)
    
    plt.show()

    # Return coordinates scaled by the downsample level
    downsample_factor = slide.level_downsamples[ds_level]
    topleft = np.floor(np.array(topleft) * downsample_factor)
    bottomright = np.ceil(np.array(bottomright) * downsample_factor)

    # Threshold the coordinates so they always return a valid region
    topleft[topleft < 0] = 0
    region_size = (bottomright - topleft) + 1  #[Width, Height]

    #region_size[0] = min(region_size[0], slide.dimensions[0])
    #region_size[1] = min(region_size[1], slide.dimensions[1])
    
    return tuple(topleft.astype(int)), tuple(region_size.astype(int))

