# Perform coarse registration

import openslide
import slides
from tifffile import imwrite
import numpy as np
from pybioimageutils import visualize
import skimage
from matplotlib import pyplot as plt
import cv2
import sys


# image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

# image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

# image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

# image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

# ---Begin processing---
# Create slide objects
slide_A = openslide.OpenSlide(image_A_path)
slide_B = openslide.OpenSlide(image_B_path)

#shift = slides.calculate_shift(slide_B, slide_A, ds_level=2)
shift = (-80, 592)

ref, moving = slides.get_registered_regions(slide_B, slide_A, shift, (5000, 5000), topleft=(10000,5000), ds_level=0)

# Get regions
ref_gray = skimage.color.rgb2gray(ref)
ref_gray = (ref_gray * 255).astype(np.uint8)

moving_gray = skimage.color.rgb2gray(moving)
moving_gray = (moving_gray * 255).astype(np.uint8)

output = visualize.composite(ref, moving, normalize_images=False)
# output = (output * 255).astype(np.uint8)
output = output.astype(np.uint8)
# imwrite('./export/merged_registered.tif', output)

fig, ax = plt.subplots()
ax.set_title('Click on points to select them')
# Set the 'picker' property to a tolerance value (e.g., 5 points)
ax.imshow(output, picker=5) 

selected_points = []

def onclick(event):
    """
    Event handler for mouse clicks.
    
    Draws a circle at the click location and updates the plot.
    """
    if event.xdata is not None and event.ydata is not None:
        # Get the coordinates
        x, y = event.xdata, event.ydata
        selected_points.append((x, y))
        
        print(f"Point selected: ({x:.2f}, {y:.2f})")
        
        # Draw a red circle at the clicked point
        circle = plt.Circle((x, y), radius=31, color='r', fill=False)
        ax.add_patch(circle)
        
        # Refresh the display to show the new circle
        plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

kpM, desM, kpR, desR = slides.compute_descriptors(moving_gray, ref_gray, selected_points, window_size=31)

print(desM.dtype)

# imwrite('./export/ref_registered_gray.tif', ref_gray)
# imwrite('./export/moving_registered_gray.tif', moving_gray)

# output = visualize.composite(ref, moving, normalize_images=False)
# # output = (output * 255).astype(np.uint8)
# output = output.astype(np.uint8)
# imwrite('./export/merged_registered.tif', output)
# 4. Initialize the Brute Force Matcher
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# # 5. Match the descriptors
# matches = bf.match(desM, desR)

# # 6. Sort matches by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(desR,desM,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 5

if len(good)>MIN_MATCH_COUNT:

    # 7. Draw the top N matches
    img_matches = cv2.drawMatches(ref_gray, kpR, moving_gray, kpM, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    src_pts = np.float32([kpM[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = ref_gray.shape[:2]
    img_registered = cv2.warpPerspective(moving_gray, M, (w, h))

    overlay = cv2.addWeighted(ref_gray, 0.5, img_registered, 0.5, 0)
    cv2.imshow("Registered Image", overlay)

else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None