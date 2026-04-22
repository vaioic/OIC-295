import numpy as np
import matplotlib.pyplot as plt
import skimage

target = skimage.io.imread("target_ds.tiff")
moving = skimage.io.imread("moving_ds.tiff")

target = skimage.color.rgb2gray(target)
moving = skimage.color.rgb2gray(moving)

# 2. Extract the template from the MOVING image
crop_coords = (1500, 2500, 100, 100)
y, x, h, w = crop_coords
template = moving[y:y+h, x:x+w]
H, W = target.shape
search_radius=150

y_start = max(0, y - search_radius)
y_end = min(H, y + h + search_radius)
x_start = max(0, x - search_radius)
x_end = min(W, x + w + search_radius)

target_window = target[y_start:y_end, x_start:x_end]

# 3. Perform Normalized Cross-Correlation
# This returns a map of correlation coefficients (-1.0 to 1.0)
result = skimage.feature.match_template(target_window, template)

# 4. Find the peak (the best match location)
ij = np.unravel_index(np.argmax(result), result.shape)
y_match = ij[0] + y_start
x_match = ij[1] + x_start

# 5. Calculate the shift
# Shift = (Target Location) - (Original Location in Moving)
y_shift = y_match - y
x_shift = x_match - x

# 6. Extract the matching section from the TARGET image
matched_section = target[y_match:y_match+h, x_match:x_match+w]

# 7. Create the Overlay (Green = Moving Template, Magenta = Target Match)
# We normalize both for better visualization
temp_norm = skimage.exposure.rescale_intensity(template)
match_norm = skimage.exposure.rescale_intensity(matched_section)

overlay = np.dstack((match_norm, temp_norm, match_norm))

# 8. Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(moving, cmap='gray')
rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', lw=2)
ax[0].add_patch(rect)
ax[0].set_title("Moving Image (Red = Template)")

ax[1].imshow(target, cmap='gray')
# Draw Search Window
ax[1].add_patch(plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                                edgecolor='yellow', facecolor='none', lw=1, linestyle='--'))
# Draw Found Match
ax[1].add_patch(plt.Rectangle((x_match, y_match), w, h, edgecolor='cyan', facecolor='none', lw=2))
ax[1].set_title(f"Target Image\nSearch Radius: {search_radius}px")

ax[2].imshow(overlay)
ax[2].set_title("Landmark Overlay (Target vs Moving)")

plt.show()