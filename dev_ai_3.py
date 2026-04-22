import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, exposure, transform
from scipy.interpolate import griddata

def generate_displacement_field(target_path, moving_path, grid_size=(30, 30), template_size=200, search_radius=100):
    target = io.imread(target_path, as_gray=True)
    moving = io.imread(moving_path, as_gray=True)
    H, W = target.shape

    margin = template_size // 2 + search_radius
    y_coords = np.linspace(margin, H - margin, grid_size[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, grid_size[1], dtype=int)

    src_points = []
    dst_points = []

    print(f"Scanning {grid_size[0]*grid_size[1]} grid points...")

    for y_c in y_coords:
        for x_c in x_coords:
            y0, x0 = y_c - template_size//2, x_c - template_size//2
            template = moving[y0:y0+template_size, x0:x0+template_size]

            if np.std(template) < 0.005: continue 

            wy0, wy1 = max(0, y0 - search_radius), min(H, y0 + template_size + search_radius)
            wx0, wx1 = max(0, x0 - search_radius), min(W, x0 + template_size + search_radius)
            
            result = feature.match_template(target[wy0:wy1, wx0:wx1], template)
            ij = np.unravel_index(np.argmax(result), result.shape)
            
            src_points.append([x_c, y_c])
            dst_points.append([ij[1] + wx0 + template_size//2, ij[0] + wy0 + template_size//2])

    return np.array(src_points), np.array(dst_points), (H, W), moving, target

def apply_interpolated_piecewise_warp(src, dst, img_shape, moving_img):
    H, W = img_shape
    
    # 1. Calculate the actual displacement vectors
    displacements = dst - src 
    
    # 2. Create a dense, uniform grid for the whole image
    # We create points every 100 pixels to ensure the mesh is stable everywhere
    grid_y, grid_x = np.mgrid[0:H:100, 0:W:100]
    full_src_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    print("Applying interpolation")

    # 3. INTERPOLATION: Fill the gaps using the matched landmarks
    # 'linear' fills gaps smoothly; 'nearest' handles the very edges
    # We use the median shift as a fallback for areas far from any tissue
    interp_dx = griddata(src, displacements[:, 0], full_src_grid, 
                         method='linear', fill_value=np.median(displacements[:, 0]))
    interp_dy = griddata(src, displacements[:, 1], full_src_grid, 
                         method='linear', fill_value=np.median(displacements[:, 1]))

    # 4. Reconstruct the full destination grid
    full_dst_grid = full_src_grid + np.vstack([interp_dx, interp_dy]).T

    # 5. Add static corner anchors to keep the frame square
    anchors = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])
    full_src_grid = np.vstack([full_src_grid, anchors])
    full_dst_grid = np.vstack([full_dst_grid, anchors])

    # 6. Apply Piecewise Affine Transform using the REPAIRED grid
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(full_dst_grid, full_src_grid)
    warped = transform.warp(moving_img, tform, output_shape=img_shape)
    
    return warped, full_src_grid, full_dst_grid, tform

# --- Workflow ---
# Update file names as needed
TARGET_FILE = 'target_ds.tiff'
MOVING_FILE = 'moving_ds.tiff'

src, dst, shape, moving, target = generate_displacement_field(
    TARGET_FILE, 
    MOVING_FILE, 
    grid_size=(25, 25), 
    template_size=300, 
    search_radius=200
)

def save_color_overlay(target_rgb_path, moving_rgb_path, tform, output_path):
    """Warps the color moving image and overlays it with the color target."""
    target_rgb = io.imread(target_rgb_path)
    moving_rgb = io.imread(moving_rgb_path)
    
    # Warp each color channel independently using the estimated transform
    print("Warping color channels...")
    warped_rgb = transform.warp(moving_rgb, tform, output_shape=target_rgb.shape)
    
    # 50/50 blend for the final color image
    blend = (target_rgb.astype(float)/255 * 0.5) + (warped_rgb * 0.5)
    blend = np.clip(blend, 0, 1)
    
    io.imsave(output_path, (blend * 255).astype(np.uint8))
    print(f"Color blend saved to {output_path}")


# Apply the warp with the interpolation fix
corrected_img, final_src, final_dst, tform = apply_interpolated_piecewise_warp(src, dst, shape, moving)

# --- Visualization ---
t_norm = exposure.rescale_intensity(target, out_range=(0, 1))
w_norm = exposure.rescale_intensity(corrected_img, out_range=(0, 1))
overlay = np.dstack((t_norm, w_norm, t_norm))

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(target, cmap='gray', alpha=0.5)
axes[0].quiver(final_src[:, 0], final_src[:, 1], 
               final_dst[:, 0] - final_src[:, 0], 
               final_dst[:, 1] - final_src[:, 1], 
               color='cyan', angles='xy', scale_units='xy', scale=1, width=0.001)
axes[0].set_title("Interpolated Displacement Field (No Gaps)")

axes[1].imshow(overlay)
axes[1].set_title("Final Registration Overlay")
plt.show()

io.imsave('overlay_v3_smaller.png', (overlay * 255).astype(np.uint8))

# 2. Create and save the Color overlay (using the original files)
save_color_overlay(TARGET_FILE, MOVING_FILE, tform, 'final_color_registration_v3_smaller.png') 