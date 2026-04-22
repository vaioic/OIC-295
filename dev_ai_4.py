import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, exposure, transform
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def generate_displacement_field(target_path, moving_path, grid_size=(30, 30), template_size=200, search_radius=100):
    # This remains the same as your current functioning logic
    target = io.imread(target_path, as_gray=True)
    moving = io.imread(moving_path, as_gray=True)
    H, W = target.shape

    margin = template_size // 2 + search_radius
    y_coords = np.linspace(margin, H - margin, grid_size[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, grid_size[1], dtype=int)

    src_points = []
    dst_points = []

    print(f"Extracting landmarks from {grid_size[0]*grid_size[1]} points...")

    for y_c in y_coords:
        for x_c in x_coords:
            y0, x0 = y_c - template_size//2, x_c - template_size//2
            template = moving[y0:y0+template_size, x0:x0+template_size]

            if np.std(template) < 0.005: 
                continue

            wy0, wy1 = max(0, y0 - search_radius), min(H, y0 + template_size + search_radius)
            wx0, wx1 = max(0, x0 - search_radius), min(W, x0 + template_size + search_radius)
            target_window = target[wy0:wy1, wx0:wx1]

            result = feature.match_template(target_window, template)
            ij = np.unravel_index(np.argmax(result), result.shape)
            
            y_match = ij[0] + wy0
            x_match = ij[1] + wx0

            src_points.append([x_c, y_c])
            dst_points.append([x_match + template_size//2, y_match + template_size//2])

    return np.array(src_points), np.array(dst_points), (H, W), moving, target

def apply_masked_interpolated_warp(src, dst, img_shape, moving_img, max_distance=400):
    H, W = img_shape
    displacements = dst - src 
    global_median = np.median(displacements, axis=0)
    
    grid_y, grid_x = np.mgrid[0:H:100, 0:W:100]
    full_src_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    print("Interpolating displacement field...")
    interp_dx = griddata(src, displacements[:, 0], full_src_grid, 
                         method='linear', fill_value=global_median[0])
    interp_dy = griddata(src, displacements[:, 1], full_src_grid, 
                         method='linear', fill_value=global_median[1])

    print("Applying distance-based stability mask...")
    tree = cKDTree(src)
    distances, _ = tree.query(full_src_grid)
    
    alpha = np.clip(1 - (distances / max_distance), 0, 1)
    final_dx = (interp_dx * alpha) + (global_median[0] * (1 - alpha))
    final_dy = (interp_dy * alpha) + (global_median[1] * (1 - alpha))

    full_dst_grid = full_src_grid + np.vstack([final_dx, final_dy]).T
    
    anchors_src = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])
    anchors_dst = anchors_src + global_median
    
    full_src_grid = np.vstack([full_src_grid, anchors_src])
    full_dst_grid = np.vstack([full_dst_grid, anchors_dst])

    print("Estimating Piecewise Affine Transform...")
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(full_dst_grid, full_src_grid)
    
    print("Warping image...")
    warped = transform.warp(moving_img, tform, output_shape=img_shape)
    
    return warped, full_src_grid, full_dst_grid, tform

# --- SAVE FUNCTIONS ---

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

# --- EXECUTION ---

TARGET_PATH = 'target_ds.tiff'
MOVING_PATH = 'moving_ds.tiff'

src_pts, dst_pts, shape, moving_gray, target_gray = generate_displacement_field(
    TARGET_PATH, MOVING_PATH,
    grid_size=(25, 25), 
    template_size=300, 
    search_radius=120
)

corrected_gray, f_src, f_dst, tform_model = apply_masked_interpolated_warp(
    src_pts, dst_pts, shape, moving_gray
)

# 1. Create and save the Magenta/Green diagnostic overlay
t_norm = exposure.rescale_intensity(target_gray, out_range=(0, 1))
c_norm = exposure.rescale_intensity(corrected_gray, out_range=(0, 1))
diagnostic_overlay = np.dstack((t_norm, c_norm, t_norm))
io.imsave('diagnostic_overlay.png', (diagnostic_overlay * 255).astype(np.uint8))

# 2. Create and save the Color overlay (using the original files)
save_color_overlay(TARGET_PATH, MOVING_PATH, tform_model, 'final_color_registration.png')

# 3. Final Plotting (as before)
plt.figure(figsize=(10,10))
plt.imshow(diagnostic_overlay)
plt.title("Magenta: Target | Green: Moving")
plt.axis('off')
plt.show()