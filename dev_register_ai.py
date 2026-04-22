import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, exposure, transform
from scipy.ndimage import shift

def generate_displacement_field(target_path, moving_path, grid_size=(40, 40), template_size=150, search_radius=100):
    """
    Scans the moving image for landmarks and finds their counterparts in the target image.
    """
    target = io.imread(target_path, as_gray=True)
    moving = io.imread(moving_path, as_gray=True)
    H, W = target.shape

    # 1. Setup Grid
    margin = template_size // 2 + search_radius
    y_coords = np.linspace(margin, H - margin, grid_size[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, grid_size[1], dtype=int)

    src_points = [] # Landmark locations in 'moving'
    dst_points = [] # Corresponding locations found in 'target'

    print(f"Scanning {grid_size[0]*grid_size[1]} grid points...")

    for y_c in y_coords:
        for x_c in x_coords:
            # Extract Template from Moving
            y0, x0 = y_c - template_size//2, x_c - template_size//2
            template = moving[y0:y0+template_size, x0:x0+template_size]

            # Skip background/empty areas
            if np.std(template) < 0.005: 
                continue

            # Define Search Window in Target
            wy0, wy1 = max(0, y0 - search_radius), min(H, y0 + template_size + search_radius)
            wx0, wx1 = max(0, x0 - search_radius), min(W, x0 + template_size + search_radius)
            target_window = target[wy0:wy1, wx0:wx1]

            # Perform Template Matching
            result = feature.match_template(target_window, template)
            ij = np.unravel_index(np.argmax(result), result.shape)
            
            # Global Match Coordinates in Target
            y_match = ij[0] + wy0
            x_match = ij[1] + wx0

            # Store the points (x, y format for skimage)
            src_points.append([x_c, y_c])
            dst_points.append([x_match + template_size//2, y_match + template_size//2])

    return np.array(src_points), np.array(dst_points), (H, W), moving, target

def apply_ultra_robust_warp(src, dst, img_shape, moving_img, grid_size):
    H, W = img_shape
    
    # 1. Calculate displacements
    displacements = dst - src
    
    # 2. Reshape to a 2D grid for smoothing
    # Note: This only works if you have a full grid (no skipped points)
    # If points were skipped, we skip this and stick to MAD filtering.
    try:
        dy = displacements[:, 1].reshape(grid_size)
        dx = displacements[:, 0].reshape(grid_size)
        
        # Apply a Gaussian filter to the SHIFTS themselves
        # sigma=1 or 2 will blend neighboring vectors and remove "spikes"
        dy_smoothed = gaussian_filter(dy, sigma=1.5)
        dx_smoothed = gaussian_filter(dx, sigma=1.5)
        
        # Flatten back
        displacements[:, 1] = dy_smoothed.flatten()
        displacements[:, 0] = dx_smoothed.flatten()
        dst = src + displacements
    except:
        print("Skipping Gaussian smoothing due to incomplete grid.")

    # 3. Proceed with PiecewiseAffineTransform
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(dst, src)
    return transform.warp(moving_img, tform, output_shape=img_shape)

def apply_robust_non_rigid_warp(src, dst, img_shape, moving_img):
    """
    Cleans the displacement field of outliers and warps the image elastically.
    """
    H, W = img_shape
    
    # 1. OUTLIER REMOVAL: Filter vectors that differ wildly from the consensus
    displacements = dst - src
    median_dist = np.median(displacements, axis=0)
    err = np.linalg.norm(displacements - median_dist, axis=1)
    
    # Keep vectors within 3x the median deviation (plus a small buffer)
    # This removes the "glitches" that cause shearing
    mask = err < (np.median(err) * 3 + 10) 
    clean_src = src[mask]
    clean_dst = dst[mask]
    
    print(f"Kept {len(clean_src)} valid vectors. Removed {len(src) - len(clean_src)} outliers.")

    # 2. BORDER ANCHORS: Pin down the image edges to prevent the frame from collapsing
    anchors = np.array([
        [0, 0], [W-1, 0], [0, H-1], [W-1, H-1],
        [W//2, 0], [W//2, H-1], [0, H//2], [W-1, H//2]
    ])
    clean_src = np.vstack([clean_src, anchors])
    clean_dst = np.vstack([clean_dst, anchors])

    # 3. NON-RIGID WARP: Piecewise Affine Transformation
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(clean_dst, clean_src) 
    warped = transform.warp(moving_img, tform, output_shape=img_shape)
    
    return warped, clean_src, clean_dst

# --- MAIN WORKFLOW ---

# Update file names as needed
TARGET_FILE = 'target_ds.tiff'
MOVING_FILE = 'moving_ds.tiff'

# 1. Generate local shifts
src, dst, shape, moving, target = generate_displacement_field(
    TARGET_FILE, 
    MOVING_FILE, 
    grid_size=(25, 25), 
    template_size=300, 
    search_radius=120
)

# 2. Apply robust warping
corrected_img, clean_src, clean_dst = apply_robust_non_rigid_warp(src, dst, shape, moving)



# 3. Prepare Overlay Visualization
t_norm = exposure.rescale_intensity(target, out_range=(0, 1))
w_norm = exposure.rescale_intensity(corrected_img, out_range=(0, 1))

# Magenta (Target) / Green (Warped) Overlay
overlay = np.dstack((t_norm, w_norm, t_norm))

# 4. Plot Results
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Subplot 1: The Cleaned Displacement Field
axes[0].imshow(target, cmap='gray', alpha=0.7)
axes[0].quiver(clean_src[:, 0], clean_src[:, 1], 
                clean_dst[:, 0] - clean_src[:, 0], 
                clean_dst[:, 1] - clean_src[:, 1], 
                color='cyan', angles='xy', scale_units='xy', scale=1, width=0.002)
axes[0].set_title("Target Image + Filtered Displacement Vectors")

# Subplot 2: The Final Non-Rigid Registration
axes[1].imshow(overlay)
axes[1].set_title("Final Non-Rigid Registration\n(Magenta: Target, Green: Corrected Moving)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()