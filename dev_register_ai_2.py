import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, exposure, transform
from scipy.interpolate import RBFInterpolator

def generate_full_registration(target_path, moving_path, grid_size=(20, 20), template_size=250):
    # 1. Load Images
    target = io.imread(target_path, as_gray=True)
    moving = io.imread(moving_path, as_gray=True)
    H, W = target.shape

    # 2. Setup Grid & Match Landmarks
    search_radius = 100
    margin = template_size // 2 + search_radius
    y_coords = np.linspace(margin, H - margin, grid_size[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, grid_size[1], dtype=int)

    src_pts = [] # Moving
    dst_pts = [] # Target

    print(f"Finding landmarks in {grid_size[0] * grid_size[1]} regions...")
    
    for y_c in y_coords:
        for x_c in x_coords:
            y0, x0 = y_c - template_size//2, x_c - template_size//2
            template = moving[y0:y0+template_size, x0:x0+template_size]

            if np.std(template) < 0.005: continue # Skip empty areas

            # Local Search Window
            wy0, wy1 = max(0, y0 - search_radius), min(H, y0 + template_size + search_radius)
            wx0, wx1 = max(0, x0 - search_radius), min(W, x0 + template_size + search_radius)
            
            result = feature.match_template(target[wy0:wy1, wx0:wx1], template)
            ij = np.unravel_index(np.argmax(result), result.shape)
            
            src_pts.append([x_c, y_c])
            dst_pts.append([ij[1] + wx0 + template_size//2, ij[0] + wy0 + template_size//2])

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    # 3. Outlier Filtering (Remove the "Shear-causers")
    displacements = dst_pts - src_pts
    median_disp = np.median(displacements, axis=0)
    errors = np.linalg.norm(displacements - median_disp, axis=1)
    
    # Filter points: Only keep those that agree with the general consensus
    mask = errors < (np.median(errors) * 2 + 10)
    clean_src = src_pts[mask]
    clean_dst = dst_pts[mask]
    
    print(f"Filtering: {len(clean_src)}/{len(src_pts)} points kept.")

    # 4. Thin Plate Spline (TPS) Interpolation
    # This creates a smooth warp field that ignores gaps and resists shearing
    # 'thin_plate_spline' kernel is the gold standard for biological warping
    interp = RBFInterpolator(clean_src, clean_dst, kernel='thin_plate_spline', smoothing=10)

    # Create a dense coordinate grid for the whole image
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    flat_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Predict where every pixel should move
    print("Computing smooth warp field...")
    warped_coords = interp(flat_grid)
    
    # Reshape back to image dimensions for the warp function
    map_x = warped_coords[:, 0].reshape((H, W))
    map_y = warped_coords[:, 1].reshape((H, W))

    # 5. Apply the Warp
    # We use transform.warp with the map coordinates
    corrected = transform.warp(moving, np.array([map_y, map_x]), order=1)

    # 6. Final Visualization
    t_norm = exposure.rescale_intensity(target, out_range=(0, 1))
    c_norm = exposure.rescale_intensity(corrected, out_range=(0, 1))
    overlay = np.dstack((t_norm, c_norm, t_norm))

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(target, cmap='gray', alpha=0.5)
    ax[0].quiver(clean_src[:, 0], clean_src[:, 1], 
                 clean_dst[:, 0] - clean_src[:, 0], 
                 clean_dst[:, 1] - clean_src[:, 1], 
                 color='cyan', angles='xy', scale_units='xy', scale=1, width=0.002)
    ax[0].set_title("Filtered Landmark Vectors")

    ax[1].imshow(overlay)
    ax[1].set_title("TPS Smoothed Overlay (Magenta: Target, Green: Corrected)")
    
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

    return corrected

# Run the full pipeline
final_result = generate_full_registration('target_ds.tiff', 'moving_ds.tiff')