import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, measure, transform, exposure, filters
from scipy.ndimage import shift
import core_functions
from skimage.feature import plot_matched_features

def register_histology_slices(target_path, moving_path):
    # 1. Load images (Assuming grayscale for registration)
    # If they are RGB, convert to grayscale
    img_target = io.imread(target_path, as_gray=True)
    img_moving = io.imread(moving_path, as_gray=True)

    img_target, img_moving = core_functions.match_image_size(img_target, img_moving)

    t_proc = filters.gaussian(exposure.equalize_hist(img_target), sigma=2)
    m_proc = filters.gaussian(exposure.equalize_hist(img_moving), sigma=2)

    # img_target = img_target[::4, ::4, :]
    # img_moving = img_moving[::4, ::4, :]

    # 2. Pre-process: Contrast Stretching
    # This helps ORB find features in weakly stained areas
    t_proc = exposure.rescale_intensity(t_proc, in_range='image', out_range=(0, 1))
    m_proc = exposure.rescale_intensity(m_proc, in_range='image', out_range=(0, 1))

    # 3. Initialize ORB and extract features
    # We increase n_keypoints to 1000 to handle complex tissue structures
    detector = feature.ORB(n_keypoints=1000)

    detector.detect_and_extract(t_proc)
    kp_target = detector.keypoints
    des_target = detector.descriptors

    detector.detect_and_extract(m_proc)
    kp_moving = detector.keypoints
    des_moving = detector.descriptors

    # 4. Match descriptors
    matches = feature.match_descriptors(des_target, des_moving, cross_check=True)

    # Filter coordinates of matched keypoints
    dst = kp_target[matches[:, 0]] # Target
    src = kp_moving[matches[:, 1]] # Moving

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_matched_features(img_target, img_moving, keypoints0=kp_target, keypoints1=kp_moving, matches=matches, ax=ax)
    plt.show()

    exit()

    # 5. Robustly estimate translation using RANSAC
    model_robust, inliers = measure.ransac(
        (src, dst), 
        transform.EuclideanTransform, 
        min_samples=2, 
        residual_threshold=5, # Increased threshold to be more forgiving
        max_trials=1000       # Increased trials to give it more chances to find a match
    )

    # --- SAFETY CHECK ---
    if model_robust is None:
        print("RANSAC failed to find a valid transformation.")
        # Fallback: use the median shift of all matches
        shifts = dst - src
        y_shift, x_shift = np.median(shifts, axis=0)
        print(f"Using fallback Median Shift: Row (Y): {y_shift:.2f}, Col (X): {x_shift:.2f}")
    else:
        x_shift, y_shift = model_robust.translation
        print(f"RANSAC Success! Inliers: {np.sum(inliers)}")

    # 6. Apply the correction (using the determined shifts)
    corrected = shift(img_moving, shift=(y_shift, x_shift), order=1)

    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_target, cmap='gray')
    axes[0].set_title("Target (Reference)")
    
    axes[1].imshow(img_moving, cmap='gray')
    axes[1].set_title("Original Moving")
    
    # Simple overlay to check registration
    overlay = np.dstack((img_target, corrected, img_target))
    axes[2].imshow(overlay)
    axes[2].set_title("Registered Overlay")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return corrected, (y_shift, x_shift)

# --- Execution ---
result, final_shift = register_histology_slices("target_ds.tiff", "moving_ds.tiff")