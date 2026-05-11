import skimage as sk
import tifffile
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm

def read_image_and_downsample(image_path, downsample=0.25):

    image_path = Path(image_path)

    img_original = sk.io.imread(image_path)
    
    if not downsample is None:
        img_rescaled = sk.transform.rescale(img_original, downsample, anti_aliasing=False, channel_axis=2)
    else:
        img_rescaled = img_original

    # Try using the hemotoxylin channel only
    #img_hed = sk.color.rgb2hed(img_rescaled)

    img_gray = sk.color.rgb2gray(img_rescaled)
    #img_gray = img_hed[..., 0]
    #img_gray = sk.util.img_as_float(img_gray)

    return img_gray, img_rescaled

def calculate_displacement_field(target, moving, grid_size=(30, 30), template_size=200, search_window=100):

    H, W = target.shape
    
    # Generate the image grid
    margin = template_size // 2 + search_window
    y_coords = np.linspace(margin, H - margin, grid_size[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, grid_size[1], dtype=int)

    src_points = []
    dst_points = []

    for yy in tqdm(y_coords):
        for xx in x_coords:

            # Calculate the top-left corner position
            y0, x0 = yy - template_size//2, xx - template_size//2

            template = moving[y0:y0 + template_size,
                              x0:x0 + template_size]
            
            # Skip patch if there is no image information
            if np.std(template) < 0.005:
                continue

            # Define the search window coordinates to reduce computation time 
            # and avoid off-target matches. The window is clipped to the image
            # size.
            wy0, wy1 = max(0, y0 - search_window), min(H, y0 + template_size + search_window)
            wx0, wx1 = max(0, x0 - search_window), min(W, x0 + template_size + search_window)

            # Perform the template matching
            try:
                corr_coeff = sk.feature.match_template(target[wy0:wy1, wx0:wx1], template)
            except Exception:
                continue

            # Find the highest response
            max_ij = np.unravel_index(np.argmax(corr_coeff), corr_coeff.shape)

            src_points.append([xx, yy])  #Note: Points with no image data are skipped
            dst_points.append([max_ij[1] + wx0 + template_size//2, max_ij[0] + wy0 + template_size//2])
        
    return np.array(src_points), np.array(dst_points)

def estimate_tform(src, dst, target_shape, mesh_grid=(100, 100), max_distance=300, corners="median"):

    print("Calculating displacement field...")

    H, W = target_shape
    
    # Calculate the displacement field
    displacements = dst - src 
    
    # Generate the source grid (over the target image)
    grid_y, grid_x = np.mgrid[0:H:mesh_grid[0], 0:W:mesh_grid[1]]
    full_src_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Interpolate the displacement field over the entire mesh. This helps to 
    # smooth variations in the displacement field and avoid "tearing" the image. The function uses the median value as a fallback for regions where the interpolation fails (typically along the image border).
    interp_dx = griddata(src, displacements[:, 0], full_src_grid, 
                         method='linear', fill_value=np.median(displacements[:, 0]))
    interp_dy = griddata(src, displacements[:, 1], full_src_grid, 
                         method='linear', fill_value=np.median(displacements[:, 1]))
    
    global_median = np.median(displacements, axis=0)

    if max_distance is not None:
        print("Applying distance-based stability mask...")
        tree = cKDTree(src)
        distances, _ = tree.query(full_src_grid)
        
        alpha = np.clip(1 - (distances / max_distance), 0, 1)
        final_dx = (interp_dx * alpha) + (global_median[0] * (1 - alpha))
        final_dy = (interp_dy * alpha) + (global_median[1] * (1 - alpha))
    else:
        final_dx = interp_dx
        final_dy = interp_dy       


    # Reconstruct the full destination grid
    full_dst_grid = full_src_grid + np.vstack([final_dx, final_dy]).T

    match corners:
        case "static":

            # Add static corner anchors to keep the frame square
            anchors = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])
            full_src_grid = np.vstack([full_src_grid, anchors])
            full_dst_grid = np.vstack([full_dst_grid, anchors])

        case "median":
            anchors_src = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])
            anchors_dst = anchors_src + global_median
            
            full_src_grid = np.vstack([full_src_grid, anchors_src])
            full_dst_grid = np.vstack([full_dst_grid, anchors_dst])

    # Apply Piecewise Affine Transform to calculate the final warp matrix
    tform = sk.transform.PiecewiseAffineTransform()
    tform.estimate(full_dst_grid, full_src_grid)
    print("Done.")
    
    return tform, full_src_grid, full_dst_grid

def warp_image(moving, tform, output_shape):
    
    print("Warping image...")
    # Calculate the corrected moving image
    warped = sk.transform.warp(moving, tform, output_shape=output_shape)
    print("Done.")

    return warped

def generate_quiver_plot(target, src, dst):
    plt.figure(figsize=(15, 15))
    plt.imshow(target, cmap='gray', alpha=0.7)
    
    # Calculate displacement vectors
    dx = dst[:, 0] - src[:, 0]
    dy = dst[:, 1] - src[:, 1]
    
    # Plot original raw vectors in Red
    plt.quiver(src[:, 0], src[:, 1], dx, dy, 
               color='red', angles='xy', scale_units='xy', scale=1, 
               width=0.002, label='Raw Matches')
    plt.show()

def export_stacked_tiff(target, corrected, output_path):

    print("Converting images to HED...")
    # Split into HED channels
    target_HED = sk.color.rgb2hed(target)
    target_HED = np.clip(target_HED, 0, 1)
    target_HED = sk.util.img_as_ubyte(target_HED)

    corrected_HED = sk.color.rgb2hed(corrected)
    corrected_HED = np.clip(corrected_HED, 0, 1)
    corrected_HED = sk.util.img_as_ubyte(corrected_HED)

    # Write all four channels -- the shape should be (channel, Y, X)
    stacked = np.stack(
        [target_HED[..., 0],
        target_HED[..., 2],
        corrected_HED[..., 0],
        corrected_HED[..., 2]],
        axis=0)

    print(stacked.shape)

    channel_names = ['Ki67_H', 'Ki67_D', 'H3K9me3_H', 'H3K9me3_D']

    print("Writing TIFF file...")
    # Save as OME-TIFF
    tifffile.imwrite(
        output_path,
        stacked,
        metadata={'axes': 'CYX', 'Channel': {'Name': channel_names}},
        ome=True
    )
    print("Done.")

def merge_images(target, moving, alpha=0.5):

    if len(target.shape) == 3:
        target = sk.color.rgb2gray(target)
    
    if len(moving.shape) == 3:
        moving = sk.color.rgb2gray(moving)
    
    merged_rgb = np.zeros((target.shape[0], target.shape[1], 3), target.dtype)

    merged_rgb[..., 0] = alpha * target
    merged_rgb[..., 1] = (1 - alpha) * moving
    merged_rgb[..., 2] = alpha * target

    return merged_rgb


def process_directory(input_dir, output_dir):

    # Expect images to be labeled as <slide_number>_H3K9me3 or Ki67
    
    pass

def process_images(target_path, moving_path, output_dir):

    if isinstance(target_path, str):
        target_path = Path(target_path)
    elif isinstance(target_path, Path):
        pass
    else:
        raise ValueError(f"Expected target_path to be a str or Path. Instead it has type {type(target_path)}.")
    
    if isinstance(moving_path, str):
        moving_path = Path(moving_path)
    elif isinstance(moving_path, Path):
        pass
    else:
        raise ValueError(f"Expected moving_path to be a str or Path. Instead it has type {type(moving_path)}.")
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(f"Expected output_dir to be a str or Path. Instead it has type {type(output_dir)}.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif output_dir.is_file():
        raise ValueError(f"Expected output_dir to specify a directory, but it appears to be a file.")
    
    target_gray, target_rgb = read_image_and_downsample(target_path)
    moving_gray, moving_rgb = read_image_and_downsample(moving_path)

    src, dst = calculate_displacement_field(target_gray, moving_gray, grid_size=(30, 30), template_size=600, search_window=100)

    # generate_quiver_plot(target_gray, src, dst)
    # exit()

    tform, _, _ = estimate_tform(src, dst, target_gray.shape)

    # Generate the warped RGB image
    warped_rgb = warp_image(moving_rgb, tform, target_rgb.shape)

    #--- Save Data ---

    # Generate an output filename - I expect that the files will start with the slice #
    split_str = (target_path.name).split('_')
    sample_id = split_str[0]

    if len(split_str) >= 3:
        image_id = (split_str[2]).split('.')[0]
        combined_id = sample_id + "_" + image_id
    else:
        combined_id = sample_id
    # print(sample_id)   

    print("Exporting RGB image")
    export_stacked_tiff(target_rgb, warped_rgb, output_dir / (combined_id + "_stacked.tif"))
    
    # Generate the magenta-green image pair
    merged_rgb = merge_images(target_gray, warped_rgb)

    # Save a magenta-green image pair
    print("Saving magenta-green image pair")
    sk.io.imsave(output_dir / (combined_id + "_merged.tif"), merged_rgb)

    # Save the source and destination points to recreate the affine transform later
    np.savez(output_dir / (combined_id + "_pts.npz"), src=src, dst=dst)
   
    # Save the final outputs
    sk.io.imsave(output_dir / (combined_id + "_target.tif"), target_rgb)
    sk.io.imsave(output_dir / (combined_id + "_moving.tif"), moving_rgb)
    sk.io.imsave(output_dir / (combined_id + "_corrected.tif"), warped_rgb)
    
if __name__ == "__main__":

    target_path = r"\\pn.vai.org\projects_primary\pospisilik\vari-core-generated-data\POSP_Josef_IHC_SwissRollOverlay\HETS_CR_HFD_LAC\30636_H3K9me3_267436.ome.tif"

    moving_path = r"\\pn.vai.org\projects_primary\pospisilik\vari-core-generated-data\POSP_Josef_IHC_SwissRollOverlay\HETS_CR_HFD_LAC\30636_Ki67_267437.ome.tif"

    process_images(target_path, moving_path, "../processed/2026-05-11 Dev Optimizing/30636")

    # target_path = r"..\data\2026-04-21 Cropped for testing\259591_Ki67.ome.tif"

    # moving_path = r"..\data\2026-04-21 Cropped for testing\259590_H3K9me3.ome.tif"

    # process_images(target_path, moving_path, "../processed/2026-05-11 Dev Limit")  

    # Run test images
    #img = read_image_and_downsample()

