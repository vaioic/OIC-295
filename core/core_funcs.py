from pathlib import Path

import numpy as np
import skimage as sk
import tifffile
from tqdm import tqdm


def export_stacked_tiff(target, corrected, output_path):

    print("Converting images to HED...")

    print(f"Target type: {target.dtype}")
    print(f"Target range: {np.max(target), np.min(target)}")
    
    print(f"Corrected type: {corrected.dtype}")
    print(f"Corrected range: {np.max(corrected), np.min(corrected)}")
    
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


def process_directory(input_dir, output_dir):
    
    # Expect images to be labeled as <slide_number>_H3K9me3 or Ki67

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    elif isinstance(input_dir, Path):
        pass
    else:
        raise ValueError(f"Expected input_dir to be a str or Path. Instead it is a {type(input_dir)}.")
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(f"Expected output_dir to be a str or Path. Instead it is a {type(output_dir)}.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif output_dir.is_file():
        raise ValueError(f"Expected output_dir to be a directory but apparently points to a file.")
    
    file_list = list(input_dir.glob("*Ki67*.ome.tif"))

    for file in file_list:

        # Get the corresponding other stain
        sample_id = (file.name).split("_")[0]

        H3_file = list(input_dir.glob(sample_id + "_H*K9*.ome.tif"))

        if not H3_file:
            raise FileNotFoundError(f"File was not found")
        elif len(H3_file) > 1:
            raise ValueError(f"Expected only one matching file. Instead {len(H3_file)} were found.")
        
        process_images(H3_file[0], file, output_dir)

    
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

    src, dst = calculate_displacement_field(target_gray, moving_gray, grid_size=(30, 30), template_size=200, search_window=100)

    # generate_quiver_plot(target_gray, src, dst)
    # exit()

    tform, _, _ = estimate_tform(src, dst, target_gray.shape, max_distance=200)

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
