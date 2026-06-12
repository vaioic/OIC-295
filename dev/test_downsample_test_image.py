import skimage as sk
from oic_toolkit import util

# --- Dataset 1 ---
img1 = sk.io.imread("../data/2026-04-21 Cropped for testing/259590_H3K9me3.ome.tif")

util.downsample_image(img1, output_path="../data/2026-06-08 Downsampled/259590_H3K9me3.tif")

img2 = sk.io.imread("../data/2026-04-21 Cropped for testing/259591_Ki67.ome.tif")

util.downsample_image(img2, output_path="../data/2026-06-08 Downsampled/259591_Ki67.tif")

# --- Dataset 2 ---
img1 = sk.io.imread("../data/HETS_CR_HFD_LAC/28630_H3K9me3_267440.ome.tif")
util.downsample_image(img1, output_path="../data/2026-06-08 Downsampled/28630_H3K9me3_267440.tif")

img2 = sk.io.imread("../data/HETS_CR_HFD_LAC/28630_Ki67_267441.ome.tif")
util.downsample_image(img2, output_path="../data/2026-06-08 Downsampled/28630_Ki67_267441.tif")

# --- Dataset 3 ---
img1 = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_H3K9me3_267412.ome.tif")
util.downsample_image(img1, output_path="../data/2026-06-08 Downsampled/30239_H3K9me3_267412.tif")

img2 = sk.io.imread("../data/HETS_CR_HFD_LAC/30239_Ki67_267466.ome.tif")
util.downsample_image(img2, output_path="../data/2026-06-08 Downsampled/30239_Ki67_267466.tif")

