import cv2
import numpy as np
from matplotlib import pyplot as plt
from pybioimageutils import visualize
from tifffile import imwrite

ref = cv2.imread('reference.tif')
moving = cv2.imread('moving.tif')
height, width, c = moving.shape # Reference dimensions

gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
gray_target = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

# 1. Create the ORB detector with a sufficient number of features
orb_detector = cv2.ORB_create(5000)

# 2. Find keypoints and descriptors for both images
kp1, d1 = orb_detector.detectAndCompute(np.array(gray_ref), None)
kp2, d2 = orb_detector.detectAndCompute(gray_target, None)

# 3. Match features between the two images using a Brute Force matcher
#    with Hamming distance as ORB produces binary descriptors
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(d1, d2)

# Keep only the top 15% of matches to filter out noise
good_matches = matches[:int(len(matches) * 0.50)]

# 5. Extract location of good matches
points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt


# 6. Find the homography matrix using RANSAC
#    RANSAC is a robust method to filter out outliers
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# 7. Warp the first image to align with the second image using the homography matrix
img1_aligned = cv2.warpPerspective(ref, h, (width, height))

# Convert the images to RGB

aligned = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB)
moved = cv2.cvtColor(moving, cv2.COLOR_BGR2RGB)

al = np.array(aligned)
re = np.array(moved)

merge = visualize.composite(al, re, normalize_images=False)

imwrite('merged_affine.tif', merge, photometric='rgb')

# plt.imshow(merge)
# plt.show()

# warp_mode = cv2.MOTION_AFFINE

# warp_matrix = np.eye(2, 3, dtype=np.float32)
# print(warp_matrix)

# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2500, 1e-1)

# (cc, warp_matrix) = cv2.findTransformECC(gray_ref, gray_target, warp_matrix, warp_mode, criteria)
# print(warp_matrix)
# #aligned_image = cv2.warpPerspective(moving, warp_matrix, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
# aligned_image = cv2.warpAffine(moving, warp_matrix, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# al = np.array(aligned_image)
# re = np.array(ref)

# merge = visualize.composite(al, re, normalize_images=False)

# plt.imshow(merge)
# plt.show()