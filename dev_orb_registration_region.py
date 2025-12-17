import skimage
import numpy as np
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from tifffile import imwrite
import cv2 as cv

#imageA = rgb2gray(skimage.io.imread('./export/259270_1.tif'))
imageA = cv.imread('./export/259270_1.tif', cv.IMREAD_GRAYSCALE)
imageB = cv.imread('./export/261082_1.tif', cv.IMREAD_GRAYSCALE)

imageA = imageA[5000:10000, 5000:10000]
imageB = imageB[5000:10000, 5000:10000]

# # Condition the images to have the same shape
# height_A, width_A = imageA.shape
# height_B, width_B = imageB.shape

# target_height = max(height_A, height_B)
# target_width = max(width_A, width_B)

# if height_A < target_height:
#     diff = target_height - height_A
#     imageA = np.pad(imageA, ((0, diff), (0, 0)), 'constant')

# if width_A < target_width:
#     diff = target_width - width_A
#     imageA = np.pad(imageA, ((0, 0), (0, diff)), 'constant')

# if height_B < target_height:
#     diff = target_height - height_B
#     imageB = np.pad(imageB, ((0, diff), (0, 0)), 'constant')

# if width_B < target_width:
#     diff = target_width - width_B
#     imageB = np.pad(imageB, ((0, 0), (0, diff)), 'constant')

# Initiate ORB detector
orb = cv.ORB_create(2000)

kpA, desA = orb.detectAndCompute(imageA, None)
kpB, desB = orb.detectAndCompute(imageB, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(np.float32(desA), np.float32(desB), k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kpA[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpB[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

    height, width = imageB.shape # Reference dimensions

    # 7. Warp the first image to align with the second image using the homography matrix
    aligned = cv.warpPerspective(imageA, M, (width, height))

    # build an RGB image with the registered sequence
    reg_im = np.zeros((imageB.shape[0], imageB.shape[1], 3))
    reg_im[..., 0] = aligned
    reg_im[..., 1] = imageB
    reg_im[..., 2] = imageB

    imwrite('./export/orb_registration2.tif', reg_im)

    # matchesMask = mask.ravel().tolist()

    # h,w = imageA.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    # dst = cv.perspectiveTransform(pts,M)

    # imageB = cv.polylines(imageB,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

