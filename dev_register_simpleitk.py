import SimpleITK as sitk
import skimage
import numpy as np

imageA = skimage.color.rgb2gray(skimage.io.imread('./export/259270_1.tif'))
imageB = skimage.color.rgb2gray(skimage.io.imread('./export/261082_1.tif'))

imageA = imageA[5000:10000, 5000:10000]
imageB = imageB[5000:10000, 5000:10000]

fixed_image = sitk.GetImageFromArray(np.float32(imageA))
moving_image = sitk.GetImageFromArray(np.float32(imageB))

resultImage = sitk.Elastix(fixed_image, moving_image)

result_image_np = sitk.GetArrayFromImage(resultImage)

print(resultImage.shape)