# In this data set, 4 image types with 3 bands, 8 bands, 8 bands and 1 band for the same scene.
# In this function; images are read from file and re-sized at the same dimensions and stacked on top.
# So we have an image with 20 bands for each scene.

import glob
import tifffile as tiff
import numpy as np
import cv2

def make_stacked_images(width, height):
    path = 'data\\sixteen_band\\'

    for img_name in glob.glob('data\\three_band\\*.tif'):
        name = img_name.replace('data\\three_band\\', '').replace('.tif', '')
        # img_t -> 3 bands
        img_t = tiff.imread(img_name)
        # (3, h, w) -> (h, w, 3)
        img_t = np.rollaxis(img_t, 0, 3)
        img_t = cv2.resize(img_t, (height, width))

        # _A.tiff -> 8 bands
        img_sA = tiff.imread(path + name + '_A.tif')
        img_sA = np.rollaxis(img_sA, 0, 3)
        img_sA = cv2.resize(img_sA, (height, width))

        # _M.tiff -> 8 bands
        img_sM = tiff.imread(path + name + '_M.tif')
        img_sM = np.rollaxis(img_sM, 0, 3)
        img_sM = cv2.resize(img_sM, (height, width))

        # _A.tiff -> 1 band
        img_sP = tiff.imread(path + name + '_P.tif')
        img_sP = cv2.resize(img_sP, (height, width))
        img_sP = np.expand_dims(img_sP, axis=2)

        # stacked -> 20 bands
        stacked = np.dstack(((np.dstack((np.dstack((img_t, img_sA)), img_sM))), img_sP))

        tiff.imsave('scaled_data\\' + name + '.tif', stacked)

'''
if __name__ == '__main__':
    h = 500
    w = int(h * 1.014)

    make_stacked_images(w,h)
'''
