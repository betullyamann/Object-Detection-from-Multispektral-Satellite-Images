# In this function; scaled images and masks converted numpy arrays.
# and these arrays converted to feature and class vectors.
# Pixels are convoluted using different filter sizes and generated different feature vectors.

import tifffile as tiff
import glob
import numpy as np
import cv2

def set_feature_and_mask_vector():
    images = []
    masks = []

    conv_coef = [1, 3, 5, 7, 9, 11, 13]
    filters = [np.array([1], np.float32),
               np.array([[1, 1, 1],
                         [1, 3, 1],
                         [1, 1, 1]], np.float32)/11.0,
               np.array([[1, 1, 1, 1, 1],
                         [1, 3, 3, 3, 1],
                         [1, 3, 9, 3, 1],
                         [1, 3, 3, 3, 1],
                         [1, 1, 1, 1, 1]], np.float32)/49.0,
               np.array([[1, 1, 1, 1, 1, 1, 1],
                         [1, 3, 3,  3, 3, 3, 1],
                         [1, 3, 9,  9, 9, 3, 1],
                         [1, 3, 9, 27, 9, 3, 1],
                         [1, 3, 9,  9, 9, 3, 1],
                         [1, 3, 3,  3, 3, 3, 1],
                         [1, 1, 1,  1, 1, 1, 1]], np.float32)/171.0,
               np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 3, 3,  3, 3,  3,  3, 3, 1],
                         [1, 3, 9,  9, 9,  9,  9, 3, 1],
                         [1, 3, 9, 27, 27, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 27, 27, 9, 3, 1],
                         [1, 3, 9,  9,  9,  9, 9, 3, 1],
                         [1, 3, 3,  3,  3,  3, 3, 3, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32)/545.0,
               np.array([[1, 1, 1,  1,  1,  1,   1,  1, 1, 1, 1],
                         [1, 3, 3,  3,  3,  3,   3,  3, 3, 3, 1],
                         [1, 3, 9,  9,  9,  9,   9,  9, 9, 3, 1],
                         [1, 3, 9, 27, 27, 27,  27, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 81,  81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 243, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 81,  81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 27, 27,  27, 27, 9, 3, 1],
                         [1, 3, 9,  9,  9,  9,   9,  9, 9,  3, 1],
                         [1, 3, 3,  3,  3,  3,   3,  3, 3, 3, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32) / 1671.0,
               np.array([[1, 1, 1,  1,  1,   1,   1,   1,  1,  1, 1, 1, 1],
                         [1, 3, 3,  3,  3,   3,   3,   3,  3,  3, 3, 3, 1],
                         [1, 3, 9,  9,  9,   9,   9,   9,  9,  9, 9, 3, 1],
                         [1, 3, 9, 27, 27,  27,  27,  27, 27, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81,  81,  81,  81, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 243, 243, 243, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 243, 729, 243, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81, 243, 243, 243, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 81,  81,  81,  81, 81, 27, 9, 3, 1],
                         [1, 3, 9, 27, 27,  27,  27,  27, 27, 27, 9, 3, 1],
                         [1, 3, 9,  9,  9,   9,   9,   9,  9,  9, 9, 3, 1],
                         [1, 3, 3,  3,  3,   3,   3,   3,  3,  3, 3, 3, 1],
                         [1, 1, 1,  1,  1,   1,   1,   1,  1,  1, 1, 1, 1]],np.float32) / 5061.0]

    # scales images and masks converted numpy arrays
    for image_name, mask_name in zip(glob.glob("scaled_data\\*.tif"), glob.glob("masks\\*.tif")):
        images.append(tiff.imread(image_name))
        masks.append(tiff.imread(mask_name))

    # arrays converted to feature and class vectors using different convolution filters.
    for f, i in zip(filters, conv_coef):
        pixels = []
        labels = []
        for img, mask in zip(images, masks):
            for i_row, m_row in zip(img, mask):
                for pixel, label in zip(i_row, m_row):
                    pixels.append(cv2.filter2D(pixel, -1, f))
                    labels.append(label)
        np.save("pixels_" + str(i), np.array(pixels))
        np.save("labels", np.array(labels))


def class_11():
    new_labels = []
    new_pixels = []
    conv = [1]

    labels = np.load("labels.npy").tolist()
    for c in conv:
        pixels = np.load("pixels_" + str(c) + ".npy").tolist()
        for label, pixel, i in zip(labels, pixels, range(len(labels))):
            sum = np.sum(label)
            if sum == 0:
                label.append(1)
                new_labels.append(label)
                new_pixels.append(pixel)
            elif sum == 1:
                label.append(1)
                new_labels.append(label)
                new_pixels.append(pixel)

        new_labels = np.array(new_labels)
        new_pixels = np.array(new_pixels)

        print(new_pixels.shape)
        print(new_labels.shape)

if __name__ == '__main__':
    #set_feature_and_mask_vector()
    class_11()
