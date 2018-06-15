# train_wkt file contains objects coordinates in images.
# grid_sizes file contains coordinates required to locate objects in images.
# In this function, coordinates read from train_wkt file and scale with grid_size files information and data set scale function.
# Calculated coordinates are printed to images to get object masks.

import pandas as pd
import tifffile as tiff
import glob
import numpy as np
from shapely import wkt
from shapely import affinity
from rasterio import features
import matplotlib.pyplot as plt

grid_sizes = pd.read_csv('data\\grid_sizes.csv', names=['imageId', 'Xmax', 'Ymin'])
train_wkt = pd.read_csv('data\\train_wkt.csv', names=['imageId', 'classType', 'MultipolygonWKT'])


def get_grid_size(img_id):
    gs = grid_sizes[grid_sizes.iloc[:, 0] == img_id]
    x_max = gs.Xmax.values[0]
    y_min = gs.Ymin.values[0]
    return x_max, y_min


# Calculating scaling coefficient for multipolygons using formula in GIS literature
def get_coefficients(img_id, w, h):
    x_max, y_min = get_grid_size(img_id)
    width = w * w / (w+1)
    heigth = h * h / (h+1)
    x_coeff = float(width) / float(x_max)
    y_coeff = float(heigth) / float(y_min)
    return x_coeff, y_coeff


# According to the images dimensions, a coordinates of the objects in the images are calculated.
def get_shapes(img_id, w, h):
    x_coeff, y_coeff = get_coefficients(img_id, w, h)
    polygons = []
    img = train_wkt[train_wkt.imageId == img_id]
    for ct in img.classType.values:
        p = img[img.classType == ct].MultipolygonWKT.values[0]
        s = wkt.loads(p)
        s = affinity.scale(s, xfact=x_coeff, yfact=y_coeff, origin=(0, 0, 0))
        polygons.append(s)
    return polygons


def rasterize_shape(polygons, w, h):
    geos = []
    for p in polygons:
        if p.wkt == "MULTIPOLYGON EMPTY":
            geo = np.zeros((h, w))
        else:
            geo = features.rasterize([p], out_shape=(h, w), fill=0, default_value=1)
        geos.append(geo)
    geos = np.stack(geos, axis=2)
    return geos


def create_masks(w, h):
    for img_name in glob.glob("scaled_data\\*.tif"):
        img_id = img_name.split("\\")[-1].replace(".tif", "")
        polygons = get_shapes(img_id, w, h)
        geos = rasterize_shape(polygons, w, h)
        tiff.imsave('masks\\' + img_id + '.tif', geos)

'''
if __name__ == '__main__':
    h = 500
    w = int(h * 1.014)

    create_masks(w, h)
'''