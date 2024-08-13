## Example: A simple example to obtain distsance map and boundary map
import numpy as np
import os
import cv2
from osgeo import gdal
import scipy.ndimage as sn

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset



maskRoot = r"D:\longJ\MAP\train\mask"
boundaryRoot = r"D:\longJ\MAP\train\contour"

for imgPath in os.listdir(maskRoot):
    input_path = os.path.join(maskRoot, imgPath)
    boundaryOutPath = os.path.join(boundaryRoot, imgPath)
    im_proj, im_geotrans, im_width, im_height, im_data = read_img(input_path)
    im_data[im_data>0] = 255
    im_data[im_data==0] = 0
    boundary = cv2.Canny(im_data, 100, 200)
    # dilation
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(boundary, kernel, iterations=1)
    write_img(boundaryOutPath, im_proj, im_geotrans, boundary)





