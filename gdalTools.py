import os, sys, time
import numpy as np
from osgeo import ogr, gdal, gdalconst
from osgeo import gdal_array as ga

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def stretch_n(bands, img_min, img_max, lower_percent=0, higher_percent=100):
    """
    :param bands:  目标数据，numpy格式
    :param img_min:   目标位深的最小值，以8bit为例，最大值为255， 最小值为0
    :param img_max:    目标位深的最大值
    :return:
    """
    out = np.zeros_like(bands).astype(np.float32)
    a = img_min
    b = img_max
    c = np.percentile(bands[:, :], lower_percent)
    d = np.percentile(bands[:, :], higher_percent)
    t = a + (bands[:, :] - c) * (b - a) / (d - c)
    t[t < a] = a
    t[t > b] = b
    out[:, :] = t
    return out


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




