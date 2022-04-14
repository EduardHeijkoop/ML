import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, osr
import glob

def preprocess_image(image):
    src = gdal.Open(image,gdalconst.GA_ReadOnly)


def main():
    training_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/Training_Data/'
    label_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/Labels/'
    image_list = glob.glob(f'{training_dir}*.tif')


if '__main__' == __name__:
    main()