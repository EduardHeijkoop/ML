import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, osr
import glob
import subprocess

def preprocess_image(image,input_size):
    label_image = image.replace('Training_Data','Labels')
    src = gdal.Open(image,gdalconst.GA_ReadOnly)
    width = src.RasterXSize
    height = src.RasterYSize
    n_images_x = int(np.floor(width/input_size[0]))
    n_images_y = int(np.floor(height/input_size[1]))
    count = 0
    for i in range(n_images_x):
        for j in range(n_images_y):
            count = count+1
            count_str = f'{count:06d}'
            output_train_image = f'{"/".join(image.split("/")[0:-1])}/subimages/{image.split("/")[-1].replace(".tif","_"+count_str+".tif")}'
            output_label_image = f'{"/".join(label_image.split("/")[0:-1])}/subimages/{label_image.split("/")[-1].replace(".tif","_"+count_str+".tif")}'
            warp_train_command = f'gdal_translate -srcwin {i*input_size[0]} {j*input_size[1]} {input_size[0]} {input_size[1]} {image} {output_train_image}'
            warp_label_command = f'gdal_translate -srcwin {i*input_size[0]} {j*input_size[1]} {input_size[0]} {input_size[1]} {label_image} {output_label_image}'
            subprocess.run(warp_train_command,shell=True)
            subprocess.run(warp_label_command,shell=True)


def main():
    training_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/Training_Data/'
    label_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/Labels/'
    image_list = glob.glob(f'{training_dir}*.tif')
    input_size = (224,224)
    for image in image_list:
        preprocess_image(image,input_size)


if '__main__' == __name__:
    main()