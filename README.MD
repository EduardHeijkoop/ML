Repository to create training and label data to train a ML model which detects buildings in high resolution satellite imagery.  
  
Create_WV_PanSharpened_Training_Data.py  
  
To run: python Create_WV_PanSharpened_Training_Data.py [--input_file </path/to/input_file.txt> --output_dir </path/to/output_dir/> --no_labels]  
  
Given a strip (i.e. a SETSM output) it will find the corresponding panchromatic and multispectral WorldView .NTF files, orthorectify both images, and then combine them to create a pan-sharpened image.  
It will then search for buildings in OpenStreetMap through the Overpass API and create labels at the same resolution as the pan-sharpened image.  
  
Preprocess_Images.py  
  
To run: python Preprocess_Image.py  
  
This will split images and labels into separate 224x224 pixel blocks to be read by the ResUNet.  
  
Train_Model_WV_PanSharpened.py  
  
To run: python Train_Model_WV_PanSharpened.py  
  
This script will train a Residual UNet (https://arxiv.org/pdf/1711.10684.pdf) from the training and label data generated previously.  
  
Get_Buildings.py  
  
To run: python Get_Buildings.py --model </path/to/model> --image </path/to/image/>  
  
Given a previously trained and saved model, this script will detect buildings in a high resolution, pan-sharpened, orthorectified image. Use Create_WV_PanSharpened_Training_Data with the --no_labels flag to create the data to run this on.  
  
Dependencies:  
Create_WV_PanSharpened_Training_Data:  
Python packages:  
numpy, pandas, geopandas, gdal/osgeo  
overpy, requests, re, shutil, xml  
  
Preprocess_Images:  
Python packages:  
numpy, pandas, gdal/osgeo  
  
Train_Model_WV_PanSharpened:  
Python packages:  
tensorflow (v2), numpy, gdal/osgeo  
  
Get_Buildings:  
Python packages:  
tensorflow (v2), numpy, gdal/osgeo, scipy