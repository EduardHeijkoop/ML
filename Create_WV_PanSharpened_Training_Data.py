import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
import subprocess
import argparse
from osgeo import gdal,gdalconst,osr,ogr
import overpy
import shapely
import datetime
import requests
import glob
import shutil
import warnings
import re
import xml.etree.ElementTree as ET

def get_extent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print(x,y)
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        y,x,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def get_raster_extents(raster,global_local_flag='global'):
    '''
    Get global or local extents of a raster
    '''
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
    local_ext = get_extent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    global_ext = reproject_coords(local_ext,src_srs,tgt_srs)
    x_local = [item[0] for item in local_ext]
    y_local = [item[1] for item in local_ext]
    x_min_local = np.nanmin(x_local)
    x_max_local = np.nanmax(x_local)
    y_min_local = np.nanmin(y_local)
    y_max_local = np.nanmax(y_local)
    x_global = [item[0] for item in global_ext]
    y_global = [item[1] for item in global_ext]
    x_min_global = np.nanmin(x_global)
    x_max_global = np.nanmax(x_global)
    y_min_global = np.nanmin(y_global)
    y_max_global = np.nanmax(y_global)
    if global_local_flag.lower() == 'global':
        return x_min_global,x_max_global,y_min_global,y_max_global
    elif global_local_flag.lower() == 'local':
        return x_min_local,x_max_local,y_min_local,y_max_local
    else:
        return None

def get_osm_buildings(lon_min,lon_max,lat_min,lat_max):
    '''
    Given a lon/lat extent (order for OSM is lat/lon),
    downloads all buildings in that region
    Returns result, which is an overpy structure
    '''
    api = overpy.Overpass()
    bbox = str(lat_min)+','+str(lon_min)+','+str(lat_max)+','+str(lon_max)
    result = api.query("""
    [out:json][timeout:900][maxsize:1073741824];
    (
    way["building"]("""+bbox+""");
    relation["building"]("""+bbox+""");
    );
    out body;
    >;
    out skel qt;
    """)
    return result

def overpy_to_gdf(overpy_struc):
    '''
    Given an overpy structure, subsets into numpy arrays of lon/lat
    Turns these into a Shapely polygon which can be used in a GeoDataFrame
    OpenStreetMap Way ID is included for added information (e.g. reverse lookup)
    Lines are of length 2, can't turn those into a polygon, so they are skipped
    '''
    gdf = gpd.GeoDataFrame()
    for way in overpy_struc.ways:
        lon = np.asarray([float(node.lon) for node in way.nodes])
        lat = np.asarray([float(node.lat) for node in way.nodes])
        if len(lon) < 3:
            continue
        lonlat = np.stack((lon,lat),axis=-1)
        poly = shapely.geometry.Polygon(lonlat)
        way_id = way.id
        tmp_start_date = way.tags['start_date'] if 'start_date' in way.tags.keys() else 'nan'
        tmp_gdf = gpd.GeoDataFrame(pd.DataFrame({'WAY_ID':[way.id],'start_date':[tmp_start_date]}),crs='EPSG:4326',geometry=[poly])
        gdf = pd.concat([gdf,tmp_gdf])
    gdf = gdf.to_crs('EPSG:4326').reset_index(drop=True)
    return gdf

def geocode_location(location_name):
    location_name = find_country_string(location_name)
    geocode_return = requests.get(f'https://geocode.xyz/{location_name}?json=1')
    if geocode_return.status_code != 200:
        raise ValueError('Can\'t access the geocode server!')
    geocode_json = geocode_return.json()
    if 'error' in geocode_json.keys():
         raise ValueError('Can\'t find this location!')
    lon = float(geocode_json['longt'])
    lat = float(geocode_json['latt'])
    return lon,lat

def find_country_string(loc):
    loc_re = re.findall(r'[A-Z][a-z]*|[a-z]+',loc)
    idx_len = np.asarray([len(l) for l in loc_re])
    idx_repeating = np.argwhere(np.logical_and(idx_len[:-1]==1,idx_len[1:]==1)).squeeze()
    idx_all = np.arange(len(loc_re))
    idx_non_repeating = np.setdiff1d(idx_all,idx_repeating+1)
    full_loc = ''
    for idx in idx_non_repeating:
        if idx in idx_repeating:
            full_loc += '%20' + loc_re[idx] + loc_re[idx+1]
        else:
            full_loc += '%20'+loc_re[idx]
    if full_loc[0:3] == '%20':
        full_loc = full_loc[3:]
    return full_loc

def unzip_strip_archive(strip,tmp_dir):
    strip_name = strip.split('/')[-1]
    if np.any(['UTM' in s for s in strip.split('/')]) == True:
        archive_dir = f'{strip.split(strip_name.split("_2m")[0])[0].split("UTM")[0].replace("PRODUCTS","DATA")}ARCHIVE/'
    else:
        archive_dir = f'{strip.split(strip_name.split("_2m")[0])[0].replace("PRODUCTS","DATA")}ARCHIVE/'
    strip_code = strip_name.split('_')[2]
    zip_list_direct_match = glob.glob(f'{archive_dir}{strip_code}*.zip')
    if len(zip_list_direct_match) == 1:
        zip_file = zip_list_direct_match[0]
        unzipped_dir = f'{tmp_dir}{strip_code}/'
        unzip_command = f'unzip -qq {zip_file} -d {unzipped_dir}'
        print('Unzipping...')
        subprocess.run(unzip_command,shell=True)
        print('Unzipping done.')
        zip_ID = 'single'
        return unzipped_dir,zip_ID
    elif len(zip_list_direct_match) == 0:
        zip_list = glob.glob(f'{archive_dir}*{strip_code}*.zip')
        zip_list.sort()
        if len(zip_list) == 0:
            unzipped_archive_dir_list = glob.glob(f'{archive_dir}{"_".join(strip_name.split("_")[0:3])}*')
            if len(unzipped_archive_dir_list) == 0:
                print('No zip files found!')
                return None,None
            else:
                unzipped_archive_dir = unzipped_archive_dir_list[0]
                unzipped_dir = f'{tmp_dir}{strip_code}/'
                os.mkdir(unzipped_dir)
                copy_command = f'cp -r {unzipped_archive_dir}/* {unzipped_dir}'
                print('Copying already unzipped files...')
                subprocess.run(copy_command,shell=True)
                zip_ID = 'single_unzipped'
                return unzipped_dir,zip_ID
        else:
            unzipped_dir = f'{tmp_dir}{strip_code}/'
            os.mkdir(unzipped_dir)
            print('Unzipping...')
            for zip_file in zip_list:
                unzip_command = f'unzip -qq {zip_file} -d {unzipped_dir}{zip_file.split("/")[-1].split(".zip")[0]}/'
                subprocess.run(unzip_command,shell=True)
            print('Unzipping done.')
            zip_ID = 'multiple'
            return unzipped_dir,zip_ID
    elif len(zip_list_direct_match) > 1:
        print('More than one zip file with a direct match found!')
        return None,None

def get_centroid_imd_file(imd):
    with open(imd) as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            if 'ULLon =' in line:
                ullon = float(line.split('ULLon =')[1].replace(';',''))
            if 'ULLat =' in line:
                ullat = float(line.split('ULLat =')[1].replace(';',''))
            if 'URLon =' in line:
                urlon = float(line.split('URLon =')[1].replace(';',''))
            if 'URLat =' in line:
                urlat = float(line.split('URLat =')[1].replace(';',''))
            if 'LLLon =' in line:
                lllon = float(line.split('LLLon =')[1].replace(';',''))
            if 'LLLat =' in line:
                lllat = float(line.split('LLLat =')[1].replace(';',''))
            if 'LRLon =' in line:
                lrlon = float(line.split('LRLon =')[1].replace(';',''))
            if 'LRLat =' in line:
                lrlat = float(line.split('LRLat =')[1].replace(';',''))
    lon_center = np.mean((ullon,urlon,lllon,lrlon))
    lat_center = np.mean((ullat,urlat,lllat,lrlat))
    outline = shapely.geometry.Polygon([(ullon,ullat),(urlon,urlat),(lrlon,lrlat),(lllon,lllat),(ullon,ullat)])
    return lon_center,lat_center,outline


def get_centroid_xml_file(xml):
    xml_tree = ET.parse(xml)
    xml_root = xml_tree.getroot()
    ullon = float(xml_root.findall('IMD/BAND_P/ULLON')[0].text)
    ullat = float(xml_root.findall('IMD/BAND_P/ULLAT')[0].text)
    urlon = float(xml_root.findall('IMD/BAND_P/URLON')[0].text)
    urlat = float(xml_root.findall('IMD/BAND_P/URLAT')[0].text)
    lllon = float(xml_root.findall('IMD/BAND_P/LLLON')[0].text)
    lllat = float(xml_root.findall('IMD/BAND_P/LLLAT')[0].text)
    lrlon = float(xml_root.findall('IMD/BAND_P/LRLON')[0].text)
    lrlat = float(xml_root.findall('IMD/BAND_P/LRLAT')[0].text)
    lon_center = np.mean((ullon,urlon,lllon,lrlon))
    lat_center = np.mean((ullat,urlat,lllat,lrlat))
    outline = shapely.geometry.Polygon([(ullon,ullat),(urlon,urlat),(lrlon,lrlat),(lllon,lllat),(ullon,ullat)])
    return lon_center,lat_center,outline

def deg2rad(deg):
    rad = deg*np.math.pi/180
    return rad

def great_circle_distance(lon1,lat1,lon2,lat2,R):
    lon1 = deg2rad(lon1)
    lat1 = deg2rad(lat1)
    lon2 = deg2rad(lon2)
    lat2 = deg2rad(lat2)
    DL = np.abs(lon2 - lon1)
    DP = np.abs(lat2 - lat1)
    dsigma = 2*np.math.asin( np.math.sqrt( np.math.sin(0.5*DP)**2 + np.math.cos(lat1)*np.math.cos(lat2)*np.math.sin(0.5*DL)**2))
    distance = R*dsigma
    return distance

def get_lonlat_polygon(polygon):
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    exterior_xy = np.asarray(polygon.exterior.xy)
    lon = np.append(lon,exterior_xy[0,:])
    lon = np.append(lon,np.nan)
    lat = np.append(lat,exterior_xy[1,:])
    lat = np.append(lat,np.nan)
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords.xy)
        lon = np.append(lon,interior_xy[0,:])
        lon = np.append(lon,np.nan)
        lat = np.append(lat,interior_xy[1,:])
        lat = np.append(lat,np.nan)
    return lon,lat

def get_lonlat_geometry(geom):
    '''
    Returns lon/lat of all exteriors and interiors of a Shapely geomery:
        -Polygon
        -MultiPolygon
        -GeometryCollection
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    if geom.geom_type == 'Polygon':
        lon_geom,lat_geom = get_lonlat_polygon(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'MultiPolygon':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'GeometryCollection':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    return lon,lat

def get_lonlat_gdf(gdf):
    '''
    Returns lon/lat of all exteriors and interiors of a GeoDataFrame
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    for geom in gdf.geometry:
        lon_geom,lat_geom = get_lonlat_geometry(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    return lon,lat

def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)



def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--strip',help='Path to input strip.',default=None)
    parser.add_argument('--input_file',help='Path to input file with list of strips.',default='/home/eheijkoop/INPUTS/WV_Training_INPUT.txt')
    parser.add_argument('--output_dir',help='Path to output directory.',default='/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/')
    parser.add_argument('--date_filter',help='Filter by date of input file.',default=False,action='store_true')
    args = parser.parse_args()
    strip = args.strip
    input_file = args.input_file
    date_filter = args.date_filter
    output_dir = args.output_dir
    if input_file == 'None':
        input_file = None
    
    '''
    To do:
    -Filter OSM by date, figure out all options for start_date
    '''

    tmp_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/Machine_Learning/WV_PanSharpened/tmp/'
    output_training_dir = f'{output_dir}Training_Data/'
    output_labels_dir = f'{output_dir}Labels/'
    check_dir_exists(output_training_dir)
    check_dir_exists(output_labels_dir)
    R_E = 6378137

    strips_already_done = np.sort(np.asarray(glob.glob(f'{output_training_dir}*.tif')))

    if strip is not None and input_file is not None:
        df_input = pd.read_csv(input_file,header=0,names=['strip','clip_lon_min','clip_lon_max','clip_lat_min','clip_lat_max'],dtype={'strip':str,'clip_lon_min':float,'clip_lon_max':float,'clip_lat_min':float,'clip_lat_max':float})
        strips_input = np.asarray(df_input.strip)
        print('Warning! Single strip and input file specified, doing both!')
        strips_input = np.append(strips_input,strip)
        strips_input = np.unique(strips_input)
        del strip
    elif strip is None and input_file is not None:
        df_input = pd.read_csv(input_file,header=0,names=['strip','clip_lon_min','clip_lon_max','clip_lat_min','clip_lat_max'],dtype={'strip':str,'clip_lon_min':float,'clip_lon_max':float,'clip_lat_min':float,'clip_lat_max':float})
        strips_input = np.asarray(df_input.strip)
    elif strip is not None and input_file is None:
        df_input = None
        strips_input = np.atleast_1d(strip)
    elif strip is None and input_file is None:
        print('Error! No input specified!')
        sys.exit()


    idx_already_done = np.any(np.asarray([[os.path.basename(s1).split('_2m')[0] in os.path.basename(s2).split('_pansharpened')[0] for s2 in strips_already_done] for s1 in strips_input]),axis=1)
    if np.any(idx_already_done) == True:
        print('Warning! Some strips are already done!')
    strips_input = strips_input[~idx_already_done]
    if df_input is not None:
        df_input = df_input[~idx_already_done].reset_index(drop=True)

    for i in range(len(strips_input)):
        strip = os.path.abspath(strips_input[i])
        clip_flag = np.any(~np.isnan([df_input.clip_lon_min[i],df_input.clip_lon_max[i],df_input.clip_lat_min[i],df_input.clip_lat_max[i]]))
        if clip_flag == True:
            clip_lon_min = df_input.clip_lon_min[i]
            clip_lon_max = df_input.clip_lon_max[i]
            clip_lat_min = df_input.clip_lat_min[i]
            clip_lat_max = df_input.clip_lat_max[i]
            clip_lon_center = (clip_lon_min + clip_lon_max)/2
            clip_lat_center = (clip_lat_min + clip_lat_max)/2
            outline_clip = shapely.geometry.Polygon([(clip_lon_min,clip_lat_max),(clip_lon_max,clip_lat_max),(clip_lon_max,clip_lat_min),(clip_lon_min,clip_lat_min),(clip_lon_min,clip_lat_max)])
            gdf_clip = gpd.GeoDataFrame(geometry=[outline_clip],crs='EPSG:4326')
            clip_shp = f'{tmp_dir}/{os.path.basename(strip).split("_2m")[0]}_clip.shp'
            gdf_clip.to_file(clip_shp)
        sensor_name = strip.split('/')[-1][0:4]
        if sensor_name == 'WV01':
            print('WorldView 1 is panchromatic only, skipping!')
            continue
        strip_name = strip.split('/')[-1]
        strip_date = strip_name[5:13]
        strip_datetime = datetime.datetime.strptime(strip_date,'%Y%m%d')
        strip_epsg = osr.SpatialReference(wkt=gdal.Open(strip,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
        location_name = strip.split('/')[-4]
        if location_name[0:3] == 'UTM':
            location_name = strip.split('/')[-5]
        print(f'Working on {strip_name} in {location_name}.')
        lon_loc,lat_loc = geocode_location(location_name)

        print('Unzipping Archive...')
        unzipped_dir,zip_ID = unzip_strip_archive(strip,tmp_dir)
        if unzipped_dir is None:
            print('No archives found! Skipping!')
            continue
        if zip_ID == 'single':
            xml_files = glob.glob(f'{unzipped_dir}/*/*/*PAN/*.XML')
        elif zip_ID == 'multiple':
            xml_files = glob.glob(f'{unzipped_dir}/*/*P1BS*/*.XML')
        elif zip_ID == 'single_unzipped':
            xml_files = glob.glob(f'{unzipped_dir}/*.xml')
        xml_files.sort()
        d_min = 1e20
        for xml in xml_files:
            lon_center_xml,lat_center_xml,outline_xml = get_centroid_xml_file(xml)
            if clip_flag == True:
                d_xml = great_circle_distance(lon_center_xml,lat_center_xml,clip_lon_center,clip_lat_center,R_E)
            else:
                d_xml = great_circle_distance(lon_center_xml,lat_center_xml,lon_loc,lat_loc,R_E)
            d_min = np.min((d_min,d_xml))
            if d_xml == d_min:
                xml_select = xml
                lon_center_xml_select,lat_center_xml_select,outline_xml_select = lon_center_xml,lat_center_xml,outline_xml

        pan_file = xml_select.replace('.XML','.NTF').replace('.xml','.ntf')
        mul_file = pan_file.replace('PAN','MUL').replace('-P1BS','-M1BS')
        pan_orthorectified_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_pan_orthorectified.tif'
        mul_orthorectified_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_mul_orthorectified.tif'
        tmp_pansharpened_orthorectified_full_res_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_pansharpened_orthorectified_full_res.tif'
        pansharpened_orthorectified_file = f'{output_training_dir}{os.path.basename(strip).split("_2m")[0]}_pansharpened_orthorectified.tif'
        tmp_binary_pansharpened_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_pansharpened_orthorectified_binary.tif'
        tmp_pansharpened_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_pansharpened_orthorectified.tif'
        tmp_label_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_label.tif'
        final_label_file = f'{output_labels_dir}{os.path.basename(strip).split("_2m")[0]}_label.tif'
        if np.logical_or(not os.path.isfile(pan_file),not os.path.isfile(mul_file)):
            print('ERROR! Cannot find either the specified PAN or MUL files!')
            print(f'PAN: {pan_file}')
            print(f'MUL: {mul_file}')
            shutil.rmtree(unzipped_dir)
            continue

        lon_min_strip,lon_max_strip,lat_min_strip,lat_max_strip = get_raster_extents(strip)
        outline_strip = shapely.geometry.Polygon([(lon_min_strip,lat_max_strip),(lon_max_strip,lat_max_strip),(lon_max_strip,lat_min_strip),(lon_min_strip,lat_min_strip),(lon_min_strip,lat_max_strip)])
        intersection_strip_imd = outline_xml_select.intersection(outline_strip)
        lon_outline_strip,lat_outline_strip = get_lonlat_geometry(outline_strip)
        lon_outline_imd,lat_outline_imd = get_lonlat_geometry(outline_xml_select)
        lon_intersection_strip_imd,lat_intersection_strip_imd = get_lonlat_geometry(intersection_strip_imd)
        lon_min_osm = np.nanmin(lon_intersection_strip_imd)
        lon_max_osm = np.nanmax(lon_intersection_strip_imd)
        lat_min_osm = np.nanmin(lat_intersection_strip_imd)
        lat_max_osm = np.nanmax(lat_intersection_strip_imd)
        osm_shp_file = f'{unzipped_dir}{os.path.basename(strip).split("_2m")[0]}_OSM_buildings_{strip_epsg}.shp'
        print('Downloading OSM Buildings...')
        osm_data = get_osm_buildings(lon_min_osm,lon_max_osm,lat_min_osm,lat_max_osm)
        gdf_osm = overpy_to_gdf(osm_data)
        gdf_osm = gdf_osm.buffer(0)

        orthorectify_pan_command = f'gdalwarp -q -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -co "TILED=YES" -t_srs EPSG:{strip_epsg} -r cubic -et 0.01 -rpc -to "RPC_DEM={strip}" {pan_file} {pan_orthorectified_file}'
        orthorectify_mul_command = f'gdalwarp -q -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -co "TILED=YES" -t_srs EPSG:{strip_epsg} -r cubic -et 0.01 -rpc -to "RPC_DEM={strip}" {mul_file} {mul_orthorectified_file}'
        pansharpen_command = f'gdal_pansharpen.py -q -b 5 -b 3 -b 2 -co compress=lzw -co bigtiff=if_safer -bitdepth 11 {pan_orthorectified_file} {mul_orthorectified_file} {tmp_pansharpened_orthorectified_full_res_file}'
        print('Orthorectifying Panchromatic Image...')
        subprocess.run(orthorectify_pan_command,shell=True)
        print('Orthorectifying Multispectral Image...')
        subprocess.run(orthorectify_mul_command,shell=True)
        print('Pansharpening Orthorectified Images...')
        subprocess.run(pansharpen_command,shell=True)
        print('Resampling to 0.5 m...')
        resample_command = f'gdalwarp -q -tr 0.5 0.5 -r cubic -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" {tmp_pansharpened_orthorectified_full_res_file} {pansharpened_orthorectified_file}'
        subprocess.run(resample_command,shell=True)
        subprocess.run(f'rm {tmp_pansharpened_orthorectified_full_res_file}',shell=True)
        binarize_command = f'gdal_calc.py --quiet --overwrite -A {pansharpened_orthorectified_file} --A_band=1 -B {pansharpened_orthorectified_file} --B_band=2 -C {pansharpened_orthorectified_file} --C_band=3 --outfile={tmp_binary_pansharpened_file} --calc="numpy.any((A>0,B>0,C>0))" --NoDataValue=-9999'
        subprocess.run(binarize_command,shell=True)
        gdf_osm = gdf_osm.to_crs(f'EPSG:{strip_epsg}')

        #need to incorporate date_filter here!
        if date_filter == True:
            #do something with the gdf to remove dates outside of the date range
            print('')
        
        gdf_osm.to_file(osm_shp_file)
        print('Creating Label Image...')
        clip_label_command = f'gdalwarp -q -s_srs EPSG:{strip_epsg} -t_srs EPSG:{strip_epsg} -cutline {osm_shp_file} -crop_to_cutline {tmp_binary_pansharpened_file} {tmp_label_file}'
        subprocess.run(clip_label_command,shell=True)
        if clip_flag == True:
            tmp_pansharpened_before_clipping = f'{tmp_dir}tmp_pansharpened_file.tif'
            tmp_label_before_clipping = f'{tmp_dir}tmp_label_file.tif'
            move_pansharpened_for_clipping_command = f'mv {pansharpened_orthorectified_file} {tmp_pansharpened_before_clipping}'
            move_label_for_clipping_command = f'mv {tmp_label_file} {tmp_label_before_clipping}'
            subprocess.run(move_pansharpened_for_clipping_command,shell=True)
            subprocess.run(move_label_for_clipping_command,shell=True)
            print('Clipping Pansharpened Image...')
            clip_pansharpened_command = f'gdalwarp -q -s_srs EPSG:{strip_epsg} -t_srs EPSG:{strip_epsg} -cutline {clip_shp} -crop_to_cutline {tmp_pansharpened_before_clipping} {pansharpened_orthorectified_file}'
            subprocess.run(clip_pansharpened_command,shell=True)
            print('Clipping Label Image...')
            clip_label_command = f'gdalwarp -q -s_srs EPSG:{strip_epsg} -t_srs EPSG:{strip_epsg} -cutline {clip_shp} -crop_to_cutline {tmp_label_before_clipping} {final_label_file}'
            subprocess.run(clip_label_command,shell=True)
            subprocess.run(f'rm {clip_shp.replace(".shp",".*")}',shell=True)
            subprocess.run(f'rm {tmp_label_before_clipping}',shell=True)
            subprocess.run(f'rm {tmp_pansharpened_before_clipping}',shell=True)
        else:
            move_label_command = f'mv {tmp_label_file} {final_label_file}'
            subprocess.run(move_label_command,shell=True)

        move_pansharpened_command = f'mv {pansharpened_orthorectified_file} {tmp_pansharpened_file}'
        move_label_command = f'mv {final_label_file} {tmp_label_file}'
        compress_pansharpened_command = f'gdal_translate -q -co compress=lzw -co bigtiff=if_safer {tmp_pansharpened_file} {pansharpened_orthorectified_file}'
        compress_label_command = f'gdal_translate -q -co compress=lzw -co bigtiff=if_safer {tmp_label_file} {final_label_file}'
        subprocess.run(move_pansharpened_command,shell=True)
        subprocess.run(move_label_command,shell=True)
        subprocess.run(compress_pansharpened_command,shell=True)
        subprocess.run(compress_label_command,shell=True)
        subprocess.run(f'rm {tmp_pansharpened_file}',shell=True)
        subprocess.run(f'rm {tmp_label_file}',shell=True)

        shutil.rmtree(unzipped_dir)
        print('')
if __name__ == '__main__':
    main()
