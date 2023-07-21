from pathlib import Path
from PIL import Image
from pyproj import Transformer
import rasterio as rio
import numpy as np
import rastertools_BOULDERING.raster as raster
import rastertools_BOULDERING.metadata as raster_metadata

def normalize_uint8(in_raster, out_raster):
    """
    Min-Max Normalization and conversion to byte. This is equivalent to
    gdal_translate -ot Byte -scale -a_nodata 0 src_dataset dst_dataset
    but it is better as you can drop the no_data equal to 0.
    :param in_raster: path to in_raster (str or Path)
    :param out_raster:path to out_raster (str or Path)
    :return:
    """
    array = raster.read(in_raster)
    out_meta = raster_metadata.get_profile(in_raster)
    if array.dtype == np.float32: # nan = -3.4028226550889045e+38
        array[array < 0] = 0.0
    array_norm = (array - array.min()) / (array.max() - array.min())
    array_uint8 = np.round(array_norm * 255, decimals=0).astype('uint8')

    out_meta.update({
             "count": 1,
             "dtype": "uint8",
             "nodata": 0})

    raster.save(out_raster, array_uint8, out_meta, False)


def rgb_to_grayscale(in_raster, out_raster):
    """
    Convert RGB or RGBA raster to grayscale (1-band).
    :param in_raster: path to in_raster (Path)
    :param out_raster: path to out_raster (Path)
    :return:
    """
    in_raster = Path(in_raster)
    array = Image.open(in_raster).convert("L")
    array = np.array(array)
    array = np.expand_dims(array, axis=0)

    if out_raster:
        None
    else:
        out_raster = in_raster.with_name(in_raster.stem + "_grayscale" + in_raster.suffix)

    out_meta = raster_metadata.get_profile(in_raster)
    out_meta.update({"count": 1})

    with rio.open(out_raster, "w", **out_meta) as dst:
        dst.write(array)

def rgb_fake_batch(folder):
    """
    Convert all png images in a folder to fake RGB png images.
    :param folder: path to folder (str or Path).
    :return:
    """
    folder = Path(folder)
    for in_raster in folder.glob('*.png'):
        fake_RGB(in_raster)

def tiff_to_png_batch(folder, is_hirise=False):
    """
    Convert all tif images in a folder to png images (1-band).
    :param folder: path to folder (str or Path).
    :param is_hirise: if True, convert to uint8.
    :return:
    """
    folder = Path(folder)
    for in_raster in folder.glob('*.tif'):
        tiff_to_png(in_raster, is_hirise)

def tiff_to_png(in_raster, out_png=False, is_hirise=False):
    """
    Convert tif image to png image (1-band).
    :param in_raster: path to raster (str or Path)
    :param out_png: path to output png (str or Path)
    :param is_hirise: if True, convert to uint8.
    :return:
    """
    in_raster = Path(in_raster)
    png = in_raster.with_name(in_raster.name.split(".tif")[0] + ".png")
    array = raster.read(in_raster, as_image=True)
    h, w, c = array.shape
    array = array.reshape((h,w))
    if is_hirise: # the constant value need to be changed...
        array = np.round(array * (255.0 / 1023.0)).astype('uint8')
    im = Image.fromarray(array)

    if out_png:
        None
    else:
        out_png = in_raster.with_name(in_raster.stem + "_fakergb" + in_raster.suffix)
    im.save(png)

def fake_RGB(in_raster, out_raster=None):
    """
    Convert 1-band raster to a fake 3-band raster (duplicates the same band).
    :param in_raster: path to in_raster.
    :param out_raster: path to out_raster.
    :return:
    """
    in_raster = Path(in_raster)
    array = Image.open(in_raster).convert("RGB")
    if out_raster:
        None
    else:
        out_raster = in_raster.with_name(in_raster.stem + "_fakergb" + in_raster.suffix)
    array.save(out_raster)

def pix2world(in_raster, row, col, dst_crs=None):
    """
    Convert pixel (row, col) to world coordinates (x,y). if to_crs
    is specified, x and y are projected to specified to the new coord sys.
    :param in_raster: path to in_raster.
    :param row: row (int).
    :param col: col (int).
    :param dst_crs: coordinate system (proj4 or wkt string).
    :return:
    """
    with rio.open(in_raster) as rio_dataset:
        crs_in_raster= rio_dataset.crs.to_wkt()
        x, y = rio_dataset.xy(row, col)
        if dst_crs:
            transformer = Transformer.from_crs(crs_in_raster, dst_crs)
            x_world, y_world = transformer.transform(x, y)
        else:
            x_world = x
            y_world = y
    return (x_world, y_world)


def world2pix(in_raster, x, y, from_crs=None):
    """
    Convert world coordinates (x,y) to pixel (row, col).
    :param in_raster:
    :param x: x coordinates in coord of sys of in_raster or in_crs.
    :param y: y coordinates in coord of sys of in_raster or in_crs.
    :param from_crs: if specified, the input x and y are in different crs than
    in_raster_crs (proj4 or wkt string).
    :return:
    """
    with rio.open(in_raster) as rio_dataset:
        crs_in_raster = rio_dataset.crs.to_wkt()
        if from_crs:
            transformer = Transformer.from_crs(from_crs, crs_in_raster)
            x_proj, y_proj = transformer.transform(x, y)
            row, col = rio_dataset.index(x_proj, y_proj)
        else:
            row, col = rio_dataset.index(x, y)
    return (row, col)