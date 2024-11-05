from pathlib import Path
from PIL import Image
from pyproj import Transformer
import rasterio as rio
import numpy as np
import rastertools_BOULDERING.raster as raster
import rastertools_BOULDERING.metadata as raster_metadata

def normalize_uint8(in_raster, out_raster):
    """
    Perform Min-Max Normalization and convert to 8-bit unsigned integer format.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster file.
    out_raster : str or Path
        Path to output raster file.

    Notes
    -----
    This is equivalent to gdal_translate -ot Byte -scale -a_nodata 0,
    but provides better handling of zero no-data values.
    For float32 inputs, values less than 0 are set to 0 before normalization.
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
    Convert RGB or RGBA raster to single-band grayscale.

    Parameters
    ----------
    in_raster : str or Path
        Path to input RGB(A) raster file.
    out_raster : str or Path, optional
        Path to output grayscale raster file. If None, appends '_grayscale'
        to input filename.

    Notes
    -----
    Uses PIL's "L" mode conversion which applies the formula:
    L = 0.299R + 0.587G + 0.114B
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
    Convert all PNG images in a folder to fake RGB PNG images.

    Parameters
    ----------
    folder : str or Path
        Path to directory containing PNG images to convert.

    Notes
    -----
    Creates new files with '_fakergb' suffix in the same directory.
    """
    folder = Path(folder)
    for in_raster in folder.glob('*.png'):
        fake_RGB(in_raster)

def tiff_to_png_batch(folder, is_hirise=False):
    """
    Convert all TIFF images in a folder to PNG format.

    Parameters
    ----------
    folder : str or Path
        Path to directory containing TIFF images.
    is_hirise : bool, optional
        If True, applies HiRISE-specific scaling. Default is False.
    """
    folder = Path(folder)
    for in_raster in folder.glob('*.tif'):
        tiff_to_png(in_raster, is_hirise)

def tiff_to_png(in_raster, out_png=False, is_hirise=False):
    """
    Convert a TIFF image to PNG format.

    Parameters
    ----------
    in_raster : str or Path
        Path to input TIFF file.
    out_png : str or Path, optional
        Path to output PNG file. If False, creates file with same name
        but .png extension.
    is_hirise : bool, optional
        If True, applies HiRISE-specific scaling (255/1023). Default is False.
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
    Convert single-band raster to fake RGB by duplicating the band.

    Parameters
    ----------
    in_raster : str or Path
        Path to input single-band raster.
    out_raster : str or Path, optional
        Path to output RGB raster. If None, appends '_fakergb' to input filename.
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
    Convert pixel coordinates to world coordinates.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster file.
    row : int
        Row coordinate (pixel).
    col : int
        Column coordinate (pixel).
    dst_crs : str, optional
        Target coordinate reference system (proj4 or WKT string).
        If None, returns coordinates in raster's CRS.

    Returns
    -------
    tuple
        (x, y) world coordinates in the target CRS.
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
    Convert world coordinates to pixel coordinates.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster file.
    x : float
        X coordinate in world space.
    y : float
        Y coordinate in world space.
    from_crs : str, optional
        Source coordinate reference system (proj4 or WKT string).
        If None, assumes coordinates are in raster's CRS.

    Returns
    -------
    tuple
        (row, col) pixel coordinates.
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