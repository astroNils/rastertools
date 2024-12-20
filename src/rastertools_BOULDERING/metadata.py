import rasterio as rio

def get_pix_coordinates(in_raster, bbox):
    """
    Get pixel coordinates from geographic coordinates.

    Parameters
    ----------
    in_raster : str or Path
        Path to the input raster file.
    bbox : list of float
        Bounding box coordinates [xmin, ymin, xmax, ymax] in the raster's CRS.

    Returns
    -------
    tuple
        Pixel coordinates as (col_ul, row_ul, width, height) where:
        - col_ul, row_ul: Upper-left pixel coordinates
        - width: Number of pixels in x direction
        - height: Number of pixels in y direction
    """
    with rio.open(in_raster) as rio_dataset:
        row_ul, col_ul = rio_dataset.index(bbox[0], bbox[3])
        row_lr, col_lr = rio_dataset.index(bbox[2], bbox[1])
        extent = (col_ul, row_ul, col_lr - col_ul, row_lr - row_ul)
    return extent

def get_crs(in_raster):
    with rio.open(in_raster) as rio_dataset:
        crs = rio_dataset.crs
    return crs

def get_resolution(in_raster):
    with rio.open(in_raster) as rio_dataset:
        res = rio_dataset.res
    return res

def get_dtypes(in_raster):
    with rio.open(in_raster) as rio_dataset:
        dtypes = rio_dataset.dtypes
    return dtypes

def get_height(in_raster):
    with rio.open(in_raster) as rio_dataset:
        height = rio_dataset.height
    return height

def get_width(in_raster):
    with rio.open(in_raster) as rio_dataset:
        width = rio_dataset.width
    return width

def get_bounds(in_raster):
    with rio.open(in_raster) as rio_dataset:
        bbox = rio_dataset.bounds
    return list(bbox)

def get_shape(in_raster):
    with rio.open(in_raster) as rio_dataset:
        shape = rio_dataset.shape
    return shape

def get_count(in_raster):
    with rio.open(in_raster) as rio_dataset:
        count = rio_dataset.count
    return count

def get_profile(in_raster):
    with rio.open(in_raster) as rio_dataset:
        profile = rio_dataset.profile.copy()
    return profile