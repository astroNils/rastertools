import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import geopandas as gpd

from itertools import product
from affine import Affine
from pathlib import Path
from rasterio import features
from rasterio.mask import mask as rio_mask
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from tqdm import tqdm
import rastertools_BOULDERING.metadata as raster_metadata
import rastertools_BOULDERING.convert as raster_convert
import rastertools_BOULDERING.misc as raster_misc

def read(in_raster, bands=None, bbox=None, as_image=False):
    """
    Read a raster file into a NumPy array.

    Parameters
    ----------
    in_raster : str or Path
        Path to the input raster file.
    bands : int or list of int, optional
        Band(s) to read (band count starts from 1). If None, all bands are read.
    bbox : list or rasterio.windows.Window, optional
        Either a bounding box [xmin, ymin, xmax, ymax] or a rasterio Window object
        defining the area to read.
    as_image : bool, optional
        If True, reshapes the array to (rows, cols, bands) format.
        If False (default), keeps (bands, rows, cols) format.

    Returns
    -------
    ndarray
        The raster data as a NumPy array.
    """

    with rio.open(in_raster) as rio_dataset:
        if bands:
            if type(bands) == int:
                bands = [bands]
            else:
                None
        else:
            bands = list(np.arange(rio_dataset.count) + 1)

        if bbox:
            # if bbox is provided as indexes
            if type(bbox) == rio.windows.Window:
                array = rio_dataset.read(bands, window=bbox)

            else:
                # if a bbox with coordinates are specified, convert to pixel
                win = rio.windows.from_bounds(*bbox, rio_dataset.transform)

                # let's round to the closest pixel
                new_col_off = np.int32(np.round(win.col_off))
                new_row_off = np.int32(np.round(win.row_off))
                new_width = np.int32(np.round(win.width))
                new_height = np.int32(np.round(win.height))

                new_win = rio.windows.Window(new_col_off, new_row_off,
                                             new_width, new_height)

                array = rio_dataset.read(bands, window=new_win)
        else:
            # if none of the above, just read the whole array
            array = rio_dataset.read(bands)

    # reshape to (rows, columns, bands) from (bands, rows, columns)
    if as_image:
        return reshape_as_image(array)
    else:
        return (array)

def save(fpath, arr, profile, is_image=True):
    """
    Save a NumPy array as a raster file.

    Parameters
    ----------
    fpath : str or Path
        Output file path.
    arr : ndarray
        Array to save as raster.
    profile : dict
        Rasterio profile containing metadata.
    is_image : bool, optional
        If True (default), assumes array is in (rows, cols, bands) format and 
        reshapes to (bands, rows, cols) before saving.
        If False, assumes array is already in (bands, rows, cols) format.
    """
    with rio.open(fpath, "w", **profile) as dst:
        if is_image:
            dst.write(reshape_as_raster(arr))
        else:
            dst.write(arr)

def clip_from_bbox(in_raster, bbox, out_raster):
    """
    Clip a raster using a bounding box.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    bbox : list or rasterio.windows.Window
        Either a bounding box [xmin, ymin, xmax, ymax] or a rasterio Window object
        defining the clip area.
    out_raster : str or Path
        Path to output clipped raster.
    """
    
    with rio.open(in_raster) as rio_dataset:
        in_meta = rio_dataset.meta
        out_meta = in_meta.copy()
        
        # can either get a rio.windows.Windows or list [xmin, ymin, xmax, ymax]
        if type(bbox) == rio.windows.Window:
            array = rio_dataset.read(window=bbox) # read array for window
            win_transform = rio_dataset.window_transform(bbox) # get new transform
        else:
            # get window (can be done with rio_dataset.index and indexes too)
            win = rio.windows.from_bounds(*bbox, rio_dataset.transform)

            # let's round to the closest pixel
            new_col_off = np.int32(np.round(win.col_off))
            new_row_off = np.int32(np.round(win.row_off))
            new_width = np.int32(np.round(win.width))
            new_height = np.int32(np.round(win.height))

            new_win = rio.windows.Window(new_col_off, new_row_off, new_width, new_height)

            # read array for window
            array = rio_dataset.read(window=new_win)
            
            # get new transform
            win_transform = rio_dataset.window_transform(new_win)
        
    # shape of array
    dst_channel, dst_height, dst_width = np.shape(array)

    # update meta information
    if in_meta['driver'] == 'VRT':
        try:
            out_meta = raster_misc.removekey(out_meta, "blockysize")
            out_meta = raster_misc.removekey(out_meta, "blockxsize")
        except:
            None
        out_meta.update({"tiled": False})
    else:
        None

    out_meta.update({"driver": "GTiff",
             "height": dst_height,
             "width": dst_width,
             "transform": win_transform})

    with rio.open(out_raster, "w", **out_meta) as dst:
        dst.write(array)


def clip_from_polygon(in_raster, in_polygon, out_raster):
    """
    Clip a raster using a polygon shapefile.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    in_polygon : str or Path
        Path to input polygon shapefile.
    out_raster : str or Path
        Path to output clipped raster.
    """

    gdf = gpd.read_file(in_polygon)
    with rio.open(in_raster) as rio_dataset:
        in_meta = rio_dataset.meta
        out_meta = in_meta.copy()
        shapes = [row["geometry"] for i, row in gdf.iterrows()]
        # clipping of raster
        out_array, out_transform = rio_mask(rio_dataset, shapes, all_touched=False, crop=True)
        out_meta.update({"driver": "GTiff",
                         "height": out_array.shape[1],
                         "width": out_array.shape[2],
                         "transform": out_transform})
    save(out_raster, out_array, out_meta, is_image=False)

def projection(in_raster, dst_crs, out_raster):
    """
    Reproject a raster to a new coordinate reference system.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    dst_crs : str or CRS
        Target coordinate reference system.
    out_raster : str or Path
        Path to output reprojected raster.

    Notes
    -----
    This function is currently not working properly.
    """

    with rio.open(in_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        out_meta = src.meta.copy()
        out_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(out_raster, 'w', **out_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.cubic)

def resample(in_raster, out_resolution, out_raster):
    """
    Resample a raster to a new resolution using cubic interpolation.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    out_resolution : float
        Target resolution in units of the raster's CRS.
    out_raster : str or Path
        Path to output resampled raster.
    """
    in_raster = Path(in_raster)

    with rio.open(in_raster) as src:
        array = src.read()
        in_meta = src.profile
        out_meta = in_meta.copy()
        bounds = src.bounds # might have to calculate bounds based on 60 m?
        original_resolution = in_meta["transform"][0]
        resampling_factor = original_resolution / out_resolution

        transform, width, height = rio.warp.calculate_default_transform(
            in_meta["crs"],
            in_meta["crs"],
            int(in_meta["width"] * resampling_factor),
            int(in_meta["height"] * resampling_factor),
            *bounds
        )

        bands = array.shape[0]
        resampled_array = np.zeros((bands, height, width), dtype=in_meta['dtype'])

        rio.warp.reproject(
            source=array,
            destination=resampled_array,
            rio_dataset_transform=in_meta["transform"],
            rio_dataset_crs=in_meta["crs"],
            src_transform=in_meta["transform"],
            dst_transform=transform,
            src_crs=in_meta["crs"],
            dst_crs=in_meta["crs"],
            resampling=Resampling.cubic,
        )

        out_meta.update({"transform": transform, "width": width, "height": height})
        save(out_raster, resampled_array, out_meta, is_image=False)

def polygonize(in_raster, array, mask_array, out_shapefile):
    """
    Convert raster data to vector polygons.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster file used for georeferencing.
    array : ndarray
        Array to be polygonized.
    mask_array : ndarray of bool
        Boolean mask array indicating which pixels to polygonize.
    out_shapefile : str or Path
        Path to output shapefile.

    Examples
    --------
    >>> array = read(in_raster, as_image=True).squeeze()
    >>> mask_array = array > 200  # brightest region in the picture
    >>> array = (mask + 0.0).astype('uint8')
    >>> polygonize(in_raster, array, mask_array, out_shapefile="test.shp")
    """

    in_meta = raster_metadata.get_profile(in_raster)
    geoms = []
    results = ({'properties': {'raster_val': v}, 'geometry': s}
               for j, (s, v) in enumerate(
        features.shapes(array, mask=mask_array, transform=in_meta["transform"])))
    geoms.append(list(results))

    gdf = gpd.GeoDataFrame.from_features(geoms[0], crs=in_meta["crs"])
    gdf.to_file(out_shapefile)

def mask(in_raster, array, out_raster, is_image=True):
    """
    Apply a mask to a raster and save the result.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    array : ndarray
        Mask array to be applied.
    out_raster : str or Path
        Path to output masked raster.
    is_image : bool, optional
        If True (default), assumes array is in (rows, cols, bands) format.
        If False, assumes array is in (bands, rows, cols) format.

    Examples
    --------
    >>> array = raster.read(in_raster, as_image=True).squeeze()
    >>> mask_array = array < 50
    >>> new_array = (mask + 0.0).astype('uint8')
    >>> mask(in_raster, np.expand_dims(new_array, 2), out_raster="masked.tif")
    """
    in_meta = raster_metadata.get_profile(in_raster)
    out_meta = in_meta.copy()
    save(out_raster, array, out_meta, is_image=is_image)

def true_footprint(in_raster, out_shapefile):
    """
    Extract the actual footprint of a raster by excluding nodata regions.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    out_shapefile : str or Path
        Path to output shapefile containing the footprint polygon.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the footprint polygon.

    Notes
    -----
    Assumes nodata values are equal to 0.
    Returns the convex hull of the non-zero regions.
    """

    in_raster = Path(in_raster)
    array = read(in_raster, as_image=True).squeeze()
    mask_array = array != 0
    array = mask_array + 0
    array = array.astype('uint8')
    polygonize(in_raster, array, mask_array, out_shapefile)

    # For cases where you have blobs of NaN within the raster.
    in_meta = raster_metadata.get_profile(in_raster)
    gdf = gpd.read_file(out_shapefile)
    gdf = gpd.GeoDataFrame(geometry=[gdf.unary_union.convex_hull], crs=in_meta["crs"].to_wkt())
    gdf.to_file(out_shapefile)
    return gdf


def footprint(in_raster, out_shapefile):
    """
    Extract the rectangular footprint of a raster including nodata regions.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    out_shapefile : str or Path
        Path to output shapefile containing the footprint polygon.
    """
    in_raster = Path(in_raster)
    in_crs = raster_metadata.get_crs(in_raster).to_wkt()
    bbox = raster_metadata.get_bounds(in_raster)
    bbox = box(*bbox)
    gs = gpd.GeoSeries(bbox, crs=in_crs)
    gs.to_file(out_shapefile)

def pad(in_raster, padding_height, padding_width, out_raster):
    """
    Pad a raster with nodata values.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    padding_height : tuple of int
        Padding in pixels for height (padding_below, padding_above).
    padding_width : tuple of int
        Padding in pixels for width (padding_left, padding_right).
    out_raster : str or Path
        Path to output padded raster.

    Notes
    -----
    The nodata value is extracted from the input raster metadata.
    """

    in_raster = Path(in_raster)
    array = read(in_raster, as_image=True).squeeze()
    in_meta = raster_metadata.get_profile(in_raster)
    out_meta = in_meta.copy()
    in_res = raster_metadata.get_resolution(in_raster)[0]
    in_bbox = raster_metadata.get_bounds(in_raster)
    padded_array = np.pad(array, (padding_height, padding_width), 'constant', constant_values=in_meta["nodata"])
    padded_array = np.expand_dims(padded_array, axis=2)

    out_bbox = [in_bbox[0] - (in_res * padding_width[0]),
                in_bbox[1] - (in_res * padding_height[0]),
                in_bbox[2] + (in_res * padding_width[1]),
                in_bbox[3] + (in_res * padding_height[1])]

    out_meta["width"] = padded_array.shape[1]
    out_meta["height"] = padded_array.shape[0]
    out_meta["transform"] = Affine(in_res, 0.0, out_bbox[0], 0.0, -in_res, out_bbox[3])

    save(out_raster, padded_array, out_meta, is_image=True)

def shift(in_raster, x_shift, y_shift, out_raster):
    """
    Shift a raster by specified x and y offsets.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    x_shift : float
        Shift in x direction (in meters).
    y_shift : float
        Shift in y direction (in meters).
    out_raster : str or Path
        Path to output shifted raster.

    Notes
    -----
    Future improvements may include support for rotation and scaling.
    """

    in_raster = Path(in_raster)
    in_meta = raster_metadata.get_profile(in_raster)
    out_meta = in_meta.copy()

    in_bbox = raster_metadata.get_bounds(in_raster)
    out_bbox = [in_bbox[0] + x_shift,
                in_bbox[1] + y_shift,
                in_bbox[2] + x_shift,
                in_bbox[3] + y_shift]


    in_transform = in_meta["transform"]
    out_transform = Affine(in_transform[0], in_transform[1], out_bbox[0],
                           in_transform[3], in_transform[4], out_bbox[3])

    out_meta.update({"transform": out_transform})

    array = read(in_raster, as_image=False)
    save(out_raster, array, out_meta, is_image=False)

def graticule(in_raster, block_width, block_height, out_shapefile, stride=(0, 0)):
    """
    Generate a grid of polygons covering the raster extent.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    block_width : int
        Width of each grid cell in pixels.
    block_height : int
        Height of each grid cell in pixels.
    out_shapefile : str or Path
        Path to output shapefile containing the grid polygons.
    stride : tuple of int, optional
        (stride_x, stride_y) spacing between grid cells, default (0, 0).

    Returns
    -------
    tuple
        (pandas.DataFrame, geopandas.GeoDataFrame) containing grid metadata and geometries.
    """
    in_raster = Path(in_raster)
    print("...Generate graticule for raster " + in_raster.name +
          " (" + str(block_width) + "x" + str(
        block_height) + " pixels, stride " +
          str(stride[0]) + "/" + str(stride[1]) + ")" + "...")

    global_graticule_name = Path(out_shapefile)
    global_graticule_name = global_graticule_name.absolute()
    pickle_name = global_graticule_name.with_name(
        global_graticule_name.stem + ".pkl")
    res = raster_metadata.get_resolution(in_raster)[0]

    (windows, transforms, bounds) = tile_windows(in_raster, block_width, block_height, stride)

    assert len(
        bounds) < 100000, "Number of tiles larger than 100,000. Please modify function generate_graticule_from_raster()."

    polygons = [shapely.geometry.box(l, b, r, t) for l, b, r, t in bounds]
    tile_id = [i for i in range(len(bounds))]
    image_id_png = [in_raster.stem + "_" + str(i).zfill(5) + "_image.png" for i
                    in range(len(bounds))]
    raster_name_abs = [in_raster.as_posix() for i in range(len(bounds))]
    raster_name_rel = [in_raster.name for i in range(len(bounds))]
    windows_px = [list(i.flatten()) for i in windows]
    transforms_p = [list(i)[:6] for i in transforms]
    product_id = [in_raster.stem for i in range(len(bounds))]
    crs = raster_metadata.get_crs(in_raster).wkt
    crs_l = [crs for i in range(len(bounds))]
    res_l = [res for i in range(len(bounds))]

    df = pd.DataFrame(list(zip(product_id, tile_id, image_id_png,
                               raster_name_abs, raster_name_rel, windows_px,
                               transforms_p, bounds, crs_l, res_l)),
                      columns=['image_id', 'tile_id', 'file_name',
                               'raster_ap', 'raster_rp', 'rwindows',
                               'transform', 'bbox_im', 'coord_sys', 'pix_res'])
    df.to_pickle(pickle_name)
    df_qgis = df[['image_id', 'tile_id', 'file_name']]

    gdf = gpd.GeoDataFrame(df_qgis, geometry=polygons)
    gdf = gdf.set_crs(crs)

    gdf.to_file(global_graticule_name)
    return (df, gdf)

def tile_windows(in_raster, block_width=512, block_height=512, stride=(0, 0)):
    """
    Generate windows for tiling a raster into blocks.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster.
    block_width : int, optional
        Width of each tile in pixels, default 512.
    block_height : int, optional
        Height of each tile in pixels, default 512.
    stride : tuple of int, optional
        (stride_x, stride_y) spacing between tiles, default (0, 0).

    Returns
    -------
    tuple
        (list of Window, list of Affine, list of bounds) containing tile windows,
        transforms, and bounds.

    Notes
    -----
    Block dimensions should ideally be multiples of 16 for optimal performance.
    """

    nwidth = raster_metadata.get_width(in_raster)
    nheight = raster_metadata.get_height(in_raster)
    stride_x = stride[0]
    stride_y = stride[1]

    offsets = product(range(stride_x, nwidth, block_width),
                      range(stride_y, nheight, block_height))

    tile_window = []
    tile_transform = []
    tile_bounds = []

    with rio.open(in_raster) as src:
        src_transform = src.transform

        # added rounding to avoid varying height, width of tiles
        # maybe redundant
        for col_off, row_off in offsets:
            window = rio.windows.Window(col_off=col_off,
                                        row_off=row_off,
                                        width=block_width,
                                        height=block_height)

            new_col_off = np.int32(np.round(window.col_off))
            new_row_off = np.int32(np.round(window.row_off))
            new_width = np.int32(np.round(window.width))
            new_height = np.int32(np.round(window.height))

            new_win = rio.windows.Window(new_col_off, new_row_off,
                                         new_width, new_height)

            win_transform = src.window_transform(new_win)
            tile_window.append(new_win)
            tile_transform.append(win_transform)

            tile_bounds.append(rio.windows.bounds(new_win, src_transform,
                                                  new_win.height,
                                                  new_win.width))

    return tile_window, tile_transform, tile_bounds
def tile(in_raster, in_pkl, block_width, block_height):
    """
    Create image tiles from a raster file based on tiling metadata.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster file.
    in_pkl : str or Path
        Path to pickle file containing tiling metadata (generated by graticule function).
    block_width : int
        Width of each tile in pixels.
    block_height : int
        Height of each tile in pixels.

    Notes
    -----
    Creates both TIFF and PNG versions of each tile in an 'images' subdirectory.
    Handles edge cases by padding tiles that would extend beyond the raster bounds.
    """

    print("...Tiling original image into small image patches...")

    in_raster = Path(in_raster)
    image_directory = (in_raster.parent / "images")
    image_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(in_pkl)
    ntiles = df.shape[0]

    for index, row in tqdm(df.iterrows(), total=ntiles):
        src_profile = raster_metadata.get_profile(in_raster)
        win_profile = src_profile
        win_profile["width"] = block_width
        win_profile["height"] = block_height
        arr = read(in_raster=in_raster, bbox=rio.windows.Window(*row.rwindows))

        # edge cases (in the East, and South, the extent can be beigger than the actual raster)
        # read_raster will then return an array with not the dimension
        h, w = arr.squeeze().shape

        if (h, w) != (block_height, block_width):
            arr = np.pad(arr.squeeze(),
                         [(0, block_height - h), (0, block_width - w)],
                         mode='constant', constant_values=0)
            arr = np.expand_dims(arr, axis=0)

        filename_tif = image_directory / row.file_name.replace(".png", ".tif")
        filename_png = image_directory / row.file_name
        win_profile["transform"] = Affine(*row["transform"])

        # generate tif and pngs (1- and 3-bands)
        save(filename_tif, arr, win_profile, is_image=False)
        raster_convert.tiff_to_png(filename_tif, filename_png)

def tile_from_dataframe(dataframe, dataset_directory, block_width, block_height):
    """
    Create image tiles from multiple rasters based on DataFrame specifications.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing tiling metadata for multiple rasters.
        Must include columns: dataset, raster_ap, file_name, rwindows, transform.
    dataset_directory : str or Path
        Base directory where tiled images will be saved.
    block_width : int
        Width of each tile in pixels.
    block_height : int
        Height of each tile in pixels.

    Notes
    -----
    - Creates both TIFF and PNG versions of each tile.
    - Organizes tiles into subdirectories by dataset.
    - Skips tiling if the number of existing tiles matches the DataFrame specifications.
    - Handles edge cases by padding tiles that would extend beyond the raster bounds.
    """

    print("...Tiling original image into small image patches...")

    dataset_directory = Path(dataset_directory)
    raster_misc.folder_structure(dataframe, dataset_directory)  # ensure folders are created
    datasets = dataframe.dataset.unique()

    nimages = 0
    for d in datasets:
        image_directory = (dataset_directory / d / "images")
        n = len(list(image_directory.glob("*.tif")))
        nimages = nimages + n

    ntiles = dataframe.shape[0]

    if nimages == ntiles:
        print("Number of tiles == Number of tiles in specified folder(s). No tiling required.")
    # if for some reasons they don't match, it just need to be re-tiled
    # we delete the image directory(ies) just to start from a clean folder
    else:
        for d in datasets:
            image_directory = (dataset_directory / d / "images")
            raster_misc.rm_tree(image_directory)

        # re-creating folder structure
        raster_misc.folder_structure(dataframe, dataset_directory)

        for index, row in tqdm(dataframe.iterrows(), total=ntiles):

            # this is only useful within the loop if generating tiling on multiple images
            in_raster = row.raster_ap
            src_profile = raster_metadata.get_profile(in_raster)
            win_profile = src_profile
            win_profile["width"] = block_width
            win_profile["height"] = block_height

            arr = read(in_raster=in_raster, bbox=rio.windows.Window(*row.rwindows))

            # edge cases (in the East, and South, the extent can be bigger than the actual raster)
            # read will then return an array with not the dimension
            h, w = arr.squeeze().shape

            if (h, w) != (block_height, block_width):
                arr = np.pad(arr.squeeze(),
                             [(0, block_height - h), (0, block_width - w)],
                             mode='constant', constant_values=0)
                arr = np.expand_dims(arr, axis=0)

            filename_tif = (dataset_directory / row.dataset / "images" / row.file_name.replace(".png", ".tif"))
            filename_png1 = (dataset_directory / row.dataset / "images" / row.file_name)
            win_profile["transform"] = Affine(*row["transform"])

            # generate tif and pngs (1- and 3-bands)
            save(filename_tif, arr, win_profile, is_image=False)
            raster_convert.tiff_to_png(filename_tif, filename_png1)
