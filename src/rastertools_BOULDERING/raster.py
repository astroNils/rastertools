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
    """Read a raster. If bbox is specified, then only the specified bbox
    within the raster is read.

    Parameters
    ----------
    raster : path,
        absolute path to raster.
    bands : list of int or int. optional
        band(s) to read (remember band count starts from 1)
    bbox : list of int, optional
        bounding box of an area within the raster [xmin, ymin, xmax, ymax]
    as_image : boolean, optional
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
    with rio.open(fpath, "w", **profile) as dst:
        if is_image:
            dst.write(reshape_as_raster(arr))
        else:
            dst.write(arr)

def clip_from_bbox(in_raster, bbox, out_raster):
    
    """
    Clip a raster using the specified bounding box .
    
    :param raster: path to in_raster (str or Path)
    :param bbox: bounding box [xmin, ymin, xmax, ymax]
    :param out_raster: path to out_raster (str or Path)
    :returns:
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

    :param in_raster: path to in_raster (str or Path)
    :param in_polygon: path to polygon shapefile (str or Path)
    :param out_raster: path to out_raster (str or Path)
    :return:
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
    This function is currently not working...
    :param in_raster:
    :param dst_crs:
    :param out_raster:
    :return:
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
    Resample input raster based on the specified out_resolution.
    Cubic interpolation is used during the resampling step.
    :param in_raster:
    :param out_resolution:
    :param out_raster:
    :return:
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
    Polygonize raster based on array and mask_array.
    :param in_raster:
    :param array:
    :param mask_array:
    :param out_shapefile:
    :return:

    :example:
    array = read(in_raster, as_image=True).squeeze()
    mask_array = array > 200 # brightest region in the picture
    array = (mask + 0.0).astype('uint8')
    polygonize(in_raster, array, mask_array, out_shapefile="/home/nilscp/tmp/shp/test.shp")
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
    Mask in_raster (can be different mathematical operation)  and save it to
    a new raster.

    :param in_raster:
    :param array:
    :param out_raster:
    :param is_image:
    :return:

    :example:
    # create a binary raster for the darkest region in the picture
    array = raster.read(in_raster, as_image=True).squeeze()
    mask_array = array < 50
    new_array = (mask + 0.0).astype('uint8')
    mask(in_raster, np.expand_dims(new_array, 2), out_raster="dummy.tif")
    """
    in_meta = raster_metadata.get_profile(in_raster)
    out_meta = in_meta.copy()
    save(out_raster, array, out_meta, is_image=is_image)

def true_footprint(in_raster, out_shapefile):

    """
    Extract footprint of the raster (excluding the nodata).
    Nodata is assumed to be equal to 0.

    :param in_raster:
    :param out_shapefile:
    :return:
    """

    in_raster = Path(in_raster)
    array = read(in_raster, as_image=True).squeeze()
    mask_array = array != 0
    array = mask_array + 0
    array = array.astype('uint8')
    polygonize(in_raster, array, mask_array, out_shapefile)

def footprint(in_raster, out_shapefile):
    """
    Extract footprint of the raster (including the nodata).
    :param in_raster:
    :param out_shapefile:
    :return:
    """
    in_raster = Path(in_raster)
    in_crs = raster_metadata.get_crs(in_raster).to_wkt()
    bbox = raster_metadata.get_bounds(in_raster)
    bbox = box(*bbox)
    gs = gpd.GeoSeries(bbox, crs=in_crs)
    gs.to_file(out_shapefile)

def pad(in_raster, padding_height, padding_width, out_raster):
    """
    Pad raster with constant nodata value (extracted from in_raster metadata).
    The padding is specified in pixels.
    :param in_raster: path to in_raster (str or Path)
    :param padding_height: [padding_below, padding_above] in pixels.
    :param padding_width: [padding_left, padding_right] in pixels.
    :param out_raster:
    :return:
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
    Need to make it more general, where I can specify shift, rotation, scale.
    :param in_raster:
    :param x_shift: (in meters)
    :param y_shift: (in meters)
    :param out_raster:
    :return:
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
    replace generate_graticule_from_raster
    :param in_raster:
    :param block_width:
    :param block_height:
    :param geopackage:
    :return:
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
    Tile the rio_dataset raster (rasterio format) to a desired number of blocks
    having specified width and height.

    :param rio_dataset: rasterio raster dataset
    :param width: width of blocks (multiplier of 16 is advised, e.g., 512)
    :param height: height of blocks (multiplier of 16 is advised, e.g., 512)
    :returns: tile_window, tile_transform: list of rasterio windows and transforms
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
