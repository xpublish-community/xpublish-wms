import logging
import contextlib
from typing import Tuple, Union

import numpy as np
import xarray as xr
from pyproj import Transformer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xpublish_wms.grid import GridType

logger = logging.getLogger('uvicorn')


def lower_case_keys(d: dict) -> dict:
    return {k.lower(): v for k, v in d.items()}


def format_timestamp(value):
    return value.dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ").values


def strip_float(value):
    return float(value.values)


def round_float_values(v: list) -> list:
    if not isinstance(v, list):
        return round(v, 5)
    return [round(x, 5) for x in v]


def speed_and_dir_for_uv(u, v):
    """
    Given u and v values or arrays, calculate speed and direction transformations
    """
    speed = np.sqrt(u**2 + v**2)

    dir_trig_to = np.arctan2(u / speed, v / speed)
    dir_trig_deg = dir_trig_to * 180 / np.pi
    dir = (dir_trig_deg) % 360

    return [speed, dir]


def ensure_crs(
    ds: Union[xr.Dataset, xr.DataArray],
    default_crs: str = "EPSG:4326",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Ensure our dataset has a CRS
    :param ds:
    :param default_crs:
    :return:
    """
    # logger.debug(f"CRS found in dataset : {ds.rio.crs}")
    if not ds.rio.crs:
        logger.debug(f"Settings default CRS : {default_crs}")
        ds.rio.write_crs(default_crs, inplace=True)
    return ds


def lnglat_to_cartesian(longitude, latitude):
    """
    Converts latitude and longitude to cartesian coordinates
    """
    lng_rad = np.deg2rad(longitude)
    lat_rad = np.deg2rad(latitude)

    logger.warning(lng_rad)

    R = 6371
    x = R * np.cos(lat_rad) * np.cos(lng_rad)
    y = R * np.cos(lat_rad) * np.sin(lng_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack((x, y, z))


to_lnglat = Transformer.from_crs(3857, 4326, always_xy=True)


def ds_bbox(ds: xr.Dataset) -> Tuple[float, float, float, float]:
    """
    Return the bounding box of the dataset
    :param ds:
    :return:
    """
    grid_type = GridType.from_ds(ds)

    if grid_type == GridType.REGULAR:
        bbox = [
            ds.cf.coords["longitude"].min().values.item(),
            ds.cf.coords["latitude"].min().values.item(),
            ds.cf.coords["longitude"].max().values.item(),
            ds.cf.coords["latitude"].max().values.item(),
        ]
    elif grid_type == GridType.SGRID:
        topology = ds.cf["grid_topology"]
        lng_coord, lat_coord = topology.attrs["face_coordinates"].split(" ")
        bbox = [
            ds[lng_coord].min().values.item(),
            ds[lat_coord].min().values.item(),
            ds[lng_coord].max().values.item(),
            ds[lat_coord].max().values.item(),
        ]

    return bbox

@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)