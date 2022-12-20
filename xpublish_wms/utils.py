import xarray as xr
import numpy as np
from pyproj import Transformer
from loguru import logger


def lower_case_keys(d: dict) -> dict:
    return dict((k.lower(), v) for k, v in d.items())

def format_timestamp(value):
    return value.dt.strftime(date_format='%Y-%m-%dT%H:%M:%SZ').values


def strip_float(value):
    return float(value.values)


def round_float_values(v: list) -> list:
    if not isinstance(v, list):
        return round(v, 5)
    return [round(x, 5) for x in v]


def speed_and_dir_for_uv(u, v):
    '''
    Given u and v values or arrays, calculate speed and direction transformations
    '''
    speed = np.sqrt(u**2 + v**2)

    dir_trig_to = np.arctan2(u/speed, v/speed)
    dir_trig_deg = dir_trig_to * 180/np.pi 
    dir = (dir_trig_deg) % 360

    return [speed, dir]


def ensure_crs(ds: xr.Dataset | xr.DataArray, default_crs: str = "EPSG:4326") -> xr.Dataset | xr.DataArray:
    """
    Ensure our dataset has a CRS
    :param ds:
    :param default_crs:
    :return:
    """
    #logger.debug(f"CRS found in dataset : {ds.rio.crs}")
    if not ds.rio.crs:
        logger.debug(f"Settings default CRS : {default_crs}")
        ds.rio.write_crs(default_crs, inplace=True)
    return ds


to_lnglat = Transformer.from_crs(3857, 4326, always_xy=True)