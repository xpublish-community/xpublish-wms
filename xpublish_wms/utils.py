import numpy as np
from pyproj import Transformer


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


def lnglat_to_cartesian(longitude, latitude):
    '''
    Converts latitude and longitude to cartesian coordinates
    '''
    lng_rad = np.deg2rad(longitude)
    lat_rad = np.deg2rad(latitude)

    R = 6371 
    x = R * np.cos(lat_rad) * np.cos(lng_rad)
    y = R * np.cos(lat_rad) * np.sin(lng_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack((x, y, z))


to_lnglat = Transformer.from_crs(3857, 4326, always_xy=True)