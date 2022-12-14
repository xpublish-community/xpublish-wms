import numpy as np


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