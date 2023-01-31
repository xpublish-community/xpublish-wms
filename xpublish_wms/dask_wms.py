import xarray as xr
import numpy as np
from xpublish_wms.utils import strip_float, format_timestamp, speed_and_dir_for_uv, round_float_values
from fastapi import HTTPException


def init(client):
    client.upload_file('utils.py')
    client.upload_file('dask_wms.py')


def create_parameter_feature_data(parameter, ds: xr.Dataset, has_time_axis, t_axis, x_axis, y_axis, values=None, name=None, id=None):
    # TODO Use standard and long name?
    name = name if name is not None else ds[parameter].cf.attrs.get('long_name', ds[parameter].cf.attrs.get('name', parameter))
    id = id if id is not None else ds[parameter].cf.attrs.get('standard_name', parameter)

    info = {
        'type': 'Parameter',
        'description': {
            'en': name,
        },
        'observedProperty': {
            'label': {
                'en': name,
            },
            'id': id,
        }
    }

    axis_names = ['t', 'x', 'y'] if has_time_axis else ['x', 'y']
    shape = [len(t_axis), len(x_axis), len(y_axis)] if has_time_axis else [len(x_axis), len(y_axis)]
    values = values if values is not None else ds[parameter]
    values = round_float_values(values.squeeze().values.tolist())

    if isinstance(values, float):
        values = [values]

    range = {
        'type': 'NdArray',
        'dataType': 'float',
        # TODO: Some fields might not have a time field, and some might have an elevation field
        'axisNames': axis_names,
        'shape': shape,
        'values': [None if np.isnan(v) else v for v in values],
    }

    return (info, range)


# example WMS map call:
# http://localhost:8090/datasets/gfswave_global/wms/?service=WMS&version=1.3.0&request=GetMap&layers=swh&crs=EPSG:3857&bbox=-10018754.171394622,7514065.628545966,-7514065.628545966,10018754.17139462&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=2022-10-29T05:00:00Z
def get_feature_info(dataset: xr.Dataset, query: dict):
    """
    Return the WMS feature info for the dataset and given parameters
    """
    if not dataset.rio.crs:
        dataset = dataset.rio.write_crs(4326)

    if ':' in query['query_layers']:
        parameters = query['query_layers'].split(':')
    else:
        parameters = query['query_layers'].split(',')
    times = list(dict.fromkeys([t.replace('Z', '') for t in query['time'].split('/')]))
    crs = query.get('crs', None) or query.get('srs')
    bbox = [float(x) for x in query['bbox'].split(',')]
    width = int(query['width'])
    height = int(query['height'])
    x = int(query['x'])
    y = int(query['y'])
    format = query['info_format']

    # We only care about the requested subset
    ds = dataset[parameters]

    # TODO: Need to reproject??

    x_coord = np.linspace(bbox[0], bbox[2], width)
    y_coord = np.linspace(bbox[1], bbox[3], height)

    has_time_axis = [ds[parameter].cf.axes.get(
        'T') is not None for parameter in parameters]
    any_has_time_axis = True in has_time_axis

    if any_has_time_axis:
        if len(times) == 1:
            resampled_data = ds.cf.interp(T=times[0], longitude=x_coord, latitude=y_coord)
        elif len(times) > 1:
            resampled_data = ds.cf.interp(longitude=x_coord, latitude=y_coord)
            resampled_data = resampled_data.cf.sel(T=slice(times[0], times[1]))
        else:
            raise HTTPException(500, f"Invalid time requested: {times}")
    else:
        resampled_data = ds.cf.interp(longitude=x_coord, latitude=y_coord)

    x_axis = [strip_float(resampled_data.cf['longitude'][x])]
    y_axis = [strip_float(resampled_data.cf['latitude'][y])]
    resampled_data = resampled_data.cf.sel({'longitude': x_axis, 'latitude': y_axis})

    # When none of the parameters have data, drop it
    if any_has_time_axis and resampled_data[resampled_data.cf.axes['T'][0]].shape:
        resampled_data = resampled_data.dropna(resampled_data.cf.axes['T'][0], how='all')

    if not any_has_time_axis:
        t_axis = None
    elif len(times) == 1:
        t_axis = str(format_timestamp(resampled_data.cf['T']))
    else:
        t_axis = str(format_timestamp(resampled_data.cf['T']))

    parameter_info = {}
    ranges = {}

    for i_parameter, parameter in enumerate(parameters):
        info, range = create_parameter_feature_data(parameter, resampled_data, has_time_axis[i_parameter], t_axis, x_axis, y_axis)
        parameter_info[parameter] = info
        ranges[parameter] = range

    # For now, harcoding uv parameter grouping
    if len(parameters) == 2 and ('u_eastward' in parameters or 'u_eastward_max' in parameters):
        speed, direction = speed_and_dir_for_uv(resampled_data[parameters[0]], resampled_data[parameters[1]])
        speed_info, speed_range = create_parameter_feature_data(parameter, resampled_data, has_time_axis[i_parameter], t_axis, x_axis, y_axis, speed, 'Magnitude of velocity', 'magnitude_of_velocity')
        speed_parameter_name = f'{parameters[0]}:{parameters[1]}-mag'
        parameter_info[speed_parameter_name] = speed_info
        ranges[speed_parameter_name] = speed_range

        direction_info, direction_range = create_parameter_feature_data(parameter, resampled_data, has_time_axis[i_parameter], t_axis, x_axis, y_axis, direction, 'Direction of velocity', 'direction_of_velocity')
        direction_parameter_name = f'{parameters[0]}:{parameters[1]}-dir'
        parameter_info[direction_parameter_name] = direction_info
        ranges[direction_parameter_name] = direction_range

    axis = {
        't': {
            'values': t_axis
        },
        'x': {
            'values': x_axis
        },
        'y': {
            'values': y_axis
        }
    } if any_has_time_axis else {
        'x': {
            'values': x_axis
        },
        'y': {
            'values': y_axis
        }
    }

    referencing = [
        {
            'coordinates': ['t'],
            'system': {
                'type': 'TemporalRS',
                        'calendar': 'gregorian',
            }
        },
        {
            'coordinates': ['x', 'y'],
            'system': {
                'type': 'GeographicCRS',
                        'id': crs,
            }
        }
    ] if any_has_time_axis else [
        {
            'coordinates': ['x', 'y'],
            'system': {
                'type': 'GeographicCRS',
                'id': crs,
            }
        }
    ]

    return {
        'type': 'Coverage',
        'title': {
            'en': 'Extracted Profile Feature',
        },
        'domain': {
            'type': 'Domain',
            'domainType': 'PointSeries',
            'axes': axis,
            'referencing': referencing,
        },
        'parameters': parameter_info,
        'ranges': ranges
    }


