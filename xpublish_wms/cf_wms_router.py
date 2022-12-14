"""
cf_wms_router

OGC WMS router for datasets with CF convention metadata
"""
from cmath import isnan
import io
import logging
import xml.etree.ElementTree as ET

import cachey
import numpy as np
import pandas as pd
import cf_xarray  # noqa
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from xpublish.dependencies import get_cache, get_dataset
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from PIL import Image
from matplotlib import cm
from pykdtree.kdtree import KDTree

from xpublish_wms.utils import format_timestamp, lower_case_keys, round_float_values, speed_and_dir_for_uv, strip_float

logger = logging.getLogger("uvicorn")


cf_wms_router = APIRouter()


# WMS Styles declaration
# TODO: Add others beyond just simple raster
styles = [
    {
        'name': 'raster/default',
        'title': 'Raster',
        'abstract': 'The default raster styling, scaled to the given range. The palette can be overriden by replacing default with a matplotlib colormap name'
    }
]


def get_spatial_kdtree(ds: xr.Dataset, cache: cachey.Cache) -> KDTree:
    cache_key = f"dataset-kdtree-{ds.attrs['title']}"
    kd = cache.get(cache_key)
    if kd:
        return kd

    lng = ds.cf['longitude']
    lat = ds.cf['latitude']

    verts = np.column_stack((lng, lat))
    kd = KDTree(verts)

    cache.put(cache_key, kd, 5)

    return kd


def create_text_element(root, name: str, text: str):
    element = ET.SubElement(root, name)
    element.text = text
    return element


def create_capability_element(root, name: str, url: str, formats: list[str]):
    cap = ET.SubElement(root, name)
    # TODO: Add more image formats
    for fmt in formats:
        create_text_element(cap, 'Format', fmt)

    dcp_type = ET.SubElement(cap, 'DCPType')
    http = ET.SubElement(dcp_type, 'HTTP')
    get = ET.SubElement(http, 'Get')
    get.append(ET.Element('OnlineResource', attrib={
               'xlink:type': 'simple', 'xlink:href': url}))
    return cap


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


def get_capabilities(ds: xr.Dataset, request: Request):
    """
    Return the WMS capabilities for the dataset
    """
    wms_url = f'{request.base_url}{request.url.path.removeprefix("/")}'

    root = ET.Element('WMS_Capabilities', version='1.3.0', attrib={
                      'xmlns': 'http://www.opengis.net/wms', 'xmlns:xlink': 'http://www.w3.org/1999/xlink'})

    service = ET.SubElement(root, 'Service')
    create_text_element(service, 'Name', 'WMS')
    create_text_element(service, 'Title', 'XPublish WMS')
    create_text_element(service, 'Abstract', 'XPublish WMS')
    service.append(ET.Element('KeywordList'))
    service.append(ET.Element('OnlineResource', attrib={
                   'xlink:type': 'simple', 'xlink:href': 'http://www.opengis.net/spec/wms_schema_1/1.3.0'}))

    capability = ET.SubElement(root, 'Capability')
    request_tag = ET.SubElement(capability, 'Request')

    get_capabilities = create_capability_element(
        request_tag, 'GetCapabilities', wms_url, ['text/xml'])
    # TODO: Add more image formats
    get_map = create_capability_element(
        request_tag, 'GetMap', wms_url, ['image/png'])
    # TODO: Add more feature info formats
    get_feature_info = create_capability_element(
        request_tag, 'GetFeatureInfo', wms_url, ['text/json'])
    # TODO: Add more image formats
    get_legend_graphic = create_capability_element(
        request_tag, 'GetLegendGraphic', wms_url, ['image/png'])

    exeption_tag = ET.SubElement(capability, 'Exception')
    exception_format = ET.SubElement(exeption_tag, 'Format')
    exception_format.text = 'text/json'

    layer_tag = ET.SubElement(capability, 'Layer')
    create_text_element(layer_tag, 'Title',
                        ds.attrs.get('title', 'Untitled'))
    create_text_element(layer_tag, 'Description',
                        ds.attrs.get('description', 'No Description'))
    create_text_element(layer_tag, 'CRS', 'EPSG:4326')
    create_text_element(layer_tag, 'CRS', 'EPSG:3857')
    create_text_element(layer_tag, 'CRS', 'CRS:84')

    bounds = {
        'CRS': 'EPSG:4326',
        'minx': f'{ds.cf.coords["longitude"].min().values.item()}',
        'miny': f'{ds.cf.coords["latitude"].min().values.item()}',
        'maxx': f'{ds.cf.coords["longitude"].max().values.item()}',
        'maxy': f'{ds.cf.coords["latitude"].max().values.item()}'
    }

    for var in ds.data_vars:
        da = ds[var]

        # If there are not spatial coords, we cant view it with this router, sorry
        if 'longitude' not in da.cf.coords:
            continue

        attrs = da.cf.attrs
        layer = ET.SubElement(layer_tag, 'Layer', attrib={'queryable': '1'})
        create_text_element(layer, 'Name', var)
        create_text_element(layer, 'Title', attrs.get('long_name', attrs.get('name', var)))
        create_text_element(layer, 'Abstract', attrs.get('long_name', attrs.get('name', var)))
        create_text_element(layer, 'CRS', 'EPSG:4326')
        create_text_element(layer, 'CRS', 'EPSG:3857')
        create_text_element(layer, 'CRS', 'CRS:84')

        create_text_element(layer, 'Units', attrs.get('units', ''))

        # min_value = float(da.min())
        # create_text_element(layer, 'MinMag', min_value)

        # max_value = float(da.max())
        # create_text_element(layer, 'MaxMag', max_value)

        # Not sure if this can be copied, its possible variables have different extents within
        # a given dataset probably, but for now...
        bounding_box_element = ET.SubElement(layer, 'BoundingBox', attrib=bounds)

        if 'T' in da.cf.axes:
            times = format_timestamp(da.cf['T'])

            time_dimension_element = ET.SubElement(layer, 'Dimension', attrib={
                'name': 'time',
                'units': 'ISO8601',
                'default': times[-1],
            })
            # TODO: Add ISO duration specifier
            time_dimension_element.text = f"{','.join(times)}"

        style_tag = ET.SubElement(layer, 'Style')

        for style in styles:
            style_element = ET.SubElement(
                style_tag, 'Style', attrib={'name': style['name']})
            create_text_element(style_element, 'Title', style['title'])
            create_text_element(style_element, 'Abstract', style['abstract'])

            legend_url = f'{wms_url}?service=WMS&request=GetLegendGraphic&format=image/png&width=20&height=20&layers={var}&styles={style["name"]}'
            create_text_element(style_element, 'LegendURL', legend_url)

    ET.indent(root, space="\t", level=0)
    return Response(ET.tostring(root).decode('utf-8'), media_type='text/xml')


def get_map(dataset: xr.Dataset, query: dict, cache: cachey.Cache):
    """
    Return the WMS map for the dataset and given parameters
    """
    if not dataset.rio.crs:
        dataset = dataset.rio.write_crs(4326)

    ds = dataset.squeeze()
    bbox = [float(x) for x in query['bbox'].split(',')]
    width = int(query['width'])
    height = int(query['height'])
    crs = query.get('crs', None) or query.get('srs')
    parameter = query['layers']
    t = query.get('time', None)
    colorscalerange = [float(x) for x in query.get('colorscalerange', 'nan,nan').split(',')]
    autoscale = query.get('autoscale', 'false') != 'false'
    style = query['styles']
    stylename, palettename = style.split('/')

    # This is an image, so only use the timestep that was requested
    if t is not None:
        tstamp = pd.to_datetime(t).tz_localize(None)
        da = ds[parameter].cf.sel({'time': tstamp}, method='nearest').squeeze()
    else:
        da = ds[parameter].cf.isel({'time': 0})

    if da.cf.coords['longitude'].dims[0] == da.cf.coords['longitude'].name:
        # Regular grid
        # Unpack the requested data and resample
        clipped = da.rio.clip_box(*bbox, crs=crs)
        resampled_data = clipped.rio.reproject(
            dst_crs=crs,
            shape=(width, height),
            resampling=Resampling.nearest,
            transform=from_bounds(*bbox, width=width, height=height),
        )
    else:
        # irregular grid
        lats = np.linspace(bbox[0], bbox[2], width)
        lngs = np.linspace(bbox[1], bbox[3], height)
        grid_lngs, grid_lats = np.meshgrid(lngs, lats)
        pts = np.column_stack((grid_lngs.ravel(), grid_lats.ravel()))
        kd = get_spatial_kdtree(ds, cache)
        _, n = kd.query(pts)
        ni = n.argsort()
        pp = n[ni]
        # This is slow because it has to pull into numpy array, can we do better? 
        z = ds.zeta[0][pp].values
        resampled_data = z[ni.argsort()]

        # if the user has supplied a color range, use it. Otherwise autoscale
    if autoscale:
        min_value = float(ds[parameter].min())
        max_value = float(ds[parameter].max())
    else:
        min_value = colorscalerange[0]
        max_value = colorscalerange[1]

    ds_scaled = (resampled_data - min_value) / (max_value - min_value)

    # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
    # Otherwise default to rainbow
    if palettename == 'default':
        palettename = 'rainbow'
    im = Image.fromarray(np.uint8(cm.get_cmap(palettename)(ds_scaled) * 255))

    image_bytes = io.BytesIO()
    im.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    return Response(content=image_bytes, media_type='image/png')


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


def get_legend_info(dataset: xr.Dataset, query: dict):
    """
    Return the WMS legend graphic for the dataset and given parameters
    """
    parameter = query['layers']
    width: int = int(query['width'])
    height: int = int(query['height'])
    vertical = query.get('vertical', 'false') == 'true'
    colorbaronly = query.get('colorbaronly', 'False') == 'True'
    colorscalerange = [float(x) for x in query.get(
        'colorscalerange', 'nan,nan').split(',')]
    if isnan(colorscalerange[0]):
        autoscale = True
    else:
        autoscale = query.get('autoscale', 'false') != 'false'
    style = query['styles']
    stylename, palettename = style.split('/')

    ds = dataset.squeeze()

    # if the user has supplied a color range, use it. Otherwise autoscale
    if autoscale:
        min_value = float(ds[parameter].min())
        max_value = float(ds[parameter].max())
    else:
        min_value = colorscalerange[0]
        max_value = colorscalerange[1]

    scaled = (np.linspace(min_value, max_value, width)
              - min_value) / (max_value - min_value)
    data = np.ones((height, width)) * scaled

    if vertical:
        data = np.flipud(data.T)
        data = data.reshape((height, width))

    # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
    # Otherwise default to rainbow
    if palettename == 'default':
        palettename = 'rainbow'
    im = Image.fromarray(np.uint8(cm.get_cmap(palettename)(data) * 255))

    image_bytes = io.BytesIO()
    im.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    return Response(content=image_bytes, media_type='image/png')


@cf_wms_router.get('/')
def wms_root(request: Request, dataset: xr.Dataset = Depends(get_dataset), cache: cachey.Cache = Depends(get_cache)):
    query_params = lower_case_keys(request.query_params)
    method = query_params['request']
    logger.info(f'WMS: {method}')
    if method == 'GetCapabilities':
        return get_capabilities(dataset, request)
    elif method == 'GetMap':
        return get_map(dataset, query_params, cache)
    elif method == 'GetFeatureInfo' or method == 'GetTimeseries':
        return get_feature_info(dataset, query_params)
    elif method == 'GetLegendGraphic':
        return get_legend_info(dataset, query_params)
    else:
        raise HTTPException(
            status_code=404, detail=f"{method} is not a valid option for REQUEST")
