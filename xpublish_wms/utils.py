import logging
import math
from typing import Optional

import cartopy.geodesic
import numpy as np
import xarray as xr
from pyproj import Transformer

logger = logging.getLogger("uvicorn")


def lower_case_keys(d: dict) -> dict:
    return {k.lower(): v for k, v in d.items()}


def format_timestamp(value):
    return value.dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ").values


def strip_float(value):
    return float(value.values)


def parse_float(value):
    if "e" in value.lower():
        part_arr = value.lower().split("e")
        return float(part_arr[0].strip()) * (10 ** float(part_arr[1].strip()))

    return float(value.strip())


def round_float_values(v) -> list:
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


def lnglat_to_mercator(longitude, latitude):
    """
    Converts data array with cf standard lng/lat to mercator coordinates
    """
    constant = 20037508.34 / 180

    longitude = xr.where(longitude == 180, longitude - 0.000001, longitude)
    longitude = xr.where(longitude == -180, longitude + 0.000001, longitude)
    longitude = longitude * constant

    latitude = xr.where(latitude == 90, latitude - 0.000001, latitude)
    latitude = xr.where(latitude == -90, latitude + 0.000001, latitude)
    latitude = (
        np.log(np.tan((90 + latitude) * math.pi / 360)) / (math.pi / 180)
    ) * constant

    return longitude, latitude


to_lnglat = Transformer.from_crs(3857, 4326, always_xy=True)


to_mercator = Transformer.from_crs(4326, 3857, always_xy=True)


def argsel2d(lons, lats, lon0, lat0):
    """Find the indices of coordinate pair closest to another point.
    Adapted from https://github.com/xoceanmodel/xroms/blob/main/xroms/utilities.py which is failing to run for some reason

    Inputs
    ------
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.
    Returns
    -------
    Index or indices of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. Number of dimensions of
    returned indices will correspond to the shape of input lons.
    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.
    Example usage
    -------------
    >>> xroms.argsel2d(ds.lon_rho, ds.lat_rho, -96, 27)
    """

    # input lons and lats can be multidimensional and might be DataArrays or lists
    pts = list(zip(*(np.asarray(lons).flatten(), np.asarray(lats).flatten())))
    endpts = list(zip(*(np.asarray(lon0).flatten(), np.asarray(lat0).flatten())))

    G = cartopy.geodesic.Geodesic()  # set up class
    dist = np.asarray(G.inverse(pts, endpts)[:, 0])  # select distances specifically
    iclosest = abs(np.asarray(dist)).argmin()  # find indices of closest point
    # return index or indices in input array shape
    inds = np.unravel_index(iclosest, np.asarray(lons).shape)

    return inds


def sel2d(ds, lons, lats, lon0, lat0):
    """Find the value of the var at closest location to lon0,lat0.
    Adapted from https://github.com/xoceanmodel/xroms/blob/main/xroms/utilities.py which is failing to run for some reason

    Inputs
    ------
    ds: DataArray, ndarray, or DataSet
        Dataset to operate of
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.
    Returns
    -------
    Value in var of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. If var has other
    dimensions, they are brought along.
    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.
    This is meant to be used by the accessor to conveniently wrap
    `argsel2d`.
    Example usage
    -------------
    >>> xroms.sel2d(ds.temp, ds.lon_rho, ds.lat_rho, -96, 27)
    """
    inds = argsel2d(lons, lats, lon0, lat0)
    return ds.isel({lats.dims[0]: inds[0], lats.dims[1]: inds[1]})


def barycentric_weights(point, v1, v2, v3):
    """
    calculate the barycentric weight for each of the triangle vertices

    Inputs
    ------
    point: [float, float]
        [Longitude, Latitude] of comparison point.
    v1: [float, float]
        Vertex 1
    v2: [float, float]
        Vertex 1
    v3: [float, float]
        Vertex 1
    Returns
    -------
    3 weights relative to each of the 3 vertices. Then the interpolated value can be calculated using
    the following formula: (w1 * value1) + (w2 * value2) + (w3 * value3)
    """

    denominator = ((v2[1] - v3[1]) * (v1[0] - v3[0])) + (
        (v3[0] - v2[0]) * (v1[1] - v3[1])
    )

    w1 = (
        ((v2[1] - v3[1]) * (point[0] - v3[0])) + ((v3[0] - v2[0]) * (point[1] - v3[1]))
    ) / denominator
    w2 = (
        ((v3[1] - v1[1]) * (point[0] - v3[0])) + ((v1[0] - v3[0]) * (point[1] - v3[1]))
    ) / denominator
    w3 = 1 - w1 - w2

    return w1, w2, w3


def lat_lng_find_tri_ind(lng, lat, lng_values, lat_values, triangles):
    """
    Find the triangle index that the inputted lng/lat is within

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: xr.DataArray
        Longitudes of points corresponding to the indices in triangles
    lat_values: xr.DataArray
        Latitudes of points corresponding to the indices in triangles
    triangles: ndarray of shape (X, 3)
        Triangle mesh of indices as generated by tri.Triangulation
    Returns
    -------
    Triangle of indices that the lng/lat is within, which can be used with barycentric_weights
    to interpolate the value accurate to the lng/lat requested, or None if the point is outside
    the triangular mesh
    """
    lnglat_data = np.stack((lng_values[triangles], lat_values[triangles]), axis=2)

    d1 = (
        (lng - lnglat_data[:, 1, 0]) * (lnglat_data[:, 0, 1] - lnglat_data[:, 1, 1])
    ) - ((lnglat_data[:, 0, 0] - lnglat_data[:, 1, 0]) * (lat - lnglat_data[:, 1, 1]))
    d2 = (
        (lng - lnglat_data[:, 2, 0]) * (lnglat_data[:, 1, 1] - lnglat_data[:, 2, 1])
    ) - ((lnglat_data[:, 1, 0] - lnglat_data[:, 2, 0]) * (lat - lnglat_data[:, 2, 1]))
    d3 = (
        (lng - lnglat_data[:, 0, 0]) * (lnglat_data[:, 2, 1] - lnglat_data[:, 0, 1])
    ) - ((lnglat_data[:, 2, 0] - lnglat_data[:, 0, 0]) * (lat - lnglat_data[:, 0, 1]))

    has_neg = np.logical_or(np.logical_or(d1 < 0, d2 < 0), d3 < 0)
    has_pos = np.logical_or(np.logical_or(d1 > 0, d2 > 0), d3 > 0)

    not_in_tri = np.logical_and(has_neg, has_pos)
    tri_index = np.where(not_in_tri == 0)

    if len(tri_index) == 0 or len(tri_index[0]) == 0:
        return None
    else:
        return tri_index[0]


def lat_lng_find_tri(lng, lat, lng_values, lat_values, triangles):
    """
    Find the triangle that the inputted lng/lat is within

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: xr.DataArray
        Longitudes of points corresponding to the indices in triangles
    lat_values: xr.DataArray
        Latitudes of points corresponding to the indices in triangles
    triangles: ndarray of shape (X, 3)
        Triangle mesh of indices as generated by tri.Triangulation
    Returns
    -------
    Triangle of indices that the lng/lat is within, which can be used with barycentric_weights
    to interpolate the value accurate to the lng/lat requested, or None if the point is outside
    the triangular mesh
    """
    tri_index = lat_lng_find_tri_ind(lng, lat, lng_values, lat_values, triangles)
    if tri_index is None:
        return None
    else:
        return triangles[tri_index][0]


def bilinear_interp(percent_point, percent_quad, value_quad):
    """
    Calculates the bilinear interpolation of values provided by the value_quad, where the percent_quad and
    percent_point variables determine where the point to be interpolated is

    Inputs
    ------
    percent_point: [..., float, float]
        [Longitude, Latitude] vertex normalized based on lat_lng_quad_percentage
    percent_quad: [..., [float, float], [float, float], [float, float], [float, float]]
        [Longitude, Latitude] vertices representing each corner of the quad normalized based on lat_lng_quad_percentage
    value_quad: [..., float, float, float, float]
        Data values at each corner represented by the percent_quad
    Returns
    -------
    The interpolated value at the percent_point specified
    """

    a = -percent_quad[0][0][0] + percent_quad[0][1][0]
    b = -percent_quad[0][0][0] + percent_quad[1][0][0]
    c = (
        percent_quad[0][0][0]
        - percent_quad[1][0][0]
        - percent_quad[0][1][0]
        + percent_quad[1][1][0]
    )
    d = percent_point[0] - percent_quad[0][0][0]
    e = -percent_quad[0][0][1] + percent_quad[0][1][1]
    f = -percent_quad[0][0][1] + percent_quad[1][0][1]
    g = (
        percent_quad[0][0][1]
        - percent_quad[1][0][1]
        - percent_quad[0][1][1]
        + percent_quad[1][1][1]
    )
    h = percent_point[1] - percent_quad[0][0][1]

    alpha_denominator = 2 * c * e - 2 * a * g
    beta_denominator = 2 * c * f - 2 * b * g

    # for regular grids, just use x/y percents as alpha/beta
    if alpha_denominator == 0 or beta_denominator == 0:
        alpha = percent_point[0]
        beta = percent_point[1]
    else:
        alpha = (
            -(
                b * e
                - a * f
                + d * g
                - c * h
                + np.sqrt(
                    -4 * (c * e - a * g) * (d * f - b * h)
                    + np.power((b * e - a * f + d * g - c * h), 2),
                )
            )
            / alpha_denominator
        )
        beta = (
            b * e
            - a * f
            - d * g
            + c * h
            + np.sqrt(
                -4 * (c * e - a * g) * (d * f - b * h)
                + np.power((b * e - a * f + d * g - c * h), 2),
            )
        ) / beta_denominator

    return (1 - alpha) * (
        (1 - beta) * value_quad[..., 0, 0] + beta * value_quad[..., 1, 0]
    ) + alpha * ((1 - beta) * value_quad[..., 0, 1] + beta * value_quad[..., 1, 1])


def lat_lng_quad_percentage(lng, lat, lng_values, lat_values, quad):
    """
    Calculates the percentage of each point in the lng_values & lat_values list, where the min & max are 0 & 1
    respectively. Also calculates the percentage for the lng/lat point

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: [float, float, float, float]
        Longitudes of points corresponding to each corner of the quad
    lat_values: [float, float, float, float]
        Latitudes of points corresponding to each corner of the quad
    Returns
    -------
    Quad of percentages based on the input lng_values & lat_values. Also returns a lng/lat vertex as a percentage
    within the percent quad
    """

    lngs = lng_values[quad[0][0] : (quad[1][0] + 1), quad[0][1] : (quad[1][1] + 1)]
    lats = lat_values[quad[0][0] : (quad[1][0] + 1), quad[0][1] : (quad[1][1] + 1)]

    lng_min = np.min(lngs)
    lng_max = np.max(lngs)
    lat_min = np.min(lats)
    lat_max = np.max(lats)

    lng_denominator = lng_max - lng_min
    lat_denominator = lat_max - lat_min

    percent_quad = np.zeros((2, 2, 2))
    percent_quad[:, :, 0] = (lngs - lng_min) / lng_denominator
    percent_quad[:, :, 1] = (lats - lat_min) / lat_denominator

    percent_lng = (lng - lng_min) / lng_denominator
    percent_lat = (lat - lat_min) / lat_denominator

    return percent_quad, [percent_lng, percent_lat]


def lat_lng_find_quad(lng, lat, lng_values, lat_values):
    """
    Find the quad that the inputted lng/lat is within

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: xr.DataArray
        Longitudes of points corresponding to the indices in triangles
    lat_values: xr.DataArray
        Latitudes of points corresponding to the indices in triangles
    Returns
    -------
    Quad of indices that the lng/lat is within, which can be used with lat_lng_quad_percentage and bilinear_interp
    to interpolate the value accurate to the lng/lat requested, or None if the point is outside the mesh
    """

    lnglat_data = np.stack((lng_values, lat_values), axis=2)

    x0y0tox0y1 = np.where(
        (
            (lnglat_data[1:, :-1, 0] - lnglat_data[:-1, :-1, 0])
            * (lat - lnglat_data[:-1, :-1, 1])
            - (lng - lnglat_data[:-1, :-1, 0])
            * (lnglat_data[1:, :-1, 1] - lnglat_data[:-1, :-1, 1])
        )
        <= 0,
        1,
        0,
    )
    x0y1tox1y1 = np.where(
        (
            (lnglat_data[1:, 1:, 0] - lnglat_data[1:, :-1, 0])
            * (lat - lnglat_data[1:, :-1, 1])
            - (lng - lnglat_data[1:, :-1, 0])
            * (lnglat_data[1:, 1:, 1] - lnglat_data[1:, :-1, 1])
        )
        <= 0,
        1,
        0,
    )
    x1y1tox1y0 = np.where(
        (
            (lnglat_data[:-1, 1:, 0] - lnglat_data[1:, 1:, 0])
            * (lat - lnglat_data[1:, 1:, 1])
            - (lng - lnglat_data[1:, 1:, 0])
            * (lnglat_data[:-1, 1:, 1] - lnglat_data[1:, 1:, 1])
        )
        <= 0,
        1,
        0,
    )
    x1y0tox0y0 = np.where(
        (
            (lnglat_data[:-1, :-1, 0] - lnglat_data[:-1, 1:, 0])
            * (lat - lnglat_data[:-1, 1:, 1])
            - (lng - lnglat_data[:-1, 1:, 0])
            * (lnglat_data[:-1, :-1, 1] - lnglat_data[:-1, 1:, 1])
            <= 0
        ),
        1,
        0,
    )

    top_left_index = np.where(
        np.logical_and(
            np.logical_and(x0y0tox0y1, x0y1tox1y1),
            np.logical_and(x1y1tox1y0, x1y0tox0y0),
        ),
    )

    if len(top_left_index) == 0 or len(top_left_index[0]) == 0:
        return None
    else:
        return [
            [top_left_index[0][0], top_left_index[1][0]],
            [top_left_index[0][0] + 1, top_left_index[1][0] + 1],
        ]


def filter_data_within_bbox(
    da: xr.DataArray,
    bbox: list[float],
    buffer: Optional[float] = 0.0,
) -> xr.DataArray:
    """
    Filter a DataArray to include only the data within a specified bounding
    box, optionally expanded by a buffer.

    This function filters the input DataArray based on geographical coordinates,
    returning a subset of the data that lies within the given bounding box. The
    bounding box can be expanded by a specified buffer (in degrees) to ensure
    inclusivity of data points on the boundaries.

    Parameters:
    - da (xr.DataArray): The DataArray to be filtered.
    - bbox (list[float]): A list of four floats representing the geographical
        bounding box in the order [min_longitude, min_latitude, max_longitude,
        max_latitude].
    - buffer (float, optional): The amount by which to expand the bounding box
        on all sides. Default is 0.0, which means no expansion.

    Returns:
    - xr.DataArray: The filtered DataArray containing only the data within the
        expanded bounding box.

    Note:
    - This function is only meant for use with EPSG:4326 (lon/lat) gridded data.
    """

    # Get the x and y values
    x = np.array(da.x.values)
    y = np.array(da.y.values)

    # Find the indices of the data within the bounding box
    x_inds = np.where((x >= (bbox[0] - buffer)) & (x <= (bbox[2] + buffer)))[0]
    y_inds = np.where((y >= (bbox[1] - buffer)) & (y <= (bbox[3] + buffer)))[0]

    # Select and return the data within the bounding box
    return da.isel(x=x_inds, y=y_inds)
