from typing import Tuple

import cf_xarray  # noqa
import numpy as np
import xarray as xr
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from xpublish_wms.query import WMSGetFeatureInfoQuery
from xpublish_wms.utils import (
    format_timestamp,
    round_float_values,
    speed_and_dir_for_uv,
    strip_float,
)


def create_parameter_feature_data(
    parameter,
    ds: xr.Dataset,
    t_axis,
    z_axis,
    x_axis,
    y_axis,
    extra_axes: dict = None,
    values=None,
    name=None,
    id=None,
) -> Tuple[dict, dict]:
    # TODO Use standard and long name?
    name = (
        name
        if name is not None
        else ds[parameter].cf.attrs.get(
            "long_name",
            ds[parameter].cf.attrs.get("name", parameter),
        )
    )
    id = (
        id if id is not None else ds[parameter].cf.attrs.get("standard_name", parameter)
    )

    info = {
        "type": "Parameter",
        "description": {
            "en": name,
        },
        "observedProperty": {
            "label": {
                "en": name,
            },
            "id": id,
        },
    }

    axis_names = []
    if t_axis is not None:
        axis_names.append("t")
    if z_axis is not None:
        axis_names.append("z")
    if extra_axes is not None:
        axis_names.extend(extra_axes.keys())
    axis_names.extend(["x", "y"])

    values = values if values is not None else ds[parameter]

    shape = []
    if t_axis is not None:
        shape.append(len(t_axis))
    if z_axis is not None:
        shape.append(len(z_axis))
    if extra_axes is not None:
        for _axis in extra_axes.values():
            shape.append(1)

    if isinstance(values, float):
        shape.extend([1, 1])
        values = [values]
    elif isinstance(values, list):
        shape.extend([1, 1])
        values = round_float_values(values)
    elif isinstance(values, xr.DataArray):
        values = values.values
        if values.ndim < 2:
            shape.extend([1, 1])
        else:
            shape = values.shape
        values = values.flatten().tolist()
    elif isinstance(values, np.ndarray):
        if values.ndim < 2:
            shape.extend([1, 1])
        else:
            shape = values.shape
        values = values.flatten().tolist()
    elif values is None:
        shape.extend([1, 1])
        values = [np.nan]

    range = {
        "type": "NdArray",
        "dataType": "float",
        "axisNames": axis_names,
        "shape": shape,
        "values": [None if np.isnan(v) else v for v in values],
    }

    return (info, range)


def get_feature_info(
    ds: xr.Dataset,
    query: WMSGetFeatureInfoQuery,
    query_params: dict,
) -> Response:
    """
    Return the WMS feature info for the dataset and given parameters
    """
    # Data selection
    if ":" in query.query_layers:
        parameters = query.query_layers.split(":")
        grouped = True
    else:
        parameters = query.query_layers.split(",")
        grouped = False
    time_str = query.time
    if time_str:
        times = list(dict.fromkeys([t.replace("Z", "") for t in time_str.split("/")]))
    else:
        times = []
    has_time_coord = [
        "time" in ds[parameter].cf.coordinates for parameter in parameters
    ]
    any_has_time_axis = True in has_time_coord

    elevation_str = query.elevation
    if elevation_str == "all":
        elevation = "all"
    elif elevation_str:
        elevation = list([float(e) for e in elevation_str.split("/")])
    else:
        elevation = []
    has_vertical_axis = [
        ds.gridded.has_elevation(ds[parameter]) for parameter in parameters
    ]
    any_has_vertical_axis = True in has_vertical_axis

    crs = query.crs
    if crs != "EPSG:4326":
        raise HTTPException(501, "Only EPSG:4326 is supported")

    bbox = [float(x) for x in query.bbox.split(",")]
    width = query.width
    height = query.height
    x = query.x
    y = query.y
    # format = query["info_format"]

    # We only care about the requested subset
    selected_ds = ds[parameters]

    # TODO: Need to reproject??
    x_coord = np.linspace(bbox[0], bbox[2], width)
    y_coord = np.linspace(bbox[1], bbox[3], height)

    if any_has_time_axis:
        if (
            len(selected_ds.cf["time"].shape) == 1
            and selected_ds.cf["time"].shape[0] == 1
        ):
            selected_ds = selected_ds.cf.isel(time=0)
        elif len(times) == 1:
            selected_ds = selected_ds.cf.interp(time=times[0])
        elif len(times) > 1:
            selected_ds = selected_ds.cf.sel(time=slice(times[0], times[1]))
        else:
            try:
                selected_ds = selected_ds.cf.isel(time=0)
            except Exception:
                # Skip it, time isn't a dimension, even though it is a coordinate
                pass

    # TODO: This is really difficult when we have multiple parameters
    # with different vertical dimensions, and even some without indexes
    if any_has_vertical_axis:
        if elevation == "all":
            # Dont select an elevation, just keep all elevation coords
            elevation = ds.gridded.elevations(selected_ds)

        selected_ds = ds.gridded.select_by_elevation(selected_ds, elevation)

    try:
        selected_ds, x_axis, y_axis = ds.gridded.sel_lat_lng(
            subset=selected_ds,
            lng=x_coord[x],
            lat=y_coord[y],
            parameters=parameters,
        )
    except ValueError as e:
        raise HTTPException(500, f"Error with grid type {ds.gridded.name}: {e}")

    # When none of the parameters have data, drop it
    if any_has_time_axis:
        time_coord_name = selected_ds.cf.coordinates["time"][0]
        if selected_ds[time_coord_name].shape:
            selected_ds = selected_ds.dropna(time_coord_name, how="all")

    if not any_has_time_axis:
        t_axis = None
    elif len(times) == 1:
        t_axis = [str(format_timestamp(selected_ds.cf["time"]))]
    else:
        t_axis = format_timestamp(selected_ds.cf["time"]).tolist()

    if not any_has_vertical_axis:
        z_axis = None
        elevation_name = None
        elevation_units = None
        elevation_positive = None
    else:
        elevation_name = selected_ds.cf["vertical"].attrs.get("standard_name", "")
        elevation_units = selected_ds.cf["vertical"].attrs.get("units", "sigma")
        elevation_positive = selected_ds.cf["vertical"].attrs.get("positive", "up")

        if len(elevation) < 2:
            z_axis = [strip_float(selected_ds.cf["vertical"])]
        else:
            z_axis = selected_ds.cf["vertical"].values.tolist()

    additional_queries = {}
    queries = set(query_params.keys())
    for param in parameters:
        available_coords = selected_ds.gridded.additional_coords(selected_ds[param])
        valid_quiries = list(set(available_coords) & queries)
        for q in valid_quiries:
            additional_queries[q] = query_params[q]

    selected_ds = selected_ds.sel(additional_queries)

    parameter_info = {}
    ranges = {}

    for i_parameter, parameter in enumerate(parameters):
        info, range = create_parameter_feature_data(
            parameter,
            selected_ds,
            t_axis,
            z_axis,
            x_axis,
            y_axis,
            extra_axes=additional_queries,
        )
        parameter_info[parameter] = info
        ranges[parameter] = range

    # For now, hardcoding uv parameter grouping
    if len(parameters) == 2 and grouped:
        speed, direction = speed_and_dir_for_uv(
            selected_ds[parameters[0]],
            selected_ds[parameters[1]],
        )
        speed_info, speed_range = create_parameter_feature_data(
            parameter,
            selected_ds,
            t_axis,
            z_axis,
            x_axis,
            y_axis,
            additional_queries,
            speed,
            "Magnitude of velocity",
            "magnitude_of_velocity",
        )
        speed_parameter_name = f"{parameters[0]}:{parameters[1]}-mag"
        parameter_info[speed_parameter_name] = speed_info
        ranges[speed_parameter_name] = speed_range

        direction_info, direction_range = create_parameter_feature_data(
            parameter,
            selected_ds,
            t_axis,
            z_axis,
            x_axis,
            y_axis,
            additional_queries,
            direction,
            "Direction of velocity",
            "direction_of_velocity",
        )
        direction_parameter_name = f"{parameters[0]}:{parameters[1]}-dir"
        parameter_info[direction_parameter_name] = direction_info
        ranges[direction_parameter_name] = direction_range

    axis = {}

    referencing = []

    if len(additional_queries) > 0:
        for key, value in additional_queries.items():
            axis[key] = {"values": [value]}
            referencing.append(
                {
                    "coordinates": [key],
                    "id": "custom",
                },
            )

    if any_has_time_axis:
        axis["t"] = {"values": t_axis}
        referencing.append(
            {
                "coordinates": ["t"],
                "system": {
                    "type": "TemporalRS",
                    "calendar": "gregorian",
                },
            },
        )
    if any_has_vertical_axis:
        axis["z"] = {"values": z_axis}
        referencing.append(
            {
                "coordinates": ["z"],
                "system": {
                    "type": "VerticalCRS",
                    "cs": {
                        "csAxes": [
                            {
                                "name": elevation_name,
                                "direction": elevation_positive,
                                "unit": elevation_units,
                            },
                        ],
                    },
                },
            },
        )

    axis["x"] = {"values": x_axis}
    axis["y"] = {"values": y_axis}
    referencing.append(
        {
            "coordinates": ["x", "y"],
            "system": {
                "type": "GeographicCRS",
                "id": crs,
            },
        },
    )

    return JSONResponse(
        content={
            "type": "Coverage",
            "title": {
                "en": "Extracted Profile Feature",
            },
            "domain": {
                "type": "Domain",
                "domainType": "PointSeries",
                "axes": axis,
                "referencing": referencing,
            },
            "parameters": parameter_info,
            "ranges": ranges,
        },
    )
