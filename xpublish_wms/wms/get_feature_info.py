from typing import Tuple

import cf_xarray  # noqa
import numpy as np
import xarray as xr
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from xpublish_wms.grid import GridType, sel2d
from xpublish_wms.utils import (
    format_timestamp,
    round_float_values,
    speed_and_dir_for_uv,
    strip_float,
)


def create_parameter_feature_data(
    parameter,
    ds: xr.Dataset,
    has_time_axis,
    t_axis,
    x_axis,
    y_axis,
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

    axis_names = ["t", "x", "y"] if has_time_axis else ["x", "y"]
    shape = (
        [len(t_axis), len(x_axis), len(y_axis)]
        if has_time_axis
        else [len(x_axis), len(y_axis)]
    )
    values = values if values is not None else ds[parameter]
    values = round_float_values(values.squeeze().values.tolist())

    if isinstance(values, float):
        values = [values]

    range = {
        "type": "NdArray",
        "dataType": "float",
        # TODO: Some fields might not have a time field, and some might have an elevation field
        "axisNames": axis_names,
        "shape": shape,
        "values": [None if np.isnan(v) else v for v in values],
    }

    return (info, range)


def get_feature_info(ds: xr.Dataset, query: dict) -> Response:
    """
    Return the WMS feature info for the dataset and given parameters
    """
    grid_type = GridType.from_ds(ds)

    # Data selection
    if ":" in query["query_layers"]:
        parameters = query["query_layers"].split(":")
    else:
        parameters = query["query_layers"].split(",")
    time_str = query.get("time", None)
    if time_str:
        times = list(dict.fromkeys([t.replace("Z", "") for t in time_str.split("/")]))
    else:
        times = []
    has_time_axis = ["time" in ds[parameter].cf.coordinates for parameter in parameters]
    any_has_time_axis = True in has_time_axis

    elevation_str = query.get("elevation", None)
    if elevation_str:
        elevation = list([float(e) for e in elevation_str.split("/")])
    else:
        elevation = None
    has_vertical_axis = [
        ds[parameter].cf.axes.get("T") is not None for parameter in parameters
    ]
    has_vertical_axis = [
        "vertical" in ds[parameter].cf.coordinates for parameter in parameters
    ]
    any_has_vertical_axis = True in has_vertical_axis

    crs = query.get("crs", None) or query.get("srs")
    bbox = [float(x) for x in query["bbox"].split(",")]
    width = int(query["width"])
    height = int(query["height"])
    x = int(query["x"])
    y = int(query["y"])
    # format = query["info_format"]

    # We only care about the requested subset
    selected_ds = ds[parameters]

    # TODO: Need to reproject??
    x_coord = np.linspace(bbox[0], bbox[2], width)
    y_coord = np.linspace(bbox[1], bbox[3], height)

    if any_has_time_axis:
        if len(times) == 1:
            selected_ds = ds.cf.interp(time=times[0])
        elif len(times) > 1:
            selected_ds = ds.cf.sel(time=slice(times[0], times[1]))
        else:
            selected_ds = ds.cf.isel(time=0)

    if any_has_vertical_axis:
        if elevation is not None:
            selected_ds = selected_ds.cf.interp(vertical=elevation)
        else:
            selected_ds = selected_ds.cf.isel(vertical=0)

    if grid_type == GridType.REGULAR:
        selected_ds = selected_ds.cf.interp(longitude=x_coord, latitude=y_coord)
        selected_ds = selected_ds.cf.isel(longitude=x, latitude=y)
        x_axis = [strip_float(selected_ds.cf["longitude"])]
        y_axis = [strip_float(selected_ds.cf["latitude"])]
    elif grid_type == GridType.SGRID:
        topology = ds.cf["grid_topology"]
        lng_coord, lat_coord = topology.attrs["face_coordinates"].split(" ")
        selected_ds = sel2d(
            selected_ds,
            lons=selected_ds.cf[lng_coord],
            lats=selected_ds.cf[lat_coord],
            lon0=x_coord[x],
            lat0=y_coord[y],
        )
        x_axis = [strip_float(selected_ds.cf[lng_coord])]
        y_axis = [strip_float(selected_ds.cf[lat_coord])]
    else:
        raise HTTPException(500, f"Unsupported grid type: {grid_type}")

    # When none of the parameters have data, drop it
    time_coord_name = selected_ds.cf.coordinates["time"][0]
    if any_has_time_axis and selected_ds[time_coord_name].shape:
        selected_ds = selected_ds.dropna(time_coord_name, how="all")

    if not any_has_time_axis:
        t_axis = None
    elif len(times) == 1:
        t_axis = str(format_timestamp(selected_ds.cf["time"]))
    else:
        t_axis = str(format_timestamp(selected_ds.cf["time"]))

    parameter_info = {}
    ranges = {}

    for i_parameter, parameter in enumerate(parameters):
        info, range = create_parameter_feature_data(
            parameter,
            selected_ds,
            has_time_axis[i_parameter],
            t_axis,
            x_axis,
            y_axis,
        )
        parameter_info[parameter] = info
        ranges[parameter] = range

    # For now, hardcoding uv parameter grouping
    if len(parameters) == 2 and (
        "u_eastward" in parameters or "u_eastward_max" in parameters
    ):
        speed, direction = speed_and_dir_for_uv(
            selected_ds[parameters[0]],
            selected_ds[parameters[1]],
        )
        speed_info, speed_range = create_parameter_feature_data(
            parameter,
            selected_ds,
            has_time_axis[i_parameter],
            t_axis,
            x_axis,
            y_axis,
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
            has_time_axis[i_parameter],
            t_axis,
            x_axis,
            y_axis,
            direction,
            "Direction of velocity",
            "direction_of_velocity",
        )
        direction_parameter_name = f"{parameters[0]}:{parameters[1]}-dir"
        parameter_info[direction_parameter_name] = direction_info
        ranges[direction_parameter_name] = direction_range

    axis = (
        {"t": {"values": t_axis}, "x": {"values": x_axis}, "y": {"values": y_axis}}
        if any_has_time_axis
        else {"x": {"values": x_axis}, "y": {"values": y_axis}}
    )

    referencing = (
        [
            {
                "coordinates": ["t"],
                "system": {
                    "type": "TemporalRS",
                    "calendar": "gregorian",
                },
            },
            {
                "coordinates": ["x", "y"],
                "system": {
                    "type": "GeographicCRS",
                    "id": crs,
                },
            },
        ]
        if any_has_time_axis
        else [
            {
                "coordinates": ["x", "y"],
                "system": {
                    "type": "GeographicCRS",
                    "id": crs,
                },
            },
        ]
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
