from io import BytesIO

import cachey
import cf_xarray
from fastapi.responses import StreamingResponse
from matplotlib.figure import Figure
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import matplotlib
import time
from scipy.spatial import Delaunay


from xpublish_wms.utils import to_lnglat

matplotlib.use('Agg')


def get_map(ds: xr.Dataset, query: dict, cache: cachey.Cache):
    start = time.time()

    # Decode request params
    parameter = query['layers']
    da = ds[parameter]

    time_str = query.get('time', None)
    if time_str:
        time_ = pd.to_datetime(time_str).tz_localize(None)
    else:
        time_ = None
    has_time = 'time' in da.cf.coordinates

    elevation_str = query.get('elevation', None)
    if elevation_str:
        elevation = float(elevation_str)
    else: 
        elevation = None
    has_elevation = 'vertical' in da.cf.coordinates

    # TODO: Make this based on the actual chunks of the dataset, for now brute forcing to time and variable
    if has_time:
        cache_key = f"{parameter}"
    else:
        cache_key = f"{parameter}_{time_str}"
    cache_coord_key = f"{parameter}_coords"

    data_cache_key = f"{cache_key}_data"
    x_cache_key = f"{cache_coord_key}_x"
    y_cache_key = f"{cache_coord_key}_y"

    # Grid
    crs = query.get('crs', None) or query.get('srs')
    bbox = [float(x) for x in query['bbox'].split(',')]
    width = int(query['width'])
    height = int(query['height'])

    if crs == 'EPSG:3857':
        bbox_lng, bbox_lat = to_lnglat.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
        bbox = [*bbox_lng, *bbox_lat]
    else:
        bbox = [bbox[0], bbox[2], bbox[1], bbox[3]]

    # Output style
    style = query.get('styles', "raster/default")
    cmap = style.split('/')[1]
    if cmap == "default":
        cmap = "jet"

    autoscale = query.get('autoscale', "false") == "true"
    if not autoscale:
        vmin, vmax = [float(x) for x in query.get('colorscalerange', 'nan,nan').split(',')]
    else:
        vmin = vmax = None

    unpack_query_checkpoint = time.time()
    print(f'Extract query: {unpack_query_checkpoint - start}s')

    if time_ is not None and has_time:
        da = da.cf.sel(
            {'time': time_},
            method="nearest"
        )
    elif has_time:
        da = da.cf.isel({
            'time': -1
        })

    if elevation is not None and has_elevation:
        da = da.cf.sel(
            {'vertical': elevation},
            method="nearest"
        )
    elif has_elevation:
        da = da.cf.isel({
            'vertical': 0
        })

    select_data_checkpoint = time.time()
    print(f'Select data: {select_data_checkpoint - unpack_query_checkpoint}s')

    data = cache.get(data_cache_key, None)
    if data is None:
        data = np.array(da.values)
        cache.put(data_cache_key, data, cost=50)

    x = cache.get(x_cache_key, None)
    if x is None:
        x = np.array(da.cf['longitude'].values)
        cache.put(x_cache_key, x, cost=50)

    y = cache.get(y_cache_key, None)
    if y is None:
        y = np.array(da.cf['latitude'].values)
        cache.put(y_cache_key, y, cost=50)

    inds = np.where((x >= (bbox[0] - 0.18)) & (x <= (bbox[1] + 0.18)) & (y >= (bbox[2] - 0.18)) & (y <= (bbox[3] + 0.18)))
    x_sel = x[inds]
    y_sel = y[inds]
    data_sel = data[inds]
    tris = tri.Triangulation(x_sel, y_sel)

    data_tris = data_sel[tris.triangles]
    mask = np.where(np.isnan(data_tris), [True], [False])
    triangle_mask = np.any(mask, axis=1)
    tris.set_mask(triangle_mask)

    download_data_checkpoint = time.time()
    print(f'Download data: {download_data_checkpoint - select_data_checkpoint}s')

    projection = ccrs.Mercator() if crs == "EPSG:3857" else ccrs.PlateCarree()

    dpi = 80
    fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
    fig.set_alpha(0)
    fig.set_figheight(height / dpi)
    fig.set_figwidth(width / dpi)
    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[], projection=projection)
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.set_clip_on(False)
    ax.set_position([0, 0, 1, 1])

    try:
        #ax.tripcolor(tris, data_sel, transform=ccrs.PlateCarree(), cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        ax.tricontourf(tris, data_sel, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, levels=50)
        #ax.pcolormesh(x, y, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    except Exception as e:
        print(e)
        print(bbox)

    ax.set_extent(bbox, crs=ccrs.PlateCarree())
    ax.axis('off')

    plot_checkpoint = time.time()
    print(f'Plot Data: {plot_checkpoint - download_data_checkpoint}s')

    buf = BytesIO()
    fig.savefig(buf, format='png', transparent=True, pad_inches=0, bbox_inches='tight')
    buf.seek(0)

    write_checkpoint = time.time()
    print(f'Write Plot: {write_checkpoint - plot_checkpoint}s')

    return StreamingResponse(buf, media_type='image/png')
