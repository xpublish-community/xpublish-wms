import io
import logging
from datetime import datetime
from typing import List

import cachey
import cf_xarray
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from fastapi.responses import StreamingResponse
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from PIL import Image
from matplotlib import cm
import matplotlib.tri as tri
import cartopy.crs as ccrs
from xpublish_wms.grid import GridType

from xpublish_wms.utils import to_lnglat


logger = logging.getLogger(__name__)


class OgcWmsGetMap:
    TIME_CF_NAME: str = "time"
    ELEVATION_CF_NAME: str = "vertical"
    DEFAULT_CRS: str = "EPSG:3857"
    DEFAULT_STYLE: str = "raster/default"
    DEFAULT_PALETTE: str = "turbo"

    #cache: cachey.Cache

    # Data selection
    parameter : str
    time: datetime = None
    has_time: bool
    elevation: float = None
    has_elevation: bool

    # Grid
    grid_type: GridType
    crs: str
    bbox = List[float]
    width: int
    height: int

    # Output style
    style: str
    colorscalerange: List[float]
    autoscale: bool

    def __init__(self, cache: cachey.Cache):
        self.cache = cache

    def get_map(self, ds: xr.Dataset, query: dict) -> StreamingResponse:
        """
        Return the WMS map for the dataset and given parameters
        """
        # Decode request params
        self.ensure_query_types(ds, query)

        # Select data according to request
        da = self.select_layer(ds)
        da = self.select_time(da)
        da = self.select_elevation(da)
        da = self.select_custom_dim(da)
       
        # Render the data using the render that matches the dataset type
        # The data selection and render are coupled because they are both driven by
        # The grid type for now. This can be revisited if we choose to interpolate or
        # use the contoured renderer for regular grid datasets
        image_buffer = io.BytesIO()
        render_result = self.render(da, image_buffer)
        if render_result:
            image_buffer.seek(0)

        return image_buffer.getvalue()

        return StreamingResponse(image_buffer, media_type="image/png")

    def ensure_query_types(self, ds: xr.Dataset, query: dict):
        """
        Decode request params

        :param query:
        :return:
        """
        self.grid_type = GridType.from_ds(ds)

        # Data selection
        self.parameter = query['layers']
        self.time_str = query.get('time', None)
        if self.time_str:
            self.time = pd.to_datetime(self.time_str).tz_localize(None)
        else:
            self.time = None
        self.has_time = 'time' in ds[self.parameter].cf.coordinates

        self.elevation_str = query.get('elevation', None)
        if self.elevation_str:
            self.elevation = float(self.elevation_str)
        else: 
            self.elevation = None
        self.has_elevation = 'vertical' in ds[self.parameter].cf.coordinates

        # Grid
        self.crs = query.get('crs', None) or query.get('srs')
        self.bbox = [float(x) for x in query['bbox'].split(',')]
        self.width = int(query['width'])
        self.height = int(query['height'])

        # Output style
        self.style = query.get('styles', self.DEFAULT_STYLE)
        # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
        # Otherwise default to rainbow
        try:
            self.stylename, self.palettename = self.style.split('/')
        except:
            self.stylename = "raster"
            self.palettename = 'default'
        finally: 
            if self.palettename == 'default': 
                self.palettename = self.DEFAULT_PALETTE

        self.colorscalerange = [float(x) for x in query.get('colorscalerange', 'nan,nan').split(',')]
        self.autoscale = query.get('autoscale', "false") == "true"

    def select_layer(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Select Dataset variable, according to WMS layers request
        :param ds:
        :return:
        """
        return ds[self.parameter]

    def select_time(self, da: xr.DataArray) -> xr.DataArray:
        """
        Ensure time selection

        If time is provided :
            - use cf_xarray to access time
            - by default use TIME_CF_NAME
            - method nearest to ensure at least one result

        Otherwise:
            - Get latest one

        :param da:
        :return:
        """
        if self.time is not None:
            da = da.cf.sel(
                {self.TIME_CF_NAME: self.time},
                method="nearest"
            )
        else:
            da = da.cf.isel({
                self.TIME_CF_NAME: -1
            })

        return da

    def select_elevation(self, da: xr.DataArray) -> xr.DataArray:
        '''
        Ensure elevation selection

        If elevation is provided :
            - use cf_xarray to access vertical coord
            - by default use ELEVATION_CF_NAME
            - method nearest to ensure at least one result

        Otherwise:
            - Get latest one

        :param da:
        :return:
        '''
        if self.elevation is not None and self.has_elevation:
            da = da.cf.sel(
                {self.ELEVATION_CF_NAME: self.elevation},
                method="nearest"
            )
        elif self.has_elevation:
            da = da.cf.isel({
                'vertical': 0
            })
        
        return da

    def select_custom_dim(self, da: xr.DataArray) -> xr.DataArray:
        """
        Select other dimension, ensuring a 2D array
        :param da:
        :return:
        """
        # Squeeze single value dimensions
        da = da.squeeze()

        # TODO: Filter dimension from custom query, if any

        # Squeeze multiple values dimensions, by selecting the last value
        for key in da.cf.coordinates.keys():
            if key in ("latitude", "longitude", "X", "Y"):
                continue

            coord = da.cf.coords[key]
            if coord.size > 1:
                da = da.cf.isel({key: -1})

        return da
    
    def render(self, da: xr.DataArray, buffer: io.BytesIO) -> bool:
        """
        Render the data array into an image buffer
        :param da:
        :return:
        """
        if self.grid_type == GridType.REGULAR:
            return self.render_regular_grid(da, buffer)
        elif self.grid_type == GridType.SGRID:
            return self.render_sgrid(da, buffer)
        else: 
            return False

    def render_regular_grid(self, da: xr.DataArray, buffer: io.BytesIO) -> bool:
        """
        Render the data array into an image buffer when the dataset is using a 
        regularly spaced rectangular grid
        :param da:
        :return:
        """
        # Some basic check for dataset
        if not da.rio.crs:
            da = da.rio.write_crs(self.DEFAULT_CRS)

        minx, miny, maxx, maxy = self.bbox
        da.compute()
        transform = from_bounds(
            west=minx, south=miny,
            east=maxx, north=maxy,
            width=self.width, height=self.height
        )
        clipped = da.rio.clip_box(
            minx=minx, miny=miny,
            maxx=maxx, maxy=maxy,
            crs=self.crs
        )
        resampled_data = clipped.rio.reproject(
            dst_crs=self.crs,
            shape=(self.width, self.height),
            resampling=Resampling.nearest,
            transform=transform,
        )
        
        if self.autoscale:
            min_value = float(da.min())
            max_value = float(da.max())
        else:
            min_value = self.colorscalerange[0]
            max_value = self.colorscalerange[1]

        da_scaled = (resampled_data - min_value) / (max_value - min_value)
        im = Image.fromarray(np.uint8(cm.get_cmap(self.palettename)(da_scaled) * 255))
        im.save(buffer, format='PNG')

        return True

    def render_sgrid(self, da: xr.DataArray, buffer: io.BytesIO) -> bool:
        """
        Render the data array into an image buffer when the dataset is using a 
        staggered (ala ROMS) grid
        :param da:
        :return:
        """
        # TODO: Make this based on the actual chunks of the dataset, for now brute forcing to time and variable
        if self.has_time:
            cache_key = f"{self.parameter}_{self.time_str}"
        else:
            cache_key = f"{self.parameter}"
        cache_coord_key = f"{self.parameter}_coords"

        data_cache_key = f"{cache_key}_data"
        x_cache_key = f"{cache_coord_key}_x"
        y_cache_key = f"{cache_coord_key}_y"

        if self.crs == 'EPSG:3857':
            bbox_lng, bbox_lat = to_lnglat.transform([self.bbox[0], self.bbox[2]], [self.bbox[1], self.bbox[3]])
            bbox = [*bbox_lng, *bbox_lat]
        else:
            bbox = [self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3]]

        data = self.cache.get(data_cache_key, None)
        if data is None:
            data = np.array(da.values)
            self.cache.put(data_cache_key, data, cost=50)

        x = self.cache.get(x_cache_key, None)
        if x is None:
            x = np.array(da.cf['longitude'].values)
            self.cache.put(x_cache_key, x, cost=50)

        y = self.cache.get(y_cache_key, None)
        if y is None:
            y = np.array(da.cf['latitude'].values)
            self.cache.put(y_cache_key, y, cost=50)

        inds = np.where((x >= (bbox[0] - 0.18)) & (x <= (bbox[1] + 0.18)) & (y >= (bbox[2] - 0.18)) & (y <= (bbox[3] + 0.18)))
        x_sel = x[inds]
        y_sel = y[inds]
        data_sel = data[inds]
        tris = tri.Triangulation(x_sel, y_sel)

        data_tris = data_sel[tris.triangles]
        mask = np.where(np.isnan(data_tris), [True], [False])
        triangle_mask = np.any(mask, axis=1)
        tris.set_mask(triangle_mask)

        projection = ccrs.Mercator() if self.crs == "EPSG:3857" else ccrs.PlateCarree()

        dpi = 80
        fig = Figure(dpi=dpi, facecolor='none', edgecolor='none')
        fig.set_alpha(0)
        fig.set_figheight(self.height / dpi)
        fig.set_figwidth(self.width / dpi)
        ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[], projection=projection)
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.set_clip_on(False)
        ax.set_position([0, 0, 1, 1])

        if not self.autoscale: 
            vmin, vmax = self.colorscalerange
        else:
            vmin, vmax = [None, None]

        try:
            #ax.tripcolor(tris, data_sel, transform=ccrs.PlateCarree(), cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
            ax.tricontourf(tris, data_sel, transform=ccrs.PlateCarree(), cmap=self.palettename, vmin=vmin, vmax=vmax, levels=80)
            #ax.pcolormesh(x, y, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        except Exception as e:
            print(e)
            print(bbox)

        ax.set_extent(bbox, crs=ccrs.PlateCarree())
        ax.axis('off')

        fig.savefig(buffer, format='png', transparent=True, pad_inches=0, bbox_inches='tight')
        return True
