import xarray as xr


def select_layer(ds: xr.Dataset, layer_name: str) -> xr.DataArray:
    '''
    Selects the given layer from the specified dataset
    '''
    return ds[layer_name]


