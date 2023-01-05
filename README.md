## xpublish-wms

[![Tests](https://github.com/asascience-open/xpublish-wms/actions/workflows/tests.yml/badge.svg)](https://github.com/asascience-open/xpublish-wms/actions/workflows/tests.yml)

[Xpublish](https://xpublish.readthedocs.io/en/latest/) routers for the [OGC WMS API](https://www.ogc.org/standards/wms).

### Documentation and code

URLs for the docs and code.

### Installation

This package is not yet published to pypi, so install from source with pip: 

```
git+https://github.com/asascience-open/xpublish-wms@72ee989
```

### Example

```python
import xarray as xr
import xpublish
from xpublish.routers import base_router, zarr_router
from xpublish_wms import cf_wms_router


ds = xr.open_dataset("dataset.nc")

rest = xpublish.Rest(
    datasets,
    routers=[
        (base_router, {"tags": ["info"]}),
        (cf_wms_router, {"tags": ["wms"], "prefix": "/wms"}),
        (zarr_router, {"tags": ["zarr"], "prefix": "/zarr"}),
    ],
)
```

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/asascience-open/xpublish-wms/issues).


## License and copyright

xpublish-wms is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/asascience-open/xpublish-wms>.
