# xpublish-wms

[![PyPI](https://img.shields.io/pypi/v/xpublish-wms)](https://pypi.org/project/xpublish-wms/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xpublish-wms.svg)](https://anaconda.org/conda-forge/xpublish-wms)

[![Tests](https://github.com/xpublish-community/xpublish-wms/actions/workflows/tests.yml/badge.svg)](https://github.com/xpublish-community/xpublish-wms/actions/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/xpublish-community/xpublish-wms/main.svg)](https://results.pre-commit.ci/latest/github/xpublish-community/xpublish-wms/main)

[Xpublish](https://xpublish.readthedocs.io/en/latest/) routers for the [OGC WMS API](https://www.ogc.org/standards/wms).

## Installation

For `conda` users you can

```shell
conda install --channel conda-forge xpublish_wms
```

or, if you are a `pip` users

```shell
pip install xpublish_wms
```

Once it's installed, the plugin will register itself with Xpublish and WMS endpoints will be included for each dataset on the server.

## Dataset Requirements

At this time, only a subset of xarray datasets will work out of the box with this plugin. To be compatible, a dataset must contain CF compliant coordinate variables for `lat`, `lon`, `time`, and `vertical`. `time` and `vertical` are optional.

Currently the following grid/model types are supported:
- Regularly spaced lat/lon grids (Tested with GFS, GFS Wave models)
- Curvilinear grids (Tested with ROMS models CBOFS, DBOFS, TBOFS, WCOFS, GOMOFS, and CIOFS models)
- FVCOM grids (Tested with LOOFS, LSOFS, LMHOFS, and NGOFS2 models)
- SELFE grids (Tested with CREOFS model)
- 2d Non Dimensional grids (Tested with RTOFS, HRRR-Conus models)

### Supporting new grid/model types

If you have a dataset that is not supported, you can add support by creating a new `xpublish_wms.Grid` subclass and registering it with the `xpublish_wms.register_grid_impl` function. See the [xpublish_wms.grids](/xpublish_wms/grid.py) module for examples.

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/xpublish-community/xpublish-wms/issues).

## License and copyright

xpublish-wms is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/xpublish-community/xpublish-wms>.

## Support

Work on this plugin is sponsored by:

![IOOS](https://cdn.ioos.noaa.gov/media/2017/12/IOOS_logo.png)

[IOOS](https://ioos.noaa.gov/) ([github](https://github.com/ioos)) funds work on this plugin via the ["Reaching for the Cloud: Architecting a Cloud-Native Service-Based Ecosystem for DMAC"](https://github.com/asascience-open/nextgen-dmac) project being led by [RPS Ocean Science](https://www.rpsgroup.com/).
