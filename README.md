## xpublish-wms

[![PyPI](https://img.shields.io/pypi/v/xpublish-wms)](https://pypi.org/project/xpublish-wms/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xpublish-wms.svg)](https://anaconda.org/conda-forge/xpublish-wms)

[![Tests](https://github.com/xpublish-community/xpublish-wms/actions/workflows/tests.yml/badge.svg)](https://github.com/xpublish-community/xpublish-wms/actions/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/xpublish-community/xpublish-wms/main.svg)](https://results.pre-commit.ci/latest/github/xpublish-community/xpublish-wms/main)

[Xpublish](https://xpublish.readthedocs.io/en/latest/) routers for the [OGC WMS API](https://www.ogc.org/standards/wms).

### Documentation and code

*Coming soon*

### Installation

For `conda` users you can

```shell
conda install --channel conda-forge xpublish_wms
```

or, if you are a `pip` users

```shell
pip install xpublish_wms
```

Once it's installed, the plugin will register itself with Xpublish and WMS endpoints will be included for each dataset on the server.

### Dataset Requirements

At this time, only a subset of xarray datasets will work out of the box with this plugin. To be compatible, a dataset must have:

- CF Compliant `latitude` and `longitude` coordinates
- One of:
    - `latitude` and `longitude` dimensions that coorespond to the CF compliant coordinates
    - CF compliant SGRID metadata (`topology`)

Currently only regularly spaced lat/lng grids and SGRID grids are supported. If a datasets meets these requirements and does not work, please file an [issue](https://github.com/xpublish-community/xpublish-wms/issues). Pull requests to support other grid systems are encouraged!

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/xpublish-community/xpublish-wms/issues).

## License and copyright

xpublish-wms is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/xpublish-community/xpublish-wms>.

## Support

Work on this plugin is sponsored by:

![IOOS](https://cdn.ioos.noaa.gov/media/2017/12/IOOS_logo.png)

[IOOS](https://ioos.noaa.gov/) ([github](https://github.com/ioos)) funds work on this plugin via the ["Reaching for the Cloud: Architecting a Cloud-Native Service-Based Ecosystem for DMAC"](https://github.com/asascience-open/nextgen-dmac) project being led by [RPS Ocean Science](https://www.rpsgroup.com/).
