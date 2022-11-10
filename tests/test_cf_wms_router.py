import pytest
import xarray
import dask.array
import xpublish
from fastapi.testclient import TestClient

from xpublish_wms.cf_wms_router import cf_wms_router


@pytest.fixture(scope="session")
def cf_dataset():
    from cf_xarray.datasets import airds

    return airds


@pytest.fixture(scope="session")
def cf_xpublish(cf_dataset):
    rest = xpublish.Rest({"air": cf_dataset}, routers=[
        (cf_wms_router, {'prefix': '/wms'})
    ])

    return rest


@pytest.fixture(scope="session")
def cf_client(cf_xpublish):
    app = cf_xpublish.app
    client = TestClient(app)

    return client


def test_cf_get_feature_info(cf_client, cf_dataset):
    # TODO

    assert 1 == 1