import cf_xarray  # noqa
import pytest
import xpublish
from fastapi.testclient import TestClient

from xpublish_wms import CfWmsPlugin


@pytest.fixture(scope="session")
def air_dataset():
    from xarray.tutorial import open_dataset

    return open_dataset("air_temperature", chunks={})


@pytest.fixture(scope="session")
def xpublish_app(air_dataset):
    rest = xpublish.Rest({"air": air_dataset}, plugins={"wms": CfWmsPlugin()})

    return rest


@pytest.fixture(scope="session")
def xpublish_client(xpublish_app):
    app = xpublish_app.app
    client = TestClient(app)

    return client


def test_cf_get_capabilities(xpublish_client):
    response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetCapabilities",
    )

    assert response.status_code == 200, "Response did not return successfully"

    raw_data = response.text
    assert "WMS_Capabilities" in raw_data, "Response does not contain WMS_Capabilities"


def test_cf_get_metadata(xpublish_client):
    response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMetadata&item=menu",
    )

    assert response.status_code == 200, "Response did not return successfully"

    raw_data = response.json()
    print(raw_data)
