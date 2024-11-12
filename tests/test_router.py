from fastapi.testclient import TestClient
import pytest
import xpublish

from xpublish_wms import CfWmsPlugin


@pytest.fixture(scope="session")
def cf_dataset():
    from cf_xarray.datasets import airds

    return airds


@pytest.fixture(scope="session")
def cf_xpublish(cf_dataset):
    rest = xpublish.Rest({"air": cf_dataset}, plugins={"wms": CfWmsPlugin()})

    return rest


@pytest.fixture(scope="session")
def cf_client(cf_xpublish):
    app = cf_xpublish.app
    client = TestClient(app)

    return client


def test_cf_get_capabilities(cf_client):
    response = cf_client.get("datasets/air/wms?version=1.3.0&service=WMS&request=GetCapabilities")

    #assert response.status_code == 200, "Response did not return successfully"

    raw_data = response.text
    print(raw_data)
    assert "WMS_Capabilities" in raw_data, "Response does not contain WMS_Capabilities"