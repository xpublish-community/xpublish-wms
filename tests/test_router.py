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

    # Try with uppercase query params
    response = xpublish_client.get(
        "datasets/air/wms?VERSION=1.3.0&SERVICE=WMS&REQUEST=GetCapabilities",
    )

    assert response.status_code == 200, "Response did not return successfully"

    raw_data = response.text
    assert "WMS_Capabilities" in raw_data, "Response does not contain WMS_Capabilities"


def test_cf_get_metadata(xpublish_client):
    # pragma: item=menu
    menu_response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMetadata&item=menu",
    )
    assert menu_response.status_code == 200, "Menu response did not return successfully"
    menu_data = menu_response.json()
    assert len(menu_data["children"]) == 1
    assert menu_data["children"][0]["id"] == "air"

    # pragma: item=layerdetails
    layerdetails_response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMetadata&item=layerdetails&layername=air",
    )
    assert (
        layerdetails_response.status_code == 200
    ), "Layer details response did not return successfully"
    layerdetails_data = layerdetails_response.json()
    assert layerdetails_data["layerName"] == "air"
    assert layerdetails_data["bbox"] == [-160.0, 15.0, -30.0, 75.0]
    assert len(layerdetails_data["timesteps"]) > 0
    assert layerdetails_data["additional_coords"] == []

    # pragma: item=timesteps
    timesteps_response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMetadata&item=timesteps&layername=air",
    )
    assert (
        timesteps_response.status_code == 200
    ), "Timestamps response did not return successfully"
    timesteps_data = timesteps_response.json()
    assert len(timesteps_data["timesteps"]) > 0
    assert timesteps_data["timesteps"][0] == "2013-01-01T00:00:00Z"
    assert timesteps_data["timesteps"][-1] == "2014-12-31T18:00:00Z"

    # pragma: item=minmax
    minmax_response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMetadata&item=minmax&layername=air&time=2013-01-01T00:00:00",
    )
    assert (
        minmax_response.status_code == 200
    ), "Minmax response did not return successfully"
    minmax_data = minmax_response.json()
    assert minmax_data == {"min": 227.0, "max": 302.6}


def test_get_map(xpublish_client):
    response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMap&layers=air&styles=raster/default&crs=EPSG:4326&bbox=-160.0,15.0,-30.0,75.0&width=512&height=513&format=image/png&colorscalerange=227.0,302.6",
    )
    assert response.status_code == 200, "Response did not return successfully"
    assert response.headers["content-type"] == "image/png", "Response is not an image"

    autoscale_response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetMap&layers=air&styles=raster/default&crs=EPSG:4326&bbox=-160.0,15.0,-30.0,75.0&width=512&height=513&format=image/png&colorscalerange=227.0,302.6&autoscale=True",
    )
    assert autoscale_response.status_code == 200, "Response did not return successfully"
    assert (
        autoscale_response.headers["content-type"] == "image/png"
    ), "Response is not an image"


def test_get_feature_info(xpublish_client):
    response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetFeatureInfo&crs=EPSG:4326&bbox=-100.0,30.0,-101.0,31.0&width=50&height=50&query_layers=air&x=25&y=25&time=2013-01-01T00:00:00",
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        response.headers["content-type"] == "application/json"
    ), "Response is not json"

    data = response.json()
    assert data["ranges"]["air"]["values"] == [287.005456059975]

    response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetFeatureInfo&crs=EPSG:4326&bbox=-100.0,30.0,-101.0,31.0&width=50&height=50&query_layers=air&x=25&y=25&time=2013-01-01T00:00:00/2013-01-02T00:00:00",
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        response.headers["content-type"] == "application/json"
    ), "Response is not json"

    data = response.json()
    assert data["domain"]["axes"]["t"]["values"] == [
        "2013-01-01T00:00:00Z",
        "2013-01-01T06:00:00Z",
        "2013-01-01T12:00:00Z",
        "2013-01-01T18:00:00Z",
        "2013-01-02T00:00:00Z",
    ]
    assert data["ranges"]["air"]["values"] == [
        287.005456059975,
        281.3892503123699,
        278.29012911286964,
        279.74098292378176,
        282.26759683465224,
    ]


def test_get_legend_graphic(xpublish_client):
    response = xpublish_client.get(
        "datasets/air/wms?version=1.3.0&service=WMS&request=GetLegendGraphic&layers=air&format=image/png&colorscalerange=227.0,302.6&width=200&height=50",
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert response.headers["content-type"] == "image/png", "Response is not an image"
