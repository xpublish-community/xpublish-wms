import cf_xarray
import rioxarray
import xarray as xr
import fsspec
from dask_gateway import Gateway, BasicAuth


class DaskConnect:
    def __init__(self, newcluster: bool = False, scale: int = 4):
        self._gateway = Gateway(
            "http://k8s-daskgate-traefikd-b0c7f98d0b-566bb45619969cbc.elb.us-east-1.amazonaws.com:8000",
            auth=BasicAuth(password="connect_1234")
        )
        clusters = self._gateway.list_clusters()
        if len(clusters) > 0 and not newcluster:
            print('Connecting to cluster: ' + clusters[0].name)
            self._cluster = self._gateway.connect(clusters[0].name)
        else:
            print('Creating new cluster')
            self._cluster = self._gateway.new_cluster()
            self._cluster.scale(scale)
        self._client = self._cluster.get_client()
        self._client.upload_file('dask_connect.py')

    def __del__(self):
        self._client.close()
        self._cluster.close()
        self._gateway.close()

    @property
    def gateway(self):
        return self._gateway

    @property
    def client(self):
        return self._client

    def shutdown_cluster(self):
        self.gateway.stop_cluster(self._cluster.name)


def load_data_s3(path: str, options: dict) -> xr.Dataset:
    fs = fsspec.filesystem("reference", fo=path, remote_protocol='s3', remote_options=options,
                           target_options=options)
    m = fs.get_mapper("")
        
    ds = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False), chunks={'valid_time':1},
                         drop_variables='orderedSequenceData')

    #if ds.cf.coords['longitude'].dims[0] == 'longitude':
    #    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
        # TODO: Yeah this should not be assumed... but for regular grids we will viz with rioxarray so for now we will assume
    #    ds = ds.rio.write_crs(4326)
    return ds