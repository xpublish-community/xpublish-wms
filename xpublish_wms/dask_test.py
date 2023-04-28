from dask_connect import DaskConnect
import dask_connect
import xarray as xr
import time
import utils
import getmap
from getmap import OgcWmsGetMap
from io import BytesIO
from PIL import Image
import dask_wms
from cachey import Cache


def loaddata():
    options = {'anon': True, 'use_ssl': False }
    #ds = dask_connect.load_data_s3('s3://ncdis-ra/jsons/fort.63_post_1980-1981.json', options)
    #ds = dask_connect.load_data_s3('s3://nextgen-dmac/kerchunk/gfswave_global_kerchunk.json', options)
    ds = dask_connect.load_data_s3('s3://nextgen-dmac/nos/nos.dbofs.fields.f048.20230428.t00z.nc.zarr', options)    	
    #ds = dask_connect.load_data_s3('/mnt/c/projects/xreds/datasets/gfswave_global_kerchunk.json', options)
    #persist = ds.persist()
    return ds


def getmap(ds):
    c = Cache(1e9, 1)
    wms = OgcWmsGetMap(c)

    # 40 year calls
    #query_params = "service=WMS&version=1.3.0&request=GetMap&layers=zeta&crs=EPSG:3857&bbox=-10018754.171394622,5009377.085697312,-5009377.085697312,10018754.17139462&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=1982-01-02T00:00:00Z&autoscale=false"
    #query_params = "service=WMS&version=1.3.0&request=GetMap&layers=zeta&crs=EPSG:3857&bbox=-10018754.171394622,0,-5009377.085697312,5009377.085697312&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=1982-01-02T00:00:00Z&autoscale=false"
    # GFS calls
    query_params = "service=WMS&version=1.3.0&request=GetMap&layers=temp&crs=EPSG:3857&bbox=-8335916.556668181,4696291.017841227,-8296780.798186172,4735426.77632324&width=512&height=512&styles=raster/default&colorscalerange=10,18&time=2023-04-30T12:00:00Z"

    query_params = query_params.split('&')
    query_params = dict(i.lower().split('=') for i in query_params)
    #return mpl_getmap.get_map(ds, query_params)
    return wms.get_map(ds, query_params)

#ds = loaddata()

dask = False
if dask:
    start = time.time_ns()
    connection = DaskConnect(False)
    elapsedMs = (time.time_ns() - start) / 1000000
    print("Connection took:", elapsedMs, "ms")

    #server.shutdown_cluster()
    start = time.time_ns()

    #v = connection.client.submit(get_vars, f)
    #print(v.result())

    connection.client.upload_file('utils.py')
    connection.client.upload_file('getmap.py')

    data = connection.client.submit(loaddata)
    f = connection.client.submit(getmap, data)

    bytes = f.result()

else:
    start = time.time_ns()
    ds = loaddata()
    bytes = getmap(ds)

#imageBinaryBytes = bytes.read()
imageStream = BytesIO(bytes)
imageFile = Image.open(imageStream)
imageFile.save('result.png', 'png')

#with open("result.png", "wb") as png:
#    png.write(bytes)

elapsedMs = (time.time_ns() - start) / 1000000
print("Execution time:", elapsedMs, "ms")

#dask_wms.init(server.client)

# example WMS map call:
# http://localhost:8090/datasets/gfswave_global/wms/?service=WMS&version=1.3.0&request=GetMap&layers=swh&crs=EPSG:3857&bbox=-10018754.171394622,7514065.628545966,-7514065.628545966,10018754.17139462&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=2022-10-29T05:00:00Z
#server.client.submit(dask_wms.getmap())
