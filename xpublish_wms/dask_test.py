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
from perflog import PerfTimer, PerfLog

perflog = PerfLog.getLogger("getmap_performance.csv", "OgcWmsGetMap baseline")

def loaddata():
    options = {'anon': False, 'use_ssl': False }
    
    #ds = dask_connect.load_data_s3('s3://ncdis-ra/jsons/fort.63_post_1980-1981.json', options)
    #ds = dask_connect.load_data_s3('s3://nextgen-dmac/kerchunk/gfswave_global_kerchunk.json', options)
    
    #DBOFS datasets
    #ds = dask_connect.load_data_s3('s3://nextgen-dmac/nos/nos.dbofs.fields.best.nc.zarr', options)
    #ds = dask_connect.load_data_s3('s3://nextgen-dmac/nos/nos.dbofs.fields.f048.20230428.t00z.nc.zarr', options)    	
    ds = dask_connect.load_data_s3('s3://ioos-code-sprint-2022/nos/nos.dbofs.fields.20230530.t00z.nc.zarr', options)    	

    #ds = dask_connect.load_data_s3('/mnt/c/projects/xreds/datasets/gfswave_global_kerchunk.json', options)
    #persist = ds.persist()
    return ds


def getmap(ds):
    c = Cache(1e9, 1)
    wms = OgcWmsGetMap(c)

    # 40 year calls
    #query_params = "service=WMS&version=1.3.0&request=GetMap&layers=zeta&crs=EPSG:3857&bbox=-10018754.171394622,5009377.085697312,-5009377.085697312,10018754.17139462&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=1982-01-02T00:00:00Z&autoscale=false"
    #query_params = "service=WMS&version=1.3.0&request=GetMap&layers=zeta&crs=EPSG:3857&bbox=-10018754.171394622,0,-5009377.085697312,5009377.085697312&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=1982-01-02T00:00:00Z&autoscale=false"
    # DBOFS call
    query_params = "service=WMS&version=1.3.0&request=GetMap&layers=temp&crs=EPSG:3857&bbox=-8335916.556668181,4696291.017841227,-8296780.798186172,4735426.77632324&width=512&height=512&styles=raster/default&colorscalerange=10,18&time=2023-05-30T12:00:00Z"
    #query_params = "service=WMS&version=1.3.0&request=GetMap&layers=temp&crs=EPSG:3857&bbox=-8335916.556668181,4696291.017841227,-8296780.798186172,4735426.77632324&width=512&height=512&styles=raster/default&colorscalerange=10,18&time=2023-05-03T12:00:00Z"

    query_params = query_params.split('&')
    query_params = dict(i.lower().split('=') for i in query_params)
    #return mpl_getmap.get_map(ds, query_params)
    return wms.get_map(ds, query_params)

#ds = loaddata()

dask = False
t = PerfTimer(perflog)
if dask:
    t.start("Dask connection")
    connection = DaskConnect(False)
    t.log("init connection")    

    #server.shutdown_cluster()

    #v = connection.client.submit(get_vars, f)
    #print(v.result())

    #connection.client.upload_file('utils.py')
    #connection.client.upload_file('getmap.py')
    #f = connection.client.submit(getmap, data)
    #bytes = f.result()

    data = connection.client.submit(loaddata)
    

else:
    t.start()
    ds = loaddata()
    bytes = getmap(ds)

#imageBinaryBytes = bytes.read()
if( bytes ):
    imageStream = BytesIO(bytes)
    imageFile = Image.open(imageStream)
    imageFile.save('result.png', 'png')

#with open("result.png", "wb") as png:
#    png.write(bytes)

t.log("Total execution time")

#dask_wms.init(server.client)

# example WMS map call:
# http://localhost:8090/datasets/gfswave_global/wms/?service=WMS&version=1.3.0&request=GetMap&layers=swh&crs=EPSG:3857&bbox=-10018754.171394622,7514065.628545966,-7514065.628545966,10018754.17139462&width=512&height=512&styles=raster/default&colorscalerange=0,10&time=2022-10-29T05:00:00Z
#server.client.submit(dask_wms.getmap())
