from enum import Enum

import xarray as xr


class GridType(Enum):
    REGULAR = 1
    SGRID = 2
    UNSUPPORTED = 255

    @classmethod
    def from_ds(cls, ds: xr.Dataset):
        if f'{ds.cf}'.startswith('SGRID'):
            return cls.SGRID
        
        try: 
            if 'latitude' in ds.cf['latitude'].dims:
                return cls.REGULAR
        except Exception:
            return cls.UNSUPPORTED
        
        return cls.UNSUPPORTED