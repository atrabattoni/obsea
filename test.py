import importlib

import cartopy.crs as ccrs
import numpy as np
import obsea
from obspy import read, read_inventory, UTCDateTime
import xarray as xr
import scipy.signal as sp

importlib.reload(obsea.ais)
importlib.reload(obsea.gis)
importlib.reload(obsea.io)
importlib.reload(obsea.core)
importlib.reload(obsea)



fname = obsea.get_dataset_path("ais_cls")
ais = obsea.read_cls(fname)
tracks = obsea.read_ais(ais, np.timedelta64(1, "D"))
track = tracks.iloc[0]
obsea.simplify(track)