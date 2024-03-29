# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from .ais import *
from .beamforming import *
from .core import *
from .datasets import *
from .detection import *
from .gis import *
from .io import *
from .plot import *
from .processing import *
from .raytracing import *
from .station import *
