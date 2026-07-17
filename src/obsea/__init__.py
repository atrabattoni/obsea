# -*- coding: utf-8 -*-
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

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
