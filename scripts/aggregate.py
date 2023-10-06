import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
from utils import *

aggregate_date(os.getcwd(), 2018)