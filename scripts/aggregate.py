import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
from utils import *

def main_aggregate(YEAR: object) -> object:
    aggregate_date(data_directory, YEAR)

if __name__ == "__main__":
    main_aggregate()
