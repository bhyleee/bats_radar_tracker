# # Establish place

import os
import pathlib
import tempfile
from datetime import datetime, date
import rasterio as rio
import shutil
from utils import *


def download(date_start, date_end, TOWER):
    start_date = date(date_start)
    end_date = date(date_end)
    current_date = datetime.today().strftime('%Y%m%d')
    DOPPLER_DIR = DATA_DIR.joinpath('doppler', current_date)
    DOPPLER_DIR.mkdir(parents=True, exist_ok=True)
    YEAR_DIR = DOPPLER_DIR.joinpath(str(start_date.strftime('%Y')))
    YEAR_DIR.mkdir(parents=True, exist_ok=True)

    # Loop through individual dates, create directories, and apply functions.
    for single_date in return_daterange(start_date, end_date):
        DATEDIR, RAWDIR, AGGSCANDIR, AGGDIR = create_date_directories(single_date.strftime("%Y%m%d"))
        # print(RAWDIR)

        # Make temp directory and close when done
        templocation = tempfile.mkdtemp()
        results = download_raw(single_date, TOWER, 'US/Pacific')

        for i, scan in enumerate(results.iter_success(), start=1):
            file = scan.scan_time.strftime('%Y%m%d_%H%M')
            try:
                download_reflectivity(scan, file, RAWDIR)
                download_velocity(scan, file, RAWDIR)
                download_differential_phase(scan, file, RAWDIR)
                download_differential_reflectivity(scan, file, RAWDIR)
                download_cross_correlation(scan, file, RAWDIR)
                download_spectrum_width(scan, file, RAWDIR)
                print(file)

                os.chdir(RAWDIR)

                file_list = []

                for i in os.listdir(RAWDIR):

                    # match scenes from same time and combine in file.
                    if os.path.isfile(os.path.join(RAWDIR, i)) and str(file) in i:
                        file_list.append(i)
                file_list.sort()
                print(file_list)

                # Read metadata of first file
                with rio.open(file_list[0]) as src0:
                    meta = src0.meta

                # print('read metadata')

                # Update meta to reflect the number of layers
                meta.update(count=len(file_list))

                # print('updated metadata')
                print(file_list)

                # Read each layer and write it to stack
                with rio.open(os.path.join(str(AGGSCANDIR) + '/' + str(file) + '.tif'), 'w', **meta) as dst:
                    for id, layer in enumerate(file_list, start=1):
                        # print(layer)
                        with rio.open(layer) as src1:
                            dst.write_band(id, src1.read(1))

                # file.close()

            except:
                print(file + ' did not work')
                # file.close()

        shutil.rmtree(templocation)

        # os.chdir(YEARDIR)
