# # Establish place

import os
import pathlib
import tempfile
from datetime import datetime, date
import rasterio as rio
import shutil
from utils import *


def download(date_start, date_end, TOWER, hours, start_time):
    start_date = datetime.strptime(date_start, '%Y-%m-%d').date()
    print(start_date)
    end_date = datetime.strptime(date_end, '%Y-%m-%d').date()
    # Loop through individual dates, create directories, and apply functions.
    for single_date in return_daterange(start_date, end_date):
        prospective_datedir = DOPPLER_DIR.joinpath(single_date.strftime("%Y%m%d"))

        if prospective_datedir.exists():
            print(f"Data directory for {single_date.strftime('%Y%m%d')} already exists. Skipping download step.")
            continue

        DATEDIR, RAWDIR, AGGSCANDIR, AGGDIR = create_date_directories(DOPPLER_DIR, single_date.strftime("%Y%m%d"))
    #
    #     # Check if the expected data files for this date are present in AGGSCANDIR
    #     expected_files_pattern = f"{single_date.strftime('%Y%m%d')}_*.tif"
    #     expected_files = list(AGGSCANDIR.glob(expected_files_pattern))
    #
    #     if expected_files:
    #         print(f"Data files for {single_date.strftime('%Y%m%d')} already present. Skipping download step.")
    #         continue

        # Make temp directory and close when done
        templocation = tempfile.mkdtemp()
        results = download_raw(single_date, TOWER, 'US/Pacific', templocation, hours, start_time)

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
                    if (RAWDIR / i).is_file() and str(file) in i:
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
    return DOPPLER_DIR

