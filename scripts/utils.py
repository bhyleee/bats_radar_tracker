import pytz
import pyart
import nexradaws
import os
import numpy as np
import pandas as pd
import rasterio as rio
import pathlib
from datetime import datetime, date, timedelta
# import shutil
# import tempfile
# import tensorflow as tf
# from tensorflow import keras

BASE_DIR = pathlib.Path(__file__).parent.parent  # points to the 'scripts' directory
DIRECTORY_ABOVE_BASE = BASE_DIR.parent
MODELS_DIR = DIRECTORY_ABOVE_BASE.joinpath('models')
DATA_DIR = DIRECTORY_ABOVE_BASE.joinpath('data')
DOPPLER_DIR = DATA_DIR.joinpath('doppler')


def create_date_directories(single_date):
    """
    Creates necessary directories for a given date.

    Parameters:
    - single_date (str): Date string formatted as "YYYYMMDD".

    Returns:
    - tuple: A tuple containing paths for the main date directory, raw data directory, scan aggregate directory, and daily aggregate directory.
    """
    DATEDIR = DOPPLER_DIR.joinpath(str(single_date))
    DATEDIR.mkdir(parents=True, exist_ok=True)
    RAWDIR = DOPPLER_DIR.joinpath(str(single_date), '/1_raw/')
    RAWDIR.mkdir(parents=True, exist_ok=True)
    AGGSCANDIR = DOPPLER_DIR.joinpath(str(single_date), '/2_scan_agg/')
    AGGSCANDIR.mkdir(parents=True, exist_ok=True)
    # CLASSDIR = pathlib.Path(DATA_DIR + '/' + str(single_date) + '/3_daily_class/')
    # CLASSDIR.mkdir(parents=True, exist_ok=True)
    AGGDIR = DOPPLER_DIR.joinpath(str(single_date), '/3_daily_agg/')
    AGGDIR.mkdir(parents=True, exist_ok=True)

    return DATEDIR, RAWDIR, AGGSCANDIR, AGGDIR


def return_daterange(start_date, end_date):
    """
    Generator that yields dates in a range from start_date to end_date.

    Parameters:
    - start_date (date): The starting date.
    - end_date (date): The ending date.

    Yields:
    - date: A date between start_date and end_date.
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def download_raw(start_date, tower, time_zone, templocation, hours):
    """
    Downloads raw weather radar data for a given date and tower.

    Parameters:
    - start_date (date): Date for which data is to be downloaded.
    - tower (str): The radar tower ID.
    - time_zone (str): Time zone of the radar data.

    Returns:
    - results (obj): A collection of downloaded scan results.
    """
    # This function downloads
    start_date = start_date.strftime("%Y%m%d")
    start_time = '1800'
    startdatetime = "".join((start_date, start_time))
    start_date1 = datetime.strptime(startdatetime, "%Y%m%d%H%M")

    # download entire evening's worth of data
    end_date = start_date1 + timedelta(hours=hours)

    conn = nexradaws.NexradAwsInterface()
    # Download data
    central_timezone = pytz.timezone(time_zone)
    radar_id = tower
    start = central_timezone.localize(start_date1)
    end = central_timezone.localize(end_date)
    scans = conn.get_avail_scans_in_range(start, end, radar_id)
    # print("There are {} scans available between {} and {}\n".format(len(scans), start, end))

    results = conn.download(scans, templocation)

    # for scan in results.iter_success():
    # print ("{} volume scan time {}".format(scan.radar_id,scan.scan_time))

    return results


def download_reflectivity(scan, file, output_directory):
    # Add reflectivity raster
    radar_reflect = scan.open_pyart()

    # mask out last 10 gates of each ray, this removes the "ring" around th radar.
    radar_reflect.fields['reflectivity']['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter_reflect = pyart.filters.GateFilter(radar_reflect)
    gatefilter_reflect.exclude_transition()
    gatefilter_reflect.exclude_masked('reflectivity')
    grid_reflect = pyart.map.grid_from_radars(
        radar_reflect,
        gatefilters=(gatefilter_reflect,),
        grid_shape=(1, 2000, 2000),
        grid_limits=((0, 1500.0), (-70000, 70000), (-70000, 70000)),
        fields=['reflectivity'],
        gridding_algo="map_gates_to_grid",
        weighting_function='NEAREST',
        roi_func='constant_roi', constant_roi=2000.0)

    output_path = os.path.join(str(output_directory) + '/' + str(file) + '_reflectivity.tif')

    output_file = pyart.io.write_grid_geotiff(grid_reflect, output_path, 'reflectivity', rgb=False, warp=True,
                                              cmap='pyart_NWSRef', vmin=0, vmax=100, level=0)
    return output_file


def download_velocity(scan, file, output_directory):
    # Add velocity raster
    radar_velocity = scan.open_pyart()

    # mask out last 10 gates of each ray, this removes the "ring" around th radar.
    radar_velocity.fields['velocity']['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter_velocity = pyart.filters.GateFilter(radar_velocity)
    gatefilter_velocity.exclude_transition()
    gatefilter_velocity.exclude_masked('velocity')
    grid_velocity = pyart.map.grid_from_radars(
        radar_velocity,
        gatefilters=(gatefilter_velocity,),
        grid_shape=(1, 2000, 2000),
        grid_limits=((0, 1500.0), (-70000, 70000), (-70000, 70000)),
        fields=['velocity'],
        roi_func='constant_roi', constant_roi=2000.0, weighting_function='Nearest')

    output_path = os.path.join(str(output_directory) + '/' + str(file) + '_velocity.tif')

    output_file = pyart.io.write_grid_geotiff(grid_velocity, output_path, 'velocity', rgb=False, warp=True,
                                              cmap='pyart_NWSRef', vmin=-70, vmax=70, level=0)

    return output_file


def download_differential_phase(scan, file, output_directory):
    ### Add differential phase raster
    radar_differential_phase = scan.open_pyart()

    # mask out last 10 gates of each ray, this removes the "ring" around th radar.
    radar_differential_phase.fields['differential_phase']['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter_differential_phase = pyart.filters.GateFilter(radar_differential_phase)
    gatefilter_differential_phase.exclude_transition()
    gatefilter_differential_phase.exclude_masked('differential_phase')
    grid_differential_phase = pyart.map.grid_from_radars(
        radar_differential_phase,
        gatefilters=(gatefilter_differential_phase,),
        grid_shape=(1, 2000, 2000),
        grid_limits=((0, 1500.0), (-70000, 70000), (-70000, 70000)),
        fields=['differential_phase'],
        roi_func='constant_roi', constant_roi=2000.0, weighting_function='Nearest')

    output_path = os.path.join(str(output_directory) + '/' + str(file) + '_differential_phase.tif')

    output_file = pyart.io.write_grid_geotiff(grid_differential_phase, output_path, 'differential_phase', rgb=False,
                                              warp=True,
                                              cmap='pyart_NWSRef', vmin=-10, vmax=150, level=0)

    return output_file


def download_differential_reflectivity(scan, file, output_directory):
    ### Add differential reflectivity raster
    radar_differential_reflectivity = scan.open_pyart()

    # mask out last 10 gates of each ray, this removes the "ring" around th radar.
    radar_differential_reflectivity.fields['differential_reflectivity']['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter_differential_reflectivity = pyart.filters.GateFilter(radar_differential_reflectivity)
    gatefilter_differential_reflectivity.exclude_transition()
    gatefilter_differential_reflectivity.exclude_masked('differential_reflectivity')
    grid_differential_reflectivity = pyart.map.grid_from_radars(
        radar_differential_reflectivity,
        gatefilters=(gatefilter_differential_reflectivity,),
        grid_shape=(1, 2000, 2000),
        grid_limits=((0, 1500.0), (-70000, 70000), (-70000, 70000)),
        fields=['differential_reflectivity'],
        roi_func='constant_roi', constant_roi=2000.0, weighting_function='Nearest')

    output_path = os.path.join(str(output_directory) + '/' + str(file) + '_differential_reflectivity.tif')

    output_file = pyart.io.write_grid_geotiff(grid_differential_reflectivity, output_path, 'differential_reflectivity',
                                              rgb=False, warp=True,
                                              cmap='pyart_NWSRef', vmin=-20, vmax=20, level=0)


def download_cross_correlation(scan, file, output_directory):
    ### Add cross correlation raster
    radar_cross_correlation_ratio = scan.open_pyart()

    # mask out last 10 gates of each ray, this removes the "ring" around th radar.
    radar_cross_correlation_ratio.fields['cross_correlation_ratio']['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter_cross_correlation_ratio = pyart.filters.GateFilter(radar_cross_correlation_ratio)
    gatefilter_cross_correlation_ratio.exclude_transition()
    gatefilter_cross_correlation_ratio.exclude_masked('cross_correlation_ratio')
    grid_cross_correlation_ratio = pyart.map.grid_from_radars(
        radar_cross_correlation_ratio,
        gatefilters=(gatefilter_cross_correlation_ratio,),
        grid_shape=(1, 2000, 2000),
        grid_limits=((0, 1500.0), (-70000, 70000), (-70000, 70000)),
        fields=['cross_correlation_ratio'],
        roi_func='constant_roi', constant_roi=2000.0, weighting_function='Nearest')

    output_path = os.path.join(str(output_directory) + '/' + str(file) + '_cross_correlation_ratio.tif')

    output_file = pyart.io.write_grid_geotiff(grid_cross_correlation_ratio, output_path, 'cross_correlation_ratio',
                                              rgb=False, warp=True,
                                              cmap='pyart_NWSRef', vmin=0, vmax=2, level=0)

    return output_file


def download_spectrum_width(scan, file, output_directory):
    ### Add spectrum width raster
    radar_spectrum_width = scan.open_pyart()

    # mask out last 10 gates of each ray, this removes the "ring" around th radar.
    radar_spectrum_width.fields['spectrum_width']['data'][:, -10:] = np.ma.masked

    # exclude masked gates from the gridding
    gatefilter_spectrum_width = pyart.filters.GateFilter(radar_spectrum_width)
    gatefilter_spectrum_width.exclude_transition()
    gatefilter_spectrum_width.exclude_masked('spectrum_width')
    grid_spectrum_width = pyart.map.grid_from_radars(
        radar_spectrum_width,
        gatefilters=(gatefilter_spectrum_width,),
        grid_shape=(1, 2000, 2000),
        grid_limits=((0, 15000.0), (-70000, 70000), (-70000, 70000)),
        fields=['spectrum_width'],
        roi_func='constant_roi', constant_roi=2000.0, weighting_function='Nearest')

    output_path = os.path.join(str(output_directory) + '/' + str(file) + '_spectrum_width.tif')

    output_file = pyart.io.write_grid_geotiff(grid_spectrum_width, output_path, 'spectrum_width', rgb=False, warp=True,
                                              cmap='pyart_NWSRef', vmin=0, vmax=20, level=0)

    return output_file


def classify_image(new_image, file, model, normalizer, classdir):
    """
    Classifies a radar image using a machine learning model.

    Parameters:
    - new_image (str): Path to the radar image to be classified.
    - file (str): Name of the radar image file.
    - model (tf.Model): Trained TensorFlow model for classification.
    - normalizer (tf.keras.layers.Layer): Normalization layer for pre-processing.
    - classdir (str): Directory where classified images are saved.

    Returns:
    - None: The function saves the classified image directly to the classdir.
    """

    new_image = os.path.abspath(new_image)

    features1 = rio.open(new_image)

    cor_features = features1.read(1).flatten()
    pha_features = features1.read(2).flatten()
    dif_features = features1.read(3).flatten()
    ref_features = features1.read(4).flatten()
    spw_features = features1.read(5).flatten()
    vel_features = features1.read(6).flatten()

    # ds3, featuresimage2_scratch = raster.read(new_image, bands='all')

    stacked_features = np.column_stack(
        (cor_features, pha_features, dif_features, ref_features, spw_features, vel_features))
    df = pd.DataFrame(stacked_features, columns=['cor', 'pha', 'dif', 'ref', 'spw', 'vel'])

    variables = ['cor', 'pha', 'dif', 'ref', 'spw', 'vel']
    numeric_features = df[variables]
    numeric_features_array = numeric_features.to_numpy()

    # numeric_features_array = scaler.fit_transform(numeric_features_array)

    # normalize features
    # normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    # normalizer.adapt(numeric_features_array)
    numeric_features_array_norm = normalizer(numeric_features_array)

    predicted = model.predict(numeric_features_array_norm, batch_size=1000, workers=10, use_multiprocessing=True)
    # predicted = predicted[:,1]

    # Export raster
    prediction = np.reshape(predicted, (features1.height, features1.width))

    # Convert values less than 0.7 prob to 0
    prediction[np.abs(prediction) < 0.9] = 0

    outFile = str(classdir) + '/' + 'classified_' + file

    with rio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = features1.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')

        with rio.open(outFile, 'w', **profile) as dst:
            dst.write(prediction.astype(rio.float32), 1)


def aggregate_date(data_directory, output_directory):
    """
    Aggregates radar data for a specific date.

    Parameters:
    - data_directory (str): Directory containing the radar data.
    - date (int): Year of the data to be aggregated.

    Returns:
    - None: The function saves aggregated data directly to the data_directory.
    """
    merge_dir = []
    for filename in os.listdir(data_directory):
        if filename.endswith('.tif') and filename.startswith('classified'):
            merge_dir.append(filename)

        elif filename.startswith('2018'):
            merge_dir.append(filename)

        else:
            pass

    # Read metadata of first file
    merge_dir.sort()
    with rio.open(merge_dir[0]) as src0:
        meta = src0.meta
        arr1 = src0.read()
        print(arr1.shape)

    # Create aggregate file
    aggregate_file = output_directory + '_new_aggregate.tif'
    with rio.open(aggregate_file, 'w', **meta) as dst:
        dst.write(arr1)

    for id, layer in enumerate(merge_dir, start=1):
        # try:
        if layer.endswith('.tif'):
            print('layer is ' + layer)
            # print('id is ' + id)
            with rio.open(layer) as src1:
                array1 = src1.read()
                print(array1.shape)
                # print('src 1 is ' + src1)
                # with rasterio.open('test.tif', 'w', **meta) as dst:
                array = rio.open(aggregate_file)
                print('aggregate file is ' + aggregate_file)
                array0 = array.read(1)
                # array0 = np.where(array0 > 0.1, 1, 0)

                # array2 = np.where(array1 > 0.1, 1, 0)
                array1[np.isnan(array1)] = 0

                new_array = np.add(array0, array1)
                print('added files')
                array.close()

                with rio.open(aggregate_file, 'w', **meta) as dst:
                    dst.write(new_array)
        else:
            pass
