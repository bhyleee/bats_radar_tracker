import argparse

from tools.classify import classify
from tools.download import download
# from aggregate import aggregate_date
from tools.utils import *


def main(start_date, end_date, tower, hours, start_time):
    print("script is starting")
    start_date_str = start_date.replace("-", "")
    # Step 1: Download data
    # Path to where you expect the downloaded data to reside
    expected_data_dir = DOPPLER_DIR.joinpath(str(start_date_str)) # Modify accordingly if different
    # #
    # Step 1: Download data
    if not data_already_downloaded(expected_data_dir):
        doppler_dir = download(start_date, end_date, tower, hours, start_time)
    else:
        print(f"Data for {start_date} to {end_date} already downloaded. Skipping download step.")
        doppler_dir = expected_data_dir

    # doppler_dir = download(start_date, end_date, tower, hours, start_time)

    # Step 2: Classify the downloaded data
    # Note: Modify this part if there's a specific directory or other parameters to provide.
    classify_dir = DOPPLER_DIR.joinpath('classified')  # Adjust the path accordingly
    classify_dir.mkdir(parents=True, exist_ok=True)
    classify(doppler_dir, classify_dir)

    # Step 3: Aggregate the classified data
    # Note: Modify this part if there's a specific directory or other parameters to provide.
    # current_directory = os.getcwd()  # Modify this if your data is in a different directory
    aggregate_dir = DOPPLER_DIR.joinpath('aggregated')  # Adjust the path accordingly
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_all_classified_data(classify_dir, aggregate_dir)  # You might want to modify the year or parameterize it


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process radar data.')
    parser.add_argument('start_date', type=str, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('end_date', type=str, help='End date in the format YYYY-MM-DD')
    parser.add_argument('tower', type=str, default='KDAX', help='The radar tower ID')
    parser.add_argument('--hours', type=int, default=10, help='Number of hours (default is 12).')
    parser.add_argument('--start_time', type=int, default=2000, help='Start time default is 8pm (2000).')

    args = parser.parse_args()

    main(args.start_date, args.end_date, args.tower, args.hours, args.start_time)
