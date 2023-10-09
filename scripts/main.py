import argparse
from download import download
from classify import classify
from aggregate import aggregate_date
from utils import *
import os


def main(start_date, end_date, tower, hours):
    print("script is starting")
    print(start_date)
    # Step 1: Download data
    doppler_dir = download(start_date, end_date, tower, hours)

    # # Step 2: Classify the downloaded data
    # # Note: Modify this part if there's a specific directory or other parameters to provide.
    # classify_dir = doppler_dir.joinpath('..', 'classified')  # Adjust the path accordingly
    # classify_dir.mkdir(parents=True, exist_ok=True)
    # classify(classify_dir)
    #
    # # Step 3: Aggregate the classified data
    # # Note: Modify this part if there's a specific directory or other parameters to provide.
    # # current_directory = os.getcwd()  # Modify this if your data is in a different directory
    # aggregate_dir = doppler_dir.joinpath('..', 'aggregated')  # Adjust the path accordingly
    # aggregate_date(classify_dir, aggregate_dir)  # You might want to modify the year or parameterize it


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process radar data.')
    parser.add_argument('start_date', type=str, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('end_date', type=str, help='End date in the format YYYY-MM-DD')
    parser.add_argument('tower', type=str, help='The radar tower ID')
    parser.add_argument('hours', type=int, default=12, help='Number of hours (default is 12).')

    args = parser.parse_args()

    main(args.start_date, args.end_date, args.tower, args.hours)
