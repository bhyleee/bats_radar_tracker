import pytest
from ..scripts.utils import *
from unittest.mock import patch, MagicMock
import pathlib
import os

BASE_DIR = pathlib.Path(__file__).parent  # Assuming this test is at the same level as main.py
DIRECTORY_ABOVE_BASE = BASE_DIR.parent
DATA_DIR = DIRECTORY_ABOVE_BASE.joinpath('data')
TEST_DATE = "2023-01-01"
base_doppler_dir = DATA_DIR.joinpath('doppler', TEST_DATE)

@patch('main.download', return_value=base_doppler_dir.joinpath('mocked_directory_path'))
@patch('main.classify')
@patch('main.aggregate_all_classified_data')
@patch('main.datetime.today', return_value=MagicMock(strftime=lambda format: TEST_DATE))
def test_main(mock_datetime, mock_aggregate, mock_classify, mock_download):
    # Mocking the content of base_doppler_dir
    mock_dir_1 = MagicMock()
    mock_dir_1.is_dir.return_value = True
    mock_dir_1.name = 'run_1'

    with patch.object(base_doppler_dir, 'iterdir', return_value=[mock_dir_1]):
        # Dummy arguments for main
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        tower = "T001"
        hours = 5
        start_time = 1900

        # Call main
        main(start_date, end_date, tower, hours, start_time)

        # Assertions for directory creation
        assert base_doppler_dir.joinpath('run_2').exists()

        # Assertions for function calls
        mock_download.assert_called_with(start_date, end_date, tower, hours, start_time)
        mock_classify.assert_called_with(base_doppler_dir.joinpath('mocked_directory_path'),
                                         base_doppler_dir.joinpath('run_2'))
        mock_aggregate.assert_called_with(base_doppler_dir.joinpath('mocked_directory_path'),
                                          base_doppler_dir.joinpath('run_2'))

# Clean up after the test
@pytest.fixture(autouse=True)
def clean_up():
    yield  # This ensures that the cleanup code runs after the test finishes
    test_run_dir = base_doppler_dir.joinpath('run_2')
    if test_run_dir.exists():
        test_run_dir.rmdir()