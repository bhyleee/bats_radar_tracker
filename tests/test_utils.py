import pytest
import pathlib
from datetime import datetime, date, timedelta
from ..scripts.utils import *

def test_create_date_directories(tmp_path):
    DOPPLER_DIR = tmp_path
    single_date = "20231020"

    DATEDIR, RAWDIR, AGGSCANDIR, AGGDIR = create_date_directories(DOPPLER_DIR, single_date)

    assert DATEDIR.exists()
    assert RAWDIR.exists()
    assert AGGSCANDIR.exists()
    assert AGGDIR.exists()


def test_return_daterange():
    start_date = date(2023, 10, 1)
    end_date = date(2023, 10, 5)
    dates = list(return_daterange(start_date, end_date))

    expected_dates = [date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3), date(2023, 10, 4)]

    assert dates == expected_dates


def test_data_already_downloaded(tmp_path):
    assert not data_already_downloaded(tmp_path)

    # Create a dummy file
    (tmp_path / "dummy.txt").write_text("test")

    assert data_already_downloaded(tmp_path)