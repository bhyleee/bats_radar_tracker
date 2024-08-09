import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from ..scripts.classify import normal_data, classify

# Assuming that you might have these constants in your classify.py. If not, adjust accordingly.
MODELS_DIR = 'path_to_models_dir'
DATA_DIR = 'path_to_data_dir'


# Create a fixture for dummy data
@pytest.fixture
def dummy_data():
    data = {
        'date': ['2023-01-01', '2023-01-02'],
        'cor': [0.1, 0.1, 0.2, 0.2],
        'pha': [0.1, 0.1, 0.2, 0.2],
        'dif': [0.1, 0.1, 0.2, 0.2],
        'ref': [0.1, 0.1, 0.2, 0.2],
        'spw': [0.1, 0.1, 0.2, 0.2],
        'vel': [0.1, 0.1, 0.2, 0.2],
        'training_class': [10, 11, 12, 13]
    }
    df = pd.DataFrame(data)
    return df


# Test for normal_data function
def test_normal_data(tmp_path, dummy_data):
    # tmp_path is a pytest fixture that provides a temporary directory unique to the test invocation
    csv_path = tmp_path / "dummy_data.csv"
    dummy_data.to_csv(csv_path)

    normalizer = normal_data(csv_path)
    assert isinstance(normalizer, tf.keras.layers.Normalization)


# Test for classify function
@patch('os.path.exists', return_value=False)  # Mock exists to always return False
@patch('classify.classify_image')  # Mock classify_image so we don't actually call it
def test_classify(mock_classify_image, mock_exists, tmp_path):
    # Setting up dummy directory structure
    rootdir = tmp_path / "root"
    scan_dir = rootdir / "2_scan_agg"
    scan_dir.mkdir(parents=True)
    tif_file = scan_dir / "sample.tif"
    tif_file.write_text("dummy_tif_content")  # Add some dummy content

    classify_dir = tmp_path / "classify"

    classify(rootdir, classify_dir)

    # Ensure classify_image was called once
    mock_classify_image.assert_called_once()
