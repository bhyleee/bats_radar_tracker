# BATS: Bat-aggregated time series
# Doppler Radar Data Pipeline

This project aims to download, process, and classify Doppler radar data. The pipeline consists of downloading raw radar scans, classifying the data using a trained model, and then aggregating the results.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Data Download**: Automates the download of radar scans for specified dates.
- **Classification**: Uses a trained model to classify radar scans.
- **Aggregation**: Processes and aggregates radar data for analysis.

## Prerequisites

- Python 3.x
- Conda (recommended for environment management)

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/doppler-radar-analysis.git
   cd doppler-radar-analysis
   
2. Set up the conda environment
    ```bash
    conda env create -f environment.yml
    conda activate doppler_env
   
3. Additional Setup
-Download the pretrained neural network (and other models if interested). 
These models are hosted in the same repository as the sample data (Zenodo)

## Usage
1. Downloading/Accessing
    ```bash
    python main.py [start_date] [end_date] [tower] --hours [hours] --start_time [start_time]
    ```
    Replace [start_date], [end_date], [tower], [hours], and [start_time] with your desired values.
2. Other Functionalities:
...

## Dataset
1. The reference data used for model training and testing are hosted on in the data/reference directory hosted on Zenodo.
2. The radar data is sourced from the NOAA NEXRAD data repository and accessed through the Python packages Py-ART and NEXRADAWS.
For the purposes of demonstration, a sample dataset is included in the data/ directory hosted on Zenodo.


## Model Information

Our classification task leverages a traditional Artificial Neural Network (ANN) constructed using the Keras Python Deep Learning package from Google’s Tensorflow library. The classifier processes input as a single pixel from the mentioned cartesian grid, essentially a six-dimensional vector of radar data.

Following the methods of Chilson et al. (2019) and Zewdie et al. (2019) – who utilized neural networks for tracking purple martins and pollen via NEXRAD respectively – we've structured our classifier as a feed-forward, fully-connected network. It's comprised of:

- A 6-unit input layer
- Three intermediate 152-unit layers with ReLU activation functions
- A concluding 2-unit layer with a SoftMax activation, outputting a scaled probability (between 0 and 1) signifying the likelihood of a pixel containing a bat swarm.

This architecture was chosen based on several metrics including precision, recall scores, AUC, and a qualitative assessment rooted in established bat dispersal strategies. The assessment was performed on a validation set extracted from the primary training set. For a deeper dive into the training methodology, refer to section 2.4.

The model underwent training for 20 epochs, utilizing mini-batches of size 32 and the Adam optimizer. The learning rate was pegged at 0.001, with Adam parameters set to the recommended defaults of `β1 = 0.9` and `β2 = 0.999`. The training was done semi-supervised, employing the standard cross-entropy loss function.

**Network Architecture**:

\[
f(x) = \mathrm{softmax}(W_3(\mathrm{relu}(W_2(\mathrm{relu}(W_1(\mathrm{relu}(W_0 x + b_0)) + b_1)) + b_2) + b_3)
\]

*Equation 1*: This represents the network's architectural flow, where `x` is the radar data input vector. The symbols `W0...W3` and `b0...b3` denote the network's trainable parameters.


## Contributing
Contributions are welcome! Please reach out here on github or via email at brianlee52@ucsb.edu

## License
This project is licensed under the MIT License.

## Acknowledgements
- Thanks to Py-ART, an essential library used in this project.
- Appreciation to the team or individuals who contributed to this project.