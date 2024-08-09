# BATS: Bat-aggregated time series
# Doppler Radar Data Pipeline

BATS is a Python-based algorithm that identifies Mexican free-tailed bats in weather radar data. BATS downloads, processes, and classifies large amounts of NOAA NEXRAD Weather Radar data using a pre-trained neural network.

Background

Mexican free-tailed bats (commonly referred to as Brazillian free-tailed bats) are a common bat species found across much of North and South America. Due to their voracious appetite and large roosting numbers, free-tailed bats are believed to provide invaluable ecosystem services in the form of pest control.

This project highlights a computer vision algorithm based on an artificial neural network that quantifies the occurence of free-tailed bats over a given area within a given time frame.

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

- Python 3.10.8
- Conda (recommended for environment management)
- The program was created using a linux-based Macbook Pro ARM64 processor

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git XXXXXXX
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

   Example:
    ```bash
    python main.py 2023-06-01 2023-06-01 KDAX --hours 1 --start_time 2000
    ```

2. Other Functionalities:
...

## Dataset
1. The reference data used for model training and testing are hosted on in the data/reference directory hosted on Zenodo.
2. The radar data is sourced from the NOAA NEXRAD data repository and accessed through the Python packages Py-ART and NEXRADAWS.
For the purposes of demonstration, a sample dataset is included in the data/ directory hosted on Zenodo.


## Model Information

Various popular machine learning algorithms were considered for this purpose, including Support Vector Machines, Random Forest, and Neural Networks. Ultimately the neural network was chosen for the final model. These exploratory model may be trained and tested in the attached jupyter notebooks.

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
Contributions are welcome! Please reach out here on github or via email at XXXX

## License
This project is licensed under the MIT License.

## Acknowledgements
- Thanks to Py-ART, an essential library used in this project.
- Appreciation to the team or individuals who contributed to this project.
