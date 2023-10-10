# Use an official Python runtime as the parent image
FROM python:3.10.8

WORKDIR /app

# Installing git
RUN apt-get update && apt-get install -y git wget

# Cloning your GitHub repository
RUN git clone https://github.com/bhyleee/bats_doppler_pipeline.git

# Switch to your repo's scripts directory
WORKDIR /app/bats_doppler_pipeline/scripts

# Fetch the Miniconda3 Linux installer
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh
RUN bash miniforge.sh -b -p $HOME/miniforge && \
    rm miniforge.sh

# Add Miniconda to PATH
ENV PATH="$HOME/miniconda/bin:$PATH"

# Debug: try to see the local space
RUN ls -la /app/bats_doppler_pipeline/scripts
RUN ls -la /app/bats_doppler_pipeline/

# Create the conda environment using the environment.yml file
RUN $HOME/miniforge/bin/conda env create -f environment.yml

# Activate the conda environment
SHELL ["conda", "run", "-n", "new_doppler_bats", "/bin/bash", "-c"]

# Switch to your repo's scripts directory
#WORKDIR /app/bats_doppler_pipeline/scripts

#let this work
# Your other Dockerfile commands, such as CMD or ENTRYPOINT, go here
COPY /data /app/data
COPY /models /app/models