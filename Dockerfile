# Use an official Python runtime as the parent image
FROM python:3.10.8

WORKDIR /app

# Installing git
RUN apt-get update && apt-get install -y git

# Cloning your GitHub repository
RUN git clone https://github.com/bhyleee/bats_doppler_test.git

# Switch to your repo's scripts directory
WORKDIR /app/bats_doppler_test/scripts

# Fetch the Miniconda3 Linux installer
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Run the installer
RUN bash miniconda.sh -b -p $HOME/miniconda && \
    rm miniconda.sh

# Add Miniconda to PATH
ENV PATH="$HOME/miniconda/bin:$PATH"

# Create the conda environment using the environment.yml file
RUN conda env create -f environment.yml

# Activate the conda environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

#let this work
# Your other Dockerfile commands, such as CMD or ENTRYPOINT, go here
COPY /data /app/data
COPY /models /app/models