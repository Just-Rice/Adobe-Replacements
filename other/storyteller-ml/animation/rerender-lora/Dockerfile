FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

RUN sh Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /usr/local/miniconda
ENV PATH /usr/local/miniconda/bin:$PATH

RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install any needed packages specified in requirements.txt
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "rerender", "/bin/bash", "-c"]

RUN conda run -n rerender python install.py

# Run python script when the container launches
ENTRYPOINT ["conda", "run", "-n", "rerender", "python", "rerender.py", "--cfg", "config/real2sculpture.json"]
