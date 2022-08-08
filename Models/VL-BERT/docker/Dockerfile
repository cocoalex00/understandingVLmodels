# VL-BERT container                                                                              #
#	This Dockerfile pulls from the anaconda image and installs the necessary requirements to:      #
#		- Run the VL-BERT code                                                                       #
##################################################################################################

ARG CUDA="10.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git wget curl ca-certificates bzip2 cmake tree htop bmon iotop g++ libgl1-mesa-glx libgtk2.0-dev libsm6 libxext6 libxrender1 

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /Miniconda3-latest-Linux-x86_64.sh \
 && /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda \
 && rm /Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
#RUN /miniconda/bin/conda install -y conda-build \
RUN /miniconda/bin/conda create -y --name py36 python=3.7\
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

COPY requirements.txt .

RUN pip install -r requirements.txt
# Install torch, APEX and other requirements
RUN pip install torch torchvision torchaudio


RUN pip install Cython


RUN apt-get install -y ninja-build 

# Clone the VL-BERT repo and install it 
COPY github/ /VL-BERT



RUN cd /VL-BERT \
	&& ./scripts/init.sh


