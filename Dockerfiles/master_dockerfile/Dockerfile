# Master container                                                                              
#	This Dockerfile pulls from the CUDA image and installs the necessary requirements to run the following models:      
#   - ViLBERT
#	- LXMERT
# 	- Oscar
#   - ViLT
#	- ALBEF
##################################################################################################

FROM  nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

COPY requirements.txt . 

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils vim git wget curl ca-certificates bzip2 cmake tree htop bmon iotop g++ libgl1-mesa-glx libgtk2.0-dev libsm6 libxext6 libxrender1 build-essential libpcap-dev \
 && apt-get install -y python-prctl

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /Miniconda3-latest-Linux-x86_64.sh \
 && /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda \
 && rm /Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7\
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

#RUN pip install --upgrade pip 
#RUN pip install torch torchvision torchaudio 
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

# idk why but numpy is broken when installed like that so I need to 
RUN pip uninstall -y numpy\
	&& pip install numpy==1.19.5

## ViLBERT
RUN git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git \
    && cd vilbert-multi-task \
    && python setup.py develop

# LXMERT does not need anything besides the requirements already installed
RUN git clone https://github.com/airsplay/lxmert.git 


# OSCAR does not need anything besides the requirements already installed
RUN git clone https://github.com/microsoft/Oscar.git \ 
    && cd Oscar \ 
    && python setup.py build develop


# ViLT
RUN git clone https://github.com/dandelin/ViLT.git \ 
    && cd ViLT \ 
    && pip install -e .

#ALBEF 
RUN git clone https://github.com/salesforce/ALBEF.git


