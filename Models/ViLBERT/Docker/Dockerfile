# ViLBERT container                                                                              #
#	This Dockerfile pulls from the anaconda image and installs the necessary requirements to:      #
#		- Run the ViLBERT code                                                                       #
#		- Run the Faster R-CNN code to extract the image features                                    #
##################################################################################################

ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

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
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib tqdm

# Install PyTorch
RUN conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch \
 && conda clean -ya
RUN pip install pillow==6.1
#RUN conda install -c conda-forge opencv=4.1.0

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

RUN apt-get install -y  gcc

RUN cuda_dir="maskrcnn_benchmark/csrc/cuda" \
 && perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu \ 
 && git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac \
 && sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' setup.py \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git \
 && cd vqa-maskrcnn-benchmark \
 && sed -i -e 's/torch.cuda.is_available()/True/g' setup.py \
 && python setup.py build develop \
 && sed -i -e 's/True/torch.cuda.is_available()/g' setup.py 


# Install Vilbert 

WORKDIR /
RUN git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git

COPY requirements.txt /vilbert-multi-task/
RUN apt-get install -y libcap-dev
RUN cd vilbert-multi-task \ 
 && pip install -r requirements.txt \
 && python setup.py develop

# copy fixed code into container (adapted to python3)
COPY imports.py /vqa-maskrcnn_benchmark/utils/
COPY extract_features.py /vilbert-multi-task/script/
COPY ImageClassificationTrain.py /vilbert-multi-task/
COPY ImageClassificationDataset.py /vilbert-multi-task/vilbert/datasets/
COPY vilbertImageClassification.py /vilbert-multi-task/vilbert/
COPY __init__.py /vilbert-multi-task/vilbert/datasets/

RUN pip install --upgrade tensorboardX


RUN conda install opencv

