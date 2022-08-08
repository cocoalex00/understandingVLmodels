FROM koallen/anaconda:gpu
MAINTAINER mi padre

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y git libprotobuf-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libleveldb-dev libsnappy-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

COPY ./bottom-up-attention /opt/butd

ENV CAFFE_ROOT=/opt/butd/caffe
WORKDIR $CAFFE_ROOT

# Compile Caffe
WORKDIR $CAFFE_ROOT
#ADD Makefile.config $CAFFE_ROOT/Makefile.config
RUN make all -j$(nproc)
RUN make test -j$(nproc)

# Set environment variables
ENV PYTHONPATH $CAFFE_ROOT/python:$PYTHONPATH
ADD anaconda.conf /etc/ld.so.conf.d/anaconda.conf
RUN ldconfig

# Add Python support
WORKDIR $CAFFE_ROOT/python
RUN pip install -r requirements.txt
WORKDIR $CAFFE_ROOT
RUN make pycaffe -j$(nproc)

# idk why the numpy and cloudpickle modules installed in the base image seem to break so they need to be reinstalled  
RUN conda remove -y numpy cloudpickle

# Install a text editor in case it's needed
RUN apt-get install -y vim 

# Upgrade pip and install the last dependencies to run the Faster R-CNN code
RUN pip install --upgrade "pip < 20.1" \
    && pip install opencv-python==3.4.0.14 cython easydict scikit-image protobuf==3.17.3 numpy
  
# Build the butd repository 
RUN cd /opt/butd/lib \
    && make

WORKDIR /
# The base image runs a jupyter notebook so the CMD command needs to be overwritten to run a bash session
CMD ["bash"]
