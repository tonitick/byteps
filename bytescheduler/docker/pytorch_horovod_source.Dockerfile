# FROM nvidia/cuda:9.0-cudnn7-devel
FROM nvidia/cuda:10.0-cudnn7-devel

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_PYTORCH=1
ENV BYTESCHEDULER_WITHOUT_MXNET=1
ENV http_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV https_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV ftp_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV gopher_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ARG HOROVOD_VERSION=b5cbf240909b467c348683c69df4d73f07147860

WORKDIR /root

# Install dev tools
RUN apt-get update && apt-get install -y git python-dev build-essential
RUN apt-get install -y wget && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
RUN apt install -y openmpi-bin libopenmpi-dev

# Clone repo
RUN git clone --branch v1.0.0 --recursive https://github.com/pytorch/pytorch
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git
RUN git clone --recursive https://github.com/horovod/horovod.git && \
    cd horovod && git reset --hard ${HOROVOD_VERSION}

# Install pytorch
RUN cd pytorch && python setup.py install

# Apply the patch and reinstall Horovod
RUN cp byteps/bytescheduler/bytescheduler/pytorch/horovod_pytorch.patch horovod/ && \
    cd horovod && git apply horovod_pytorch.patch && python setup.py install

# Install ByteScheduler
RUN pip install bayesian-optimization torchvision==0.2.2.post3 && cd byteps/bytescheduler && python setup.py install

# Examples
WORKDIR /root/byteps/bytescheduler/examples/

# # Run
# mpirun --allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters 100
