# FROM nvidia/cuda:9.0-cudnn7-devel
FROM nvidia/cuda:10.0-cudnn7-devel

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_MXNET=1
ENV BYTESCHEDULER_WITHOUT_PYTORCH=1
# ENV MXNET_ROOT=/root/incubator-mxnet
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV http_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV https_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV ftp_proxy=http://proxy.cse.cuhk.edu.hk:8000/
ENV gopher_proxy=http://proxy.cse.cuhk.edu.hk:8000/

WORKDIR /root

# Install dev tools
RUN apt-get update && apt-get install -y git python-dev build-essential
RUN apt-get install -y wget && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
# RUN apt-get install apt-get install software-properties-common

# Install gcc 4.8
RUN apt -y install gcc-4.8
RUN apt -y install g++-4.8

# Pin GCC to 4.8 (priority 200) to compile correctly against Tensorflow.
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.8 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.8 200

# Bazel
RUN apt -y install unzip zip
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-installer-linux-x86_64.sh
RUN chmod +x bazel-0.19.2-installer-linux-x86_64.sh
RUN ./bazel-0.19.2-installer-linux-x86_64.sh

# Pip dependencies
RUN pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'
RUN pip install -U --user keras_applications --no-deps
RUN pip install -U --user keras_preprocessing --no-deps
RUN pip install --upgrade enum34

# Build and install
RUN git clone --branch r1.13 https://github.com/tensorflow/tensorflow.git
# RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git
RUN git clone --branch dev --recursive https://github.com/tonitick/byteps.git
RUN cp byteps/bytescheduler/bytescheduler/tensorflow/tf.patch tensorflow/ && cd tensorflow && echo "" >> tf.patch && git apply tf.patch
RUN cd tensorflow && ./configure bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
RUN cd tensorflow && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip install /tmp/tensorflow_pkg/tensorflow-1.13.2-cp27-cp27mu-linux_x86_64.whl
RUN cd byteps/bytescheduler/bytescheduler/tensorflow/ && make

# Benchmark
RUN git clone --branch cnn_tf_v1.13_compatible https://github.com/tensorflow/benchmarks.git
RUN cp byteps/bytescheduler/examples/tensorflow/benchmarks.patch benchmarks/
RUN cd benchmarks && git apply benchmarks.patch
RUN cp byteps/bytescheduler/bytescheduler/tensorflow/libplugin.so /root/benchmarks/scripts/tf_cnn_benchmarks

# Examples
WORKDIR /root/benchmarks/scripts/tf_cnn_benchmarks

# Run
RUN python tf_cnn_benchmarks.py --num_gpus=4 --batch_size=32 --model=resnet50 --variable_update=parameter_server
