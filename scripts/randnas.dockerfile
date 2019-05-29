FROM nvidia/cuda:9.0-cudnn7-runtime
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV DATA_PATH=/data
ENV PATH=/usr/local/cuda-9.0/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

RUN apt-get update
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim 

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    pyenv install 3.5.5 && pyenv global 3.5.5 && pyenv rehash

RUN pip install torch==0.3.1 -f https://download.pytorch.org/whl/cu90/stable
RUN pip install torchvision==0.2.0 
RUN pip install boto3

RUN git clone https://github.com/liamcli/randomNAS_release.git /opt/randomNAS
RUN git clone https://github.com/liamcli/darts.git /opt/darts_fork

COPY run_stage1.sh /
COPY run_stage2.sh /

RUN mkdir /results
