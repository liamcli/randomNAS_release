FROM nvidia/cuda:9.0-cudnn7-runtime

# Setup Ubuntu
RUN apt-get update --yes
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim 

# Get Miniconda and make it the main Python interpreter
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n pytorch_env python=3.5
RUN echo "source activate pytorch_env" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch_env
RUN conda install pytorch=0.3.1 torchvision=0.2.0 cuda90 -c pytorch
RUN conda install boto3

COPY run_stage1.sh /
COPY run_stage2.sh /

RUN mkdir /results

