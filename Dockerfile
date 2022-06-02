FROM nvidia/cudagl:11.4.0-devel-ubuntu18.04

# https://zenn.dev/flyingbarbarian/scraps/1275681132babd
ENV DEBIAN_FRONTEND noninteractive

# install zsh https://github.com/ohmyzsh/ohmyzsh#prerequisites
RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    wget curl git zsh
# SHELL ["/bin/zsh", "-c"]
# RUN curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh
SHELL ["/bin/bash", "-c"]

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /root
RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    sudo vim wget
    
WORKDIR /opt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

COPY environment.yml /user/fit2form/

WORKDIR /user/fit2form

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda env create -n fit2form -f environment.yml && \
    conda init && \
    echo "conda activate fit2form" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV fit2form && \
    PATH /opt/conda/envs/fit2form/bin:$PATH


# RUN pip install -U numpy && apt-get install freeglut3-dev

# X window ----------------
RUN apt-get update && apt-get install -y \
    xvfb x11vnc python-opengl icewm
RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 &; x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &; icewm-session &"' >> /root/.zshrc

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN pip install -U numpy


CMD ["/bin/bash"]