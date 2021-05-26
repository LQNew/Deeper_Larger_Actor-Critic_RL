# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       1.4.0  (pip)
# tensorflow    1.14.0 (pip)
# DMC Control Suite
# MuJoCo
# ==================================================================
FROM liqingya/py36-cu100:torch1.4-tensorflow1.14

ENV LANG C.UTF-8

# ==================================================================
# copy dependencies
# ------------------------------------------------------------------
COPY .tmux.conf /root/
COPY .zshrc /root/
COPY dependencies /root/dependencies

RUN cd /root/dependencies/ && \
    unzip /root/dependencies/mujoco200_linux.zip && \
    mkdir -p /root/.mujoco && \
    mv /root/dependencies/mujoco200_linux /root/.mujoco/mujoco200_linux && \
    cp -r /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 && \
    mv /root/dependencies/patchelf /usr/local/bin/patchelf && \
    chmod +x /usr/local/bin/patchelf && \
    unzip /root/dependencies/ohmyzsh-master.zip && \
    unzip /root/dependencies/zsh-autosuggestions-master.zip && \
    unzip /root/dependencies/zsh-syntax-highlighting-master.zip && \
    mv /root/dependencies/ohmyzsh-master /root/.oh-my-zsh && \
    mv /root/dependencies/zsh-autosuggestions-master /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    mv /root/dependencies/zsh-syntax-highlighting-master /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
COPY mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout 100" && \
    PIP_UNINSTALL="python -m pip uninstall -y" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        curl \
        cmake \
        sudo \
        g++ \
        htop \
        tmux \
        openssh-client \
        openssh-server \
        qt5-default \
        openssl1.0 \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-doc \
        libglib2.0-0 \
        libglfw3 \
        libglew2.0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libxrender1 \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        libevent-dev \
        locales \
        net-tools \
        qt5-default \
        xpra \
        xserver-xorg-dev \
        zsh \ 
        tzdata \
        && \
    # $GIT_CLONE https://github.com/ohmyzsh/ohmyzsh.git /root/.oh-my-zsh && \
    # $GIT_CLONE https://github.com/zsh-users/zsh-autosuggestions.git /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    # $GIT_CLONE https://github.com/zsh-users/zsh-syntax-highlighting.git /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    chsh -s /bin/zsh && \
    $PIP_INSTALL -i https://mirrors.aliyun.com/pypi/simple \
        pip \
        absl-py \
        Cython \
        ffmpeg \
        h5py \
        future \
        pyyaml \
        pyparsing \
        pandas \
        cffi \
        pillow \
        six \
        scipy \
        matplotlib \
        tqdm \
        mpi4py \
        termcolor \
        seaborn \
        tb-nightly \
        visdom \
        imageio \
        imageio-ffmpeg \
        pytest \
        munkres \
        psutil \
        joblib \
        glfw \
        scikit-image \
        box2d-py \
        gym \
        mujoco-py \
        && \
    cd /root/dependencies/ && \
    unzip /root/dependencies/dm_control-master.zip && \
    unzip /root/dependencies/dmc2gym-master.zip && \
    cd /root/dependencies/dm_control-master && \
    python setup.py install && \ 
    cd /root/dependencies/dmc2gym-master && \
    python setup.py install && \  
    cd /root/ && \
    rm -rf /root/dependencies && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
# ==================================================================
# config timezone
# ------------------------------------------------------------------
    sed -i -e "s/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/" /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -f /etc/localtime && \
    rm -rf /usr/share/zoneinfo/UTC && \
    dpkg-reconfigure --frontend=noninteractive tzdata
# # ==================================================================
# # enable all users using sudo 
# # ------------------------------------------------------------------
#     cp /etc/sudoers /etc/sudoers.new && \
#     echo "%docker ALL=(ALL:ALL) ALL" >> /etc/sudoers.new  && \
#     visudo -c -f /etc/sudoers.new && \
#     cp /etc/sudoers.new /etc/sudoers  && \
#     rm /etc/sudoers.new
# ==================================================================
# initialize conda
# ------------------------------------------------------------------
    # . /opt/conda/etc/profile.d/conda.sh && \
    # /bin/bash -c "source activate base"
    # source activate && \
    # conda activate base

EXPOSE 6006
CMD ["/bin/zsh"]