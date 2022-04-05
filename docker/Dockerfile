FROM ubuntu:latest
WORKDIR /opt


# install gcc, python3, pip, wget, curl, git, gsl, slurm:
RUN \
    apt-get update        && \
    apt-get install --yes  --no-install-recommends  \
        build-essential      \
        gfortran             \
        python3-dev          \
        python3-pip          \
        wget                 \
        curl                 \
        git                  \
        libgsl-dev        && \
    apt-get clean all
RUN ln -s /usr/bin/python3 /usr/bin/python


# install MPI with MPICH (what we need in nersc) and mpi4py:
ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich
RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix
RUN /sbin/ldconfig
RUN python -m pip install mpi4py


# install python package to run fastpm:
RUN python -m pip install Cython
RUN python -m pip install numpy scipy matplotlib pandas astropy
RUN python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[mpi,jackknife,corrfunc]
RUN python -m pip install git+https://github.com/cosmodesi/pypower
RUN python -m pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class]
RUN python -m pip install pfft-python pmesh bigfile nbodykit
RUN python -m pip install psutil


# to reinstall fastpm-python increase the value of ARG
ARG nn=4
RUN python -m pip install git+https://github.com/echaussidon/fastpm-python
