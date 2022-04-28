# How to use docker:

## 0. Remark and tips:

* Change the value of nn to recompile only the new version of fastpm-python

* **Remark:** to install linux package with geographic location, you need to disable the question adding : `ENV DEBIAN_FRONTEND noninteractive`

* Compilation of MPICH 3.3 (do not know if the problem still exist with higher version): You need to add the flag  `FFLAGS "-w -fallow-argument-mismatch -O2"` with the new version of gfortran (10 ?) (see here: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=91731) to avoid error from functions that take 'void *' argument in MPICH.

## 1. Why do we want to use docker ?

Launching python on a large number of nodes can take a lot of time. The solution at NERSC is to use shifter to reduce the launching time (~0s.). See the official documentation:

    * https://docs.nersc.gov/development/languages/python/parallel-python/
    * https://docs.nersc.gov/development/languages/python/python-shifter/
    * https://docs.nersc.gov/development/shifter/how-to-use/

What we need to do:

    * Create a docker image
    * download it at NERSC
    * Use shifter command to use the container

## 2. How to build a docker image ?

    * To build the image (warning installing MPICH is quite long) run `docker build -t edmondchau/fastpm-python .`
    * To add a tag do `docker tag edmondchau/fastpm-python edmondchau/fastpm-python:tagada`. Note you can direclty do `docker build -t edmondchau/fastpm-python:tagada .` by default the tag is `latest`.
    * To upload the image in docker hub run `docker push edmondchau/fastpm-python` (`:tagada` --> `:latest` by default) (the repo edmondchau/fastpm-python is already created before pushing)

## 3. How to use the image in your computer ?

    * Simply run: `docker run -it edmondchau/fastpm-python /bin/bash`

## 3. How to use the image at NERSC ?

    * First download the image `shifterimg -v pull edmondchau/fastpm-python:latest` and check if it is available with `shifterimg images` (for private image see NERSC documentation above)
    * **WARNING:** **before** to use shifter, you need to erase the current `PYTHONPATH`: `export PYTHONPATH='' && echo $PYTHONPATH`
    * Test in interactive mode:
        * logging node : `shifter --image=edmondchau/fastpm-python:latest /bin/bash`
        * interactive node (haswell) : `salloc -N 1 -C haswell --qos interactive -L SCRATCH,project --image=edmondchau/fastpm-python:latest -t 00:30:00 shifter /bin/bash`
        * Run a script is done remplacing /bin/bash by ./script.sh
        * Cannot use srun and slurm command with the two above commands -> need to use a script launched with sbatch
    * use shifter in a sbatch script:
        * add `#SBATCH --image=edmondchau/fastpm-python:latest`
        * add shifter front of the command
        * cf *launch_test.sh*
