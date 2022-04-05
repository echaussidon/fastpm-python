# How to use docker:

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

    * To build the image (warning installing MPICH is quite long) run `docker build edmondchau/fastpm-python .`
    * To upload the image in docker hub run `docker push edmondchau/fastpm-python` (the repo edmondchau/fastpm-python is already created before pushing)

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
