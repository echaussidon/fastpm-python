#!/bin/bash
# sbatch launch.sh

#SBATCH --image=edmondchau/fastpm-python:1.0.0
#SBATCH --nodes=200
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=knl
#SBTACH --account=desi

#SBATCH --mail-type=ALL
#SBATCH --mail-user=edmond.chaussidon@cea.fr

#SBATCH --job-name=fnl-0-halos
#SBATCH --output="%x-%j.out"

# Problem with shifter at NERSC, it does not clear the PYTHONPATH... (exactly what we do not want to do...)
export PYTHONPATH=''

# pour knl
export OMP_NUM_THREADS=2

# Parameters:
nproc=13600
nproc_proc=6800
sim_name='run-knl-fnl-0'
aout_list=("0.3300" "0.5000" "0.6700")
min_mass_halos=2.25e12

# Timer initialisation:
SECONDS=0

# Run fastpm with congig.py parameters:
echo
echo "Start fastpm-python for sim_name:$sim_name with nproc:$nproc"
echo
srun -n $nproc shifter python -m fastpm.main /global/cscratch1/sd/edmondc/Mocks/$sim_name

# Run post processing for each aout:
for aout in ${aout_list[@]};
do
    echo
    echo "Start post-processing at aout: $aout"
    echo
    srun -n $nproc_proc shifter python -m fastpm.post_processing --sim $sim_name --aout $aout --min_mass_halos $min_mass_halos --subsampling True --subsampling_nbr 102 --delet_original True
done

echo
if (( $SECONDS > 3600 )); then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )); then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi
echo
