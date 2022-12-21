#!/bin/bash

# launch fastpm then all the post_processing treatment (halos finder + subsampler ect)

## For this size you need to ask an increase scratch quotas (typically 100 Tb)
## You need to change sim_name and $DIR which is define in all the job (if not running in my scratch)
## Do not forget to update the config.py file

## When you are ready:
## chmod u+x launch.sh
## ./launch.sh
## when all the jobs are done: ./seff-and-clean.sh

## Do not forget to save the simulation in the CFS and in feynmann

sim_name='run-knl-3-fnl-TODO'
aout_list=("0.2857" "0.3076" "0.3333" "0.3636" "0.4000")

fastpm=$(sbatch --parsable fastpm.job $sim_name)
echo "fastpm: "$fastpm
echo $fastpm >> slurm-job-id.txt

for aout in ${aout_list[@]};
do
    echo "Work at a="$aout
    halos=$(sbatch --job-name halos-$aout --parsable --dependency=afterok:$fastpm halos.job $sim_name $aout)
    echo "    * halos: "$halos
    echo $halos >> slurm-job-id.txt

    subsample=$(sbatch --job-name subsampling-$aout --parsable --dependency=afterok:$halos subsample.job $sim_name $aout)
    echo "    * subsample: "$subsample
    echo $subsample >> slurm-job-id.txt
    echo
done
