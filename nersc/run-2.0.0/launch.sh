#!/bin/bash
# ./launch.sh

# launch fastpm then hall the post_processing treatment (halos finder + subsampler ect)

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
