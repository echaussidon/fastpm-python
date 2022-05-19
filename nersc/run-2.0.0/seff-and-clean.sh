#!/bin/bash
# ./seff-and-clean.sh

# Collect seff output for all the jobs and add it to the corresponding .out file:
aout_list=("0.2857" "0.3076" "0.3333" "0.3636" "0.4000")

# Collect slurm job-id for the simulation into an array:
IFS=$'\n' read -d '' -r -a lines < slurm-job-id.txt
i=0

echo >> fastpm-${lines[$i]}.out
seff ${lines[$i]} >> fastpm-${lines[$i]}.out
i=$(($i+1))

for aout in ${aout_list[@]};
do
    echo >> halos-$aout-${lines[$i]}.out
    seff ${lines[$i]} >> halos-$aout-${lines[$i]}.out
    i=$(($i+1))

    echo >> subsampling-$aout-${lines[$i]}.out
    seff ${lines[$i]} >> subsampling-$aout-${lines[$i]}.out
    i=$(($i+1))
done

# Clean the directory:

# rm .txt from memory-monitor and move the .png into the directory
rm -rf memory-monitor
mkdir memory-monitor
mv *-memory-monitor.png memory-monitor

# move .out into log directory
mkdir log
rm slurm-job-id.txt
mv *.out log

# move power spectrum into dedicated diretory
mkdir power-spectrum
mv *.npy power-spectrum
