#!/bin/bash

end=1
for node in `scontrol show hostname $SLURM_NODELIST`; do
    for ((i=1; i<=$end; i++)); do echo $node >> file.txt; done
    end=64
done
