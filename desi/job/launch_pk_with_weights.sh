#!/bin/bash


# Script to launch:
# ./launch.sh 'True' 'False'
# ./launch.sh 'False' 'True'
fnl_0=$1
fnl_25=$2
fnl_12=$3
fnl__25=$4

# Parameters for fnl = 0:
sim_name='run-knl-3-fnl-0'
aout_list=("0.3636")   # ("0.2857" "0.3076" "0.3333" "0.3636" "0.4000")
release='Y1'
region='SSGC'
npasses='1' # pixels with at least # pass >= npasses --> 1 gives all the observable footprint

which_contamination='Y1' # either TS or Y1
seed_data='123'
seed_randoms='31'

generate_randoms='True'
name_randoms='run-knl-3-randoms'

generate_contamination='False'
name_contamination='run-knl-3-contamination'

if [[ $fnl_0 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout
        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-ini-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'True' 'False' 'False' 'False')
        #echo "    * power_spectrum: "$power_spectrum
        
        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-inicorr-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-cont-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
        #echo "    * power_spectrum: "$power_spectrum

        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-corr-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
        #echo "    * power_spectrum: "$power_spectrum
    done
fi

# Parameters for fnl = 25:
sim_name='run-knl-3-fnl-25'
generate_randoms='False'
generate_contamination='False'
seed_data='123'

if [[ $fnl_25 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout
        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-ini-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'True' 'False' 'False' 'False')
        #echo "    * power_spectrum: "$power_spectrum
        
        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-inicorr-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-cont-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
        #echo "    * power_spectrum: "$power_spectrum

        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-corr-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
        #echo "    * power_spectrum: "$power_spectrum
    done
fi

# Parameters for fnl = 12:
sim_name='run-knl-3-fnl-12'
generate_randoms='False'
generate_contamination='False'
seed_data='23'

if [[ $fnl_12 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout
        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-ini-fnl-12-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'True' 'False' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum
        
        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-inicorr-fnl-12-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-cont-fnl-12-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-corr-fnl-12-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
        echo "    * power_spectrum: "$power_spectrum
    done
fi

# Parameters for fnl = -25:
sim_name='run-knl-3-fnl--25'
generate_randoms='False'
generate_contamination='False'
seed_data='123'

if [[ $fnl__25 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-ini-fnl--25-${aout}-${release}-${region}-${npasses} --parsable\
         #                power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'True' 'False' 'False' 'False')
        #echo "    * power_spectrum: "$power_spectrum
        
        power_spectrum=$(sbatch --job-name power_spectrum_with_weights-inicorr-fnl--25-${aout}-${release}-${region}-${npasses} --parsable\
                         power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-cont-fnl--25-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
        #echo "    * power_spectrum: "$power_spectrum

        #power_spectrum=$(sbatch --job-name power_spectrum_with_weights-corr-fnl--25-${aout}-${release}-${region}-${npasses} --parsable\
        #                 power_spectrum_with_weights.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
        #echo "    * power_spectrum: "$power_spectrum
    done
fi
