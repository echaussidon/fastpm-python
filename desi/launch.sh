#!/bin/bash


# Script to launch:
# ./launch.sh 'True' 'False'
# ./launch.sh 'False' 'True'
fnl_0=$1
fnl_25=$2


# Parameters for fnl = 0:
sim_name='run-knl-3-fnl-0'
aout_list=("0.3333")   # ("0.2857" "0.3076" "0.3333" "0.3636" "0.4000")
release='Y5'
generate_randoms='True'
name_randoms='run-knl-3-randoms'
seed_data='9'
seed_randoms='18'

if [[ $fnl_0 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout

        make_desi_survey=$(sbatch --job-name make_desi_survey-fnl-0-$aout --parsable\
                          make_desi_survey.job $sim_name $aout $release $generate_randoms $name_randoms $seed_data $seed_randoms)
        echo "    * make_desi_survey: "$make_desi_survey

        imaging_systematics=$(sbatch --job-name imaging_systematis-fnl-0-$aout --parsable\
                              --dependency=afterok:$make_desi_survey\
                              imaging_systematics.job $sim_name $aout $release $name_randoms)
        echo "    * imaging_systematics: "$imaging_systematics

        power_spectrum=$(sbatch --job-name power_spectrum-ini-fnl-0-$aout --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $name_randoms 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-cont-fnl-0-$aout --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $name_randoms 'False' 'True' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-corr-fnl-0-$aout --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $name_randoms 'False' 'False' 'True')
        echo "    * power_spectrum: "$power_spectrum
    done
fi

# Parameters for fnl = 25:
sim_name='run-knl-3-fnl-25'
generate_randoms='False'
seed_data='123'

if [[ $fnl_25 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout

        make_desi_survey=$(sbatch --job-name make_desi_survey-fnl-25-$aout --parsable\
                           make_desi_survey.job $sim_name $aout $release $generate_randoms $name_randoms $seed_data $seed_randoms)
        echo "    * make_desi_survey: "$make_desi_survey

        imaging_systematics=$(sbatch --job-name imaging_systematis-fnl-25-$aout --parsable\
                              --dependency=afterok:$make_desi_survey\
                              imaging_systematics.job $sim_name $aout $release $name_randoms)
        echo "    * imaging_systematics: "$imaging_systematics

        power_spectrum=$(sbatch --job-name power_spectrum-ini-fnl-25-$aout --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $name_randoms 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-cont-fnl-25-$aout --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $name_randoms 'False' 'True' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-corr-fnl-25-$aout --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $name_randoms 'False' 'False' 'True')
        echo "    * power_spectrum: "$power_spectrum
    done
fi
