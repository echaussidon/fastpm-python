#!/bin/bash


# Script to launch:
# ./launch.sh 'True' 'False'
# ./launch.sh 'False' 'True'
fnl_0=$1
fnl_25=$2


# Parameters for fnl = 0:
sim_name='run-knl-3-fnl-0'
aout_list=("0.3636")   # ("0.2857" "0.3076" "0.3333" "0.3636" "0.4000")
release='Y1'
region='SNGC'
npasses='1' # pixels with at least # pass >= npasses --> 1 gives all the observable footprint
generate_randoms='True'
name_randoms='run-knl-3-randoms'
generate_contamination='True'
name_contamination='run-knl-3-contamination'
which_contamination='Y1' # either TS or Y1
seed_data='11'
seed_randoms='31'

if [[ $fnl_0 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout

        make_desi_survey=$(sbatch --job-name make_desi_survey-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
                          make_desi_survey.job $sim_name $aout $release $region $npasses $generate_randoms $name_randoms $generate_contamination $name_contamination $seed_data $seed_randoms)
        echo "    * make_desi_survey: "$make_desi_survey

        imaging_systematics=$(sbatch --job-name imaging_systematis-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
                              --dependency=afterok:$make_desi_survey\
                              imaging_systematics.job $sim_name $aout $release $region $npasses $name_randoms $which_contamination)
        echo "    * imaging_systematics: "$imaging_systematics

        power_spectrum=$(sbatch --job-name power_spectrum-ini-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'True' 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-cont-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-corr-fnl-0-${aout}-${release}-${region}-${npasses} --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
        echo "    * power_spectrum: "$power_spectrum
    done
fi

# aout='0.3636'
# power_spectrum=$(sbatch --job-name power_spectrum-ini-fnl-0-$aout-${release}-$region --parsable\
#                 power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'True' 'False' 'False')
# echo "    * power_spectrum: "$power_spectrum
# power_spectrum=$(sbatch --job-name power_spectrum-cont-fnl-0-$aout-${release}-$region --parsable\
#                  power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
# echo "    * power_spectrum: "$power_spectrum
# power_spectrum=$(sbatch --job-name power_spectrum-corr-fnl-0-$aout-${release}-$region --parsable\
#                  power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
# echo "    * power_spectrum: "$power_spectrum


# Parameters for fnl = 25:
sim_name='run-knl-3-fnl-25'
generate_randoms='False'
generate_contamination='False'
seed_data='123'

if [[ $fnl_25 == 'True' ]]; then
    for aout in ${aout_list[@]}; do
        echo "Work at a="$aout

        make_desi_survey=$(sbatch --job-name make_desi_survey-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
                           make_desi_survey.job $sim_name $aout $release $region $npasses $generate_randoms $name_randoms $generate_contamination $name_contamination $seed_data $seed_randoms)
        echo "    * make_desi_survey: "$make_desi_survey

        imaging_systematics=$(sbatch --job-name imaging_systematis-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
                              --dependency=afterok:$make_desi_survey\
                              imaging_systematics.job $sim_name $aout $release $region $npasses $name_randoms $which_contamination)
        echo "    * imaging_systematics: "$imaging_systematics

        power_spectrum=$(sbatch --job-name power_spectrum-ini-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'True' 'True' 'False' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-cont-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
        echo "    * power_spectrum: "$power_spectrum

        power_spectrum=$(sbatch --job-name power_spectrum-corr-fnl-25-${aout}-${release}-${region}-${npasses} --parsable\
                         --dependency=afterok:$imaging_systematics\
                         power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
        echo "    * power_spectrum: "$power_spectrum
    done
fi

# aout='0.3636'
# power_spectrum=$(sbatch --job-name power_spectrum-ini-fnl-25-$aout-${release}-$region --parsable\
#                  power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'True' 'False' 'False')
# echo "    * power_spectrum: "$power_spectrum
# power_spectrum=$(sbatch --job-name power_spectrum-cont-fnl-25-$aout-${release}-$region --parsable\
#                  power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'True' 'False')
# echo "    * power_spectrum: "$power_spectrum
# power_spectrum=$(sbatch --job-name power_spectrum-corr-fnl-25-$aout-${release}-$region --parsable\
#                  power_spectrum.job $sim_name $aout $release $region $npasses $name_randoms 'False' 'False' 'False' 'True')
# echo "    * power_spectrum: "$power_spectrum
