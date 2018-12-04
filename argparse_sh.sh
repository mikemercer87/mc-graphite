#$ -S /bin/bash

#$ -q serial
#$ -N alphacorr

# Example job submission script to calculate the equilibrium lattice properties over a variable chemical potential grid.

source /etc/profile

module add python/2.7.3

lowmu=`nth $SGE_TASK_ID low_mu_arr`
highmu=`nth $SGE_TASK_ID high_mu_arr`
id=$(($SGE_TASK_ID-1))

echo $id

python mc_argparse.py --n_iterations 1000000 --q_relaxation 990000 --binsize 200 --P 2 --alpha4 1.80712905 --beta4 77.2249

echo Job running on compute node 'uname -n'
# echo $myvalue
