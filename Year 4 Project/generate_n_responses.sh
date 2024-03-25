#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4


#SBATCH -t 12:00:00
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --mem=28g

#SBATCH --job-name=gen_responses
#SBATCH --mail-user rmvc61@durham.ac.uk
#SBATCH --mail-type=ALL

#SBATCH --gres=gpu:ampere:1

# Source the bash profile (required to use the module command)
source /etc/profile


# Run your program (replace this with your program)
cd $HOME/Year\ 4\ Project/

./testing.py generating

echo "Complete"