#!/bin/bash
#SBATCH --job-name=Grinnni_Test
#SBATCH --output=Grinnni_Test.txt
#SBATCH --error=Grinnni_Test.err
#SBATCH --partition=k2-math-physics-debug
#SBATCH --mem-per-cpu=25G

source /users/40230511/.bashrc
conda activate smilei

SimDirectory='/mnt/scratch2/users/40230511/Grinni/tests'

SimName='Data'

python ExampleScript.py "${SimDirectory}" "${SimName}"