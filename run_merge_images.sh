#!/bin/bash
#
#SBATCH --job-name=run_merge_images
#SBATCH --ntasks=1
#SBATCH --partition=long

cd ~
. venv/bin/activate
