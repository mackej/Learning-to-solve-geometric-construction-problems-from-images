#!/bin/bash
#SBATCH --job-name=hypothesis_search
#SBATCH --output=search/search_nn_%A.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=node-[02-12],dgx-[2-5]
#SBATCH --mail-type=END,FAIL



module purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load cuDNN/7.1.4.18-fosscuda-2018b
module load Anaconda3/5.0.1

. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate geometry_from_images_test

echo start
date +"%T"

nvidia-smi

python run_hypothesis_tree_search.py

echo finish

