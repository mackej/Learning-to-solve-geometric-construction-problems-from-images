#!/bin/bash
#SBATCH --job-name=detector_geometry_primitives
#SBATCH --output=train_nn_%A.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=102G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
##SBATCH --nodelist=dgx-3

module purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load cuDNN/7.1.4.18-fosscuda-2018b
module load Anaconda3/5.0.1

. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate geometry_from_images_test

echo start
date +"%T"

nvidia-smi

python DetectorGeometryDataset.py --head_epochs 150 --all_epochs 200 --use_weights "coco" --gpus 1 --train_epochs 10_000 --val_epochs 1000 --history_size 1 --generate_levels ".*" --mask_size 5 --visualize 0

echo finish
