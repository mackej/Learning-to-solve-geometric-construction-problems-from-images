#!/bin/bash
#SBATCH --job-name=ALL_at_once
#SBATCH --output=Train_logs/train_nn_%A.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=100:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=node-[02-12]
#SBATCH --mail-type=END,FAIL
#SBATCH --nodelist=node-17

module purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load cuDNN/7.1.4.18-fosscuda-2018b
module load Anaconda3/5.0.1

. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate geometry_from_images_test

echo start
date +"%T"

nvidia-smi

python TrainGeometryDataset.py   --log_folder_name "All_at_once/20201016_2" --generate_levels ".*" --head_epochs 200 --four_plus_epochs 300 --all_epochs 400 --use_weights "coco" --gpus 2 --data_generation_style "on-the-fly" --train_epochs 10_000 --val_epochs 1000 --history_size 1 --tool_set "min_by_construction" --mask_size 5 --use_heat_map 0 --visualize 0 --load_gen_from_file 0

echo finish

