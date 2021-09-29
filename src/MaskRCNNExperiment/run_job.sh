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
##SBATCH --nodelist=node-17

module purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load cuDNN/7.1.4.18-fosscuda-2018b
module load Anaconda3/5.0.1

. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate geometry_from_images_test

echo start
date +"%T"

nvidia-smi

LEVELS="Epsilon.*"
LOG_FOLDER="Partial_goals_uniform/Epsilon_20210904_2"
MODEL_PATH="logs/"$LOG_FOLDER"/mask_rcnn_geometryfromimages_0260.h5"

python TrainGeometryDataset.py --number_of_partial_goals 2 --log_folder_name=$LOG_FOLDER --generate_levels=$LEVELS --head_epochs 120 --four_plus_epochs 180 --all_epochs 260 --use_weights "coco" --gpus 2 --data_generation_style "on-the-fly" --train_epochs 10_000 --val_epochs 1000 --history_size 1 --tool_set "min_by_construction" --mask_size 5 --use_heat_map 0 --visualize 0 --load_gen_from_file 0

python TestModel.py --model_path=$MODEL_PATH --hint 0 --episodes 500 --log_failed_levels 1 --visualization 0 --white_visualization 0 --load_levels_from_failed_logs 0 --generate_levels=$LEVELS --history_size 1 --hint 0 --additional_moves 2 --log_failed_levels_file "unfinished_envs_02" --model_type "SingleModelInference"

echo finish

