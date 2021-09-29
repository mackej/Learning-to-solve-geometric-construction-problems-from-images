#!/bin/bash
#SBATCH --job-name=geom_test
#SBATCH --output=NN_tests/test_nn_%A.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=51G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
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

LEVELS="zeta.*"
LOG_FOLDER="Zeta_with_partial_goals/20210809_one_partial"
MODEL_PATH="logs/"$LOG_FOLDER"/mask_rcnn_geometryfromimages_0260.h5"

python TestModel.py --number_of_partial_goals 1 --model_path=$MODEL_PATH --hint 0 --episodes 500 --log_failed_levels 1 --visualization 0 --white_visualization 0 --load_levels_from_failed_logs 0 --generate_levels=$LEVELS --history_size 1 --hint 0 --additional_moves 2 --log_failed_levels_file "unfinished_envs_02" --model_type "SingleModelInference"

echo finish
