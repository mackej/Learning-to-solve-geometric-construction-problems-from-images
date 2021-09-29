#!/bin/bash
#SBATCH --job-name=euclideaVyolact
#SBATCH --output=Train_logs/train_nn_%A.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=node-[02-12],dgx-2
#SBATCH --mail-type=END,FAIL
##SBATCH --nodelist=dgx-3

module purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load cuDNN/7.1.4.18-fosscuda-2018b
module load Anaconda3/5.0.1

. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate yolact

echo start
date +"%T"

nvidia-smi

LEVELS="alpha.*"
LOG_FOLDER="weights/alpha_fine_tune_layers_2021_07_27"

nvidia-smi > $LOG_FOLDER"/nvidi-smi"

python train.py  --generate_levels=$LEVELS --save_folder=$LOG_FOLDER --log_folder=$LOG_FOLDER --config=Euclidea_config --start_iter=0 --batch_size=8 --num_workers=0 --keep_latest
python analyze_loss.py --log_folder=$LOG_FOLDER --range 3
model_path=$(ls $LOG_FOLDER"/"*.pth -t1 | head -n1)
python TestModel.py --generate_levels=$LEVELS --model_path=$model_path --visualization 0 --episodes 500 --log_failed_levels 1 --load_levels_from_failed_log 0 --log_failed_levels_file=$LOG_FOLDER"/unfinished"
echo finish

