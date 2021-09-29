#!/bin/bash
#SBATCH --job-name=hypothesis_search
#SBATCH --output=search/search_nn_%A.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
##SBATCH --exclude=node-[02-12],dgx-[2-5]
##SBATCH --nodelist=dgx-2
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


python tree_search_partial_goals.py > uniform_partial_goals_Zeta_TS_20_ep.csv



#python ParalellSearch_test.py --number_of_cpu_probes 32 --number_of_gpu_masters 2 --batch_size 8


#python run_whole_level_pack_inference.py --pack="beta"  >beta_out_long_inference.csv
#python run_whole_level_pack_inference.py --pack="gamma"  >gamma_out_long_inference.csv
#python run_whole_level_pack_inference.py --pack="delta"  >delta_out_long_inference.csv
#python run_whole_level_pack_inference.py --pack="epsilon"  >epsilon_out_long_inference.csv
#python run_whole_level_pack_inference.py --pack="zeta"  >zeta_out_long_inference.csv


echo finish

