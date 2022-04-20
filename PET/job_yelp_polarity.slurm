#!/bin/bash
#SBATCH --job-name=PET_CILVR
#SBATCH --open-mode=append
#SBATCH --output=./job_logs/yelp/%j_%x.out
#SBATCH --error=./job_logs/yelp/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH

conda activate hw1_dagger

nvidia-smi

export OMP_NUM_THREADS=1

python cli.py \
--method ipet \
--pattern_ids 0 1 2 3 \
--data_dir /scratch/xc2057/NLP/yelp_polarity/yelp_review_polarity_csv \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name yelp-polarity \
--output_dir ./output/yelp_ipet \
--do_train \
--do_eval
"