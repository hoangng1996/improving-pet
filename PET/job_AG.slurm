#!/bin/bash
#SBATCH --job-name=AGNews
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ss14358/%j_%x.out
#SBATCH --error=/scratch/ss14358/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

singularity exec --nv --bind /scratch --overlay /scratch/ss14358/overlay-25GB-500K.ext3:ro /scratch/ss14358/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python cli.py \
--method pet \
--pattern_ids 0 1 2 3 4 5 \
--data_dir ../agnews-data \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name agnews \
--output_dir agnews-output \
--do_train \
--do_eval \
--train_examples 10 \
--unlabeled_examples 40000 \
--split_examples_evenly \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--pet_max_steps 250 \
--lm_training \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 4 \
--sc_gradient_accumulation_steps 4 \
--sc_max_steps 5000"
