#!/bin/bash
#SBATCH --job-name=PET_CILVR
#SBATCH --open-mode=append
#SBATCH --output=./job_logs/agnews/%j_%x.out
#SBATCH --error=./job_logs/agnews/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH

conda activate hw1_dagger

nvidia-smi

export OMP_NUM_THREADS=1

python cli.py \
--method pet \
--pattern_ids 0 1 2 3 4 5 \
--lm_training \
--data_dir /home/xc2057/pet/agnews/original \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name agnews \
--output_dir /scratch/xc2057/nlp_final/agnews/T10_new_aug \
--do_train \
--do_eval \
--train_examples 10 \
--unlabeled_examples 1000 \
--split_examples_evenly \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--pet_repetitions 1 \
--pet_max_steps 250 \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 4 \
--sc_gradient_accumulation_steps 4 \
--sc_max_steps 2500 \
--sc_repetitions 1 \
--overwrite_output_dir \
--augmentation
"