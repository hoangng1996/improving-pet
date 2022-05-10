#!/bin/bash
#SBATCH --job-name=RTE
#SBATCH --open-mode=append
#SBATCH --output=/scratch/sy2199/%j_%x.out
#SBATCH --error=/scratch/sy2199/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --requeue

singularity exec --nv --bind /scratch --overlay /scratch/sy2199/overlay-25GB-500K.ext3:ro /scratch/sy2199/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir ../rte-data \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name rte \
--output_dir ../rte-output \
--do_train \
--do_eval \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 2 \
--pet_gradient_accumulation_steps 8 \
--pet_max_steps 250 \
--pet_max_seq_length 256 \
--sc_per_gpu_train_batch_size 2 \
--sc_per_gpu_unlabeled_batch_size 2 \
--sc_gradient_accumulation_steps 8 \
--sc_max_steps 5000 \
--sc_max_seq_length 256
"
