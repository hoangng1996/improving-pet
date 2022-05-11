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
source bin/init.sh
sh bin/setup.sh
sh bin/train.sh config/RTE.json
"