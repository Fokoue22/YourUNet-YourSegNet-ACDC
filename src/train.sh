#!/bin/bash
#SBATCH --job-name=yourUNet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs_%j.out

# Activer l'environnement
source /scratch/ndom2748/tpenv/bin/activate

cd /scratch/ndom2748/Project/ift780-tp4/src

python train.py --model yourUNet \
                --num-epochs 15 \
                --batch_size 8 \
                --lr 0.001


