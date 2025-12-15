#!/bin/bash
#SBATCH --job-name=LatentAdditive_GET_celltypeaware_ATTENTION
#SBATCH --output=/gpfs/home/junxif/xin_lab/multiome/log/get/LA_GET_CTaware_ATTENTION.out
#SBATCH --error=/gpfs/home/junxif/xin_lab/multiome/log/get/LA_GET_CTaware_ATTENTION.err
#SBATCH --partition=alphafold
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:rtxa5000:2


source ~/.bashrc
mamba activate perturbench

SCRIPT=/gpfs/home/junxif/xin_lab/multiome/py_scripts/get/add_get_embedding_attention.py

echo "Running: LatentAdditive_GET_celltypeaware_ATTENTION"
python $SCRIPT

echo "===== END ====="
echo "Time: $(date)"
