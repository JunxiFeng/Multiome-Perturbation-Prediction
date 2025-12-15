#!/bin/bash
#SBATCH --job-name=LatentAdditive_ATAC_celltypeaware_GATED
#SBATCH --output=/gpfs/home/junxif/xin_lab/multiome/log/atac/LA_ATAC_CTaware_GATED.out
#SBATCH --error=/gpfs/home/junxif/xin_lab/multiome/log/atac/LA_ATAC_CTaware_GATED.err
#SBATCH --partition=alphafold
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:rtxa5000:2


source ~/.bashrc
mamba activate perturbench

SCRIPT=/gpfs/home/junxif/xin_lab/multiome/py_scripts/atac/add_atac_embedding_gated.py

echo "Running: LatentAdditive_ATAC_celltypeaware_GATED"
python $SCRIPT

echo "===== END ====="
echo "Time: $(date)"
