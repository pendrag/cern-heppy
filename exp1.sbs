#!/bin/bash
#SBATCH --job-name=heppy
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output="/mnt/beegfs/sinai-cern/heppy/cern-heppy/slurm-output/exp1_%A_%a.out"
#SBATCH --error="/mnt/beegfs/sinai-cern/heppy/cern-heppy/slurm-output/exp1_%A_%a.err"

# Limpieza de módulos cargados
module purge

# Carga de módulos software
spack load --dependencies miniconda3
spack load --dependencies cuda@11.1

# Activación de entorno virtual
source /mnt/beegfs/sinai-cern/heppy/cern-heppy/venv/bin/activate

# Ejecución del script
srun python $1
