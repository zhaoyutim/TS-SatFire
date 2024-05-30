#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  50GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 4


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate torchenv
PYTHONUNBUFFERED=1; python3 run_spatial_model.py -m $1 -mode $2 -b $3 -r $4 -lr $5 -nh $6 -ed $7 -nc $8 -ts $9 -it ${10}
