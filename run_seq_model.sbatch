#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  70GB
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 4

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate myenv
PYTHONUNBUFFERED=1; python3 run_seq_model.py -m $1 -mode $2 -ts $3 -it $4 -nc $5 -b $6 -r $7 -lr $8 -nh $9 -ed ${10} -nl ${11} -md ${12}
