#!/bin/bash
 
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=closest
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --partition=gpuA40x4
#SBATCH --account=beih-delta-gpu
#SBATCH --output=training.out
#SBATCH --constraint=perf,nvperf

module load nccl

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
model_size="huge32"
srun ./profile_rank.sh "$model_size"
