#!/bin/bash
#SBATCH --job-name=sft-llm           # name
#SBATCH --nodes=3                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=150          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 200:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x.out              # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export SCRIPT=/opt/shared/train.py

srun --jobid $SLURM_JOBID bash -c 'torchrun \
 --nproc-per-node $GPUS_PER_NODE \
 --nnodes $SLURM_NNODES \
 --node-rank $SLURM_PROCID \
 --master-addr $MASTER_ADDR \
 --master-port $MASTER_PORT \
 $SCRIPT'
