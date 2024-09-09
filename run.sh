#!/bin/bash

#PJM -L rscunit=rscunit_pg01    # Resource unit name
#PJM -L rscgrp=gpu-rd-small        # Resource group name
#PJM -L vnode=1                 # number of Nodes
#PJM -L vnode-core=32           # CPU cores
#PJM --mpi proc=32              # MPI processes
#PJM -L gpu-share=4                  # GPU Cards
#PJM -L elapse=10:00:00          # elapse time
#PJM -o 'joblog/run_%j.out'
#PJM -e 'joblog/run_%j.err'



. /usr/share/Modules/init/profile.sh
module use /package/x86_64/nvidia/hpc_sdk/23.1/modulefiles
module load nvhpc/23.1
module use /package/x86_64/modulefiles
module load anaconda3

export PYTHONPATH=/nwpr/nvidia/com006/.local/lib/python3.10/site-packages
conda activate /nwpr/nvidia/com006/.conda/envs/corrdiff

set -x

CUDA_PATH=/package/x86_64/nvidia/hpc_sdk/23.1/Linux_x86_64/23.1/cuda/11.8
export PATH=${CUDA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}
echo $LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}


which python

torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py
