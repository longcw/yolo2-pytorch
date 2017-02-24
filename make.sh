#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd layers/reorg/src
echo "Compiling reorg layer kernels by nvcc..."
nvcc -c -o reorg_cuda_kernel.cu.o reorg_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py