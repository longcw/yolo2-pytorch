#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "reorg_cuda_kernel.h"

#define BLOCK 512

dim3 cuda_gridsize(int n)
{
    int k = (n-1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out, cudaStream_t stream)
{
    int size = w*h*c*batch;
    cudaError_t err;

    reorg_kernel<<<cuda_gridsize(size), BLOCK, 0, stream>>>(size, x, w, h, c, batch, stride, forward, out);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}



#ifdef __cplusplus
}
#endif
