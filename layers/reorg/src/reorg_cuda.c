#include <THC/THC.h>
#include "reorg_cuda_kernel.h"

extern THCState *state;

int reorg_cuda(THCudaTensor *x_tensor, int w, int h, int c, int batch, int stride, int forward, THCudaTensor *out_tensor)
{
    float * x = THCudaTensor_data(state, x_tensor);
    float * out = THCudaTensor_data(state, out_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);
    reorg_ongpu(x, w, h, c, batch, stride, forward, out, stream);

    return 1;
}