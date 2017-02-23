#ifndef _REORG_CUDA_KERNEL
#define _REORG_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif
