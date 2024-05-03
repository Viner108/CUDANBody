#include "cuda_runtime.h"
#ifndef KERNEL_H
#define KERNEL_H


__global__ void Acceleration_GPU(float* X, float* Y, float* AX, float* AY, int nt, int N);

__global__ void Position_GPU(float* X, float* Y, float* VX, float* VY, float* AX, float* AY, float tau, int nt, int Np);

#endif // KERNEL_H