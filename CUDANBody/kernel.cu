#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void Acceleration_GPU(float* X, float* Y, float* AX, float* AY, int nt, int N)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	float ax = 0.f;
	float ay = 0.f;
	float xx, yy, rr;
	int sh = (nt - 1) * N;
	for (int j = 0; j < N; j++)
	{
		if (j != id) {
			xx = X[j + sh] - X[id + sh];
			yy = Y[j + sh] - Y[id + sh];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr < 0.01f) {
				rr = 10.f / (rr * rr * rr);
				ax += xx * rr;
				ay += yy * rr;
			}
		}
	}
	AX[id] = ax;
	AY[id] = ay;
}

__global__ void Position_GPU(float* X, float* Y, float* VX, float* VY, float* AX, float* AY, float tau, int nt, int Np)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int sh = (nt - 1) * Np;
	X[id + nt * Np] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
	Y[id + nt * Np] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;

	VX[id] = AX[id] * tau;
	VY[id] = AY[id] * tau;
}