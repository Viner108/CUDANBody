#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"

void Acceleration_CPU(float* X, float* Y, float* AX, float* AY, int nt, int N, int id) {
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

void Position_CPU(float* X, float* Y, float* VX, float* VY, float* AX, float* AY, float tau, int nt, int Np, int id) {
	int sh = (nt - 1) * Np;
	X[id + nt * Np] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
	Y[id + nt * Np] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;

	VX[id] = AX[id] * tau;
	VY[id] = AY[id] * tau;
}

int comparisonGPUAndCPU() {
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop);
	cudaEventCreate(&stop1);

	int N = 10240; //число частиц
	int NT = 10; // число шагов
	float tau = 0.001f; // шаг по времени 0.001 с

	float* hX;
	float* hY;
	float* hVX;
	float* hVY;
	float* hAX;
	float* hAY;

	unsigned int mem_size = sizeof(float) * N;
	unsigned int mem_size_big = sizeof(float) * N * NT;

	hX = (float*)malloc(mem_size_big);
	hY = (float*)malloc(mem_size_big);
	hVX = (float*)malloc(mem_size);
	hVY = (float*)malloc(mem_size);
	hAX = (float*)malloc(mem_size);
	hAY = (float*)malloc(mem_size);

	float vv, phi;
	for (int j = 0; j < N; j++) {
		phi = (float)rand();
		hX[j] = rand() * cosf(phi) * 1.e-4f;
		hY[j] = rand() * sinf(phi) * 1.e-4f;
		vv = (hX[j] * hX[j] + hX[j] * hX[j]) * 10.f;
		hVX[j] = -vv * sinf(phi);
		hVY[j] = -vv * cosf(phi);
	}


	float* dX;
	float* dY;
	float* dVX;
	float* dVY;
	float* dAX;
	float* dAY;

	cudaMalloc((void**)&dX, mem_size_big);
	cudaMalloc((void**)&dY, mem_size_big);
	cudaMalloc((void**)&dVX, mem_size);
	cudaMalloc((void**)&dVY, mem_size);
	cudaMalloc((void**)&dAX, mem_size);
	cudaMalloc((void**)&dAY, mem_size);

	int N_thread = 256;
	int N_blocks = N / N_thread;

	

	cudaMemcpy(dX, hX, mem_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, mem_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dVX, hVX, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dVY, hVY, mem_size, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	for (int j = 0; j < NT; j++) {
		Acceleration_GPU << < N_blocks, N_thread >> > (dX, dY, dAX, dAY, j, N);
		Position_GPU << < N_blocks, N_thread >> > (dX, dY, dVX, dVY, dAX, dAY, tau, j, N);		
	}

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);

	cudaMemcpy(hX, dX, mem_size_big, cudaMemcpyDeviceToHost);
	cudaMemcpy(hY, dY, mem_size_big, cudaMemcpyDeviceToHost);


	
	

	for (int i = 0; i < N; i++)
	{
		printf("X[%d] = %.5f\n", i, hX[i]);
		printf("Y[%d] = %.5f\n", i, hY[i]);
	}

	cudaEventRecord(start1, 0);

	int id;
	for (int j = 0; j < NT; j++) {
		for (id = 0; id < N; id++) {
			Acceleration_CPU(hX, hY, hAX, hAY, j, N, id);
			Position_CPU(hX, hY, hVX, hVY, hAX, hAY, tau, j, N, id);
		}
	}


	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&timerValueCPU, start1, stop1);
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);
	printf("\n Rate: %f x\n", timerValueGPU / timerValueCPU);


	free(hX);
	free(hY);
	free(hAX);
	free(hAY);
	free(hVX);
	free(hVY);

	cudaFree(dX);
	cudaFree(dVX);
	cudaFree(dY);
	cudaFree(dVY);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;

}

int main()
{
	return comparisonGPUAndCPU();
}


