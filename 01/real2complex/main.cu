#include <stdio.h>
#include <cuda_runtime.h>
#include "rename_device_func.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

#define BS 1024

__global__
void real2complex(double* r, double2* c, int N) 
{
	const int indx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (indx < N) {
		return;
	}
	
	c[indx].x = r[indx];
	c[indx].y = .0;
}

__global__ 
void cAbs(double2* c, double* a, int N) {
	int indx = threadIdx.x + blockDim.x * blockIdx.x;
	if (indx < N) {
		a[indx] = sqrtf(c[indx].x * c[indx].x + c[indx].y * c[indx].y);
	}
}


int main() {
	int N = 200'000'000;
	std::vector<double> r(N);

	for (int i = 0; i < N; ++i) {
		r[i] = sin(i);
	}

	int blocks = static_cast<int>((N - 0.5) / BS) + 1;
	std::vector<double> a(N);
	double* r_d, * a_d;
	double2* c_d;

	cudaMalloc((void**)&r_d, N * sizeof(double));
	cudaMalloc((void**)&c_d, N * sizeof(double2));
	cudaMalloc((void**)&a_d, N * sizeof(double));

	cudaMemcpy(r_d, r.data(), N * sizeof(double), cudaMemcpyHostToDevice);

	real2complex KERNEL_ARGS2(blocks, BS) (r_d, c_d, N);

	cAbs KERNEL_ARGS2(blocks, BS) (c_d, a_d, N);

	cudaMemcpy(a.data(), a_d, N * sizeof(double), cudaMemcpyDeviceToHost);
	auto sumR = accumulate(a.begin(), a.end(), 0.);
	for (int i = 0; i < N; i++) {
		std::cout << r[i] << " : " << a[i] << "\n";
	}
	std::cout << sumR;


	cudaFree(a_d);
	cudaFree(r_d);
	cudaFree(c_d);

}
