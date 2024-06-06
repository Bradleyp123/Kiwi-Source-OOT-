
#include "data_initialization.cpp"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to calculate direction sign mask for multiple rays
__global__ void calculateDirectionSignMask(float* direction, int* results, int numRays) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numRays) {
		int ormask, andmask;
		int32_t* treat_as_int = (int32_t*)&direction[idx * 12]; // 4 components per ray (4x3=12)

		ormask = andmask = treat_as_int[0];
		for (int i = 1; i < 12; i++) {
			ormask |= treat_as_int[i];
			andmask &= treat_as_int[i];
		}

		if (ormask >= 0) {
			results[idx] = 0;
		}
		else {
			if (andmask < 0)
				results[idx] = 1;
			else
				results[idx] = -1;
		}
	}
}

// CUDA kernel to add a triangle
__global__ void addTriangleKernel(int32_t* ids, float* vertices, float* colors, uint16_t* flags, int32_t* materials, int ntris, int Flags) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ntris) {
		// Add triangle logic here
		// This would involve storing triangles in some global memory
		// Assuming these arrays are preallocated and sized correctly
	}
}

// Function to launch CUDA kernel to calculate direction sign mask
void calculateDirectionSignMaskCuda(float* direction, int* results, int numRays) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;
	calculateDirectionSignMask << <blocksPerGrid, threadsPerBlock >> >(direction, results, numRays);
	cudaDeviceSynchronize();
}

// Function to launch CUDA kernel to add a triangle
void addTriangleCuda(int32_t* ids, float* vertices, float* colors, uint16_t* flags, int32_t* materials, int ntris, int Flags) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (ntris + threadsPerBlock - 1) / threadsPerBlock;
	addTriangleKernel << <blocksPerGrid, threadsPerBlock >> >(ids, vertices, colors, flags, materials, ntris, Flags);
	cudaDeviceSynchronize();
}

int main() {
	// Define number of rays
	int numRays = 100;

	// Allocate host memory
	float* h_direction = (float*)malloc(numRays * 12 * sizeof(float));
	int* h_results = (int*)malloc(numRays * sizeof(int));

	// Initialize directions (example values)
	for (int i = 0; i < numRays * 12; i++) {
		h_direction[i] = (float)i;
	}

	// Allocate device memory
	float* d_direction;
	int* d_results;
	cudaMalloc((void**)&d_direction, numRays * 12 * sizeof(float));
	cudaMalloc((void**)&d_results, numRays * sizeof(int));

	// Copy data to device
	cudaMemcpy(d_direction, h_direction, numRays * 12 * sizeof(float), cudaMemcpyHostToDevice);

	// Launch kernel to calculate direction sign mask
	calculateDirectionSignMaskCuda(d_direction, d_results, numRays);

	// Copy results back to host
	cudaMemcpy(h_results, d_results, numRays * sizeof(int), cudaMemcpyDeviceToHost);

	// Print results (example)
	for (int i = 0; i < numRays; i++) {
		printf("Ray %d: DirectionSignMask = %d\n", i, h_results[i]);
	}

	// Free device memory
	cudaFree(d_direction);
	cudaFree(d_results);

	// Free host memory
	free(h_direction);
	free(h_results);

	return 0;
}
