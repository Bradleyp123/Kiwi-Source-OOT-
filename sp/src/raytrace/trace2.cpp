//========= Copyright Valve Corporation, All rights reserved. ============//
// $Id$
#include "raytrace.h"
#include <mathlib/halton.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "data_initialization.cpp"

__device__ uint32 MapDistanceToPixel(float t)
{
	if (t < 0) return 0xffff0000;
	if (t > 100) return 0xff000000;
	int a = t * 1000; a &= 0xff;
	int b = t * 10; b &= 0xff;
	int c = t * .01; c &= 0xff;
	return 0xff000000 + (a << 16) + (b << 8) + c;
}

#define IGAMMA (1.0/2.2)
#define MAGIC_NUMBER (1<<23)

__constant__ float4 Four_MagicNumbers = { MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER };
__constant__ int32_t Four_255s[4] = { 0xff, 0xff, 0xff, 0xff };
#define PIXMASK (*(reinterpret_cast<const float4 *>(&Four_255s)))

__global__ void MapLinearIntensitiesKernel(const FourVectors* intensities, uint32_t* p1, uint32_t* p2, uint32_t* p3, uint32_t* p4)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Convert four pixels worth of SSE-style RGB into ARGB lwords
	const float4 pixscale = { 255.0f, 255.0f, 255.0f, 255.0f };
	float4 r, g, b;
	r = fminf(pixscale, pixscale * powf(intensities[idx].x, IGAMMA));
	g = fminf(pixscale, pixscale * powf(intensities[idx].y, IGAMMA));
	b = fminf(pixscale, pixscale * powf(intensities[idx].z, IGAMMA));

	// Convert to integer
	r = __float_as_int(__int_as_float(r + Four_MagicNumbers)) & PIXMASK;
	g = __float_as_int(__int_as_float(g + Four_MagicNumbers)) & PIXMASK;
	b = __float_as_int(__int_as_float(b + Four_MagicNumbers)) & PIXMASK;

	p1[idx] = (__float_as_int(r) & 0xff) | ((__float_as_int(g) & 0xff) << 8) | ((__float_as_int(b) & 0xff) << 16);
	p2[idx] = ((__float_as_int(r) >> 8) & 0xff) | ((__float_as_int(g) >> 8) & 0xff << 8) | ((__float_as_int(b) >> 8) & 0xff << 16);
	p3[idx] = ((__float_as_int(r) >> 16) & 0xff) | ((__float_as_int(g) >> 16) & 0xff << 8) | ((__float_as_int(b) >> 16) & 0xff << 16);
	p4[idx] = ((__float_as_int(r) >> 24) & 0xff) | ((__float_as_int(g) >> 24) & 0xff << 8) | ((__float_as_int(b) >> 24) & 0xff << 16);
}

__global__ void RenderSceneKernel(
	int width, int height, int stride, uint32_t* output_buffer,
	Vector CameraOrigin, Vector ULCorner, Vector URCorner, Vector LLCorner, Vector LRCorner,
	RayTraceLightingMode_t lmode, const FourVectors* BackgroundColor, const Vector* TriangleColors, const LightDesc_t* LightList, int LightCount)
{
	// CUDA kernel implementation of RenderScene
	// ...
}

class RayTracingEnvironment {
public:
	void RenderScene(int width, int height, int stride, uint32_t* output_buffer, Vector CameraOrigin, Vector ULCorner, Vector URCorner, Vector LLCorner, Vector LRCorner, RayTraceLightingMode_t lmode);
	void ComputeVirtualLightSources();
	void AddToRayStream(RayStream& s, Vector const& start, Vector const& end, RayTracingSingleResult* rslt_out);
	void FinishRayStream(RayStream& s);

private:
	void FlushStreamEntry(RayStream& s, int msk);
};

void RayTracingEnvironment::RenderScene(
	int width, int height, int stride, uint32_t* output_buffer,
	Vector CameraOrigin, Vector ULCorner, Vector URCorner, Vector LLCorner, Vector LRCorner,
	RayTraceLightingMode_t lmode)
{
	// Memory allocations and setup
	FourVectors* d_BackgroundColor;
	Vector* d_TriangleColors;
	LightDesc_t* d_LightList;
	uint32_t* d_output_buffer;

	cudaMalloc(&d_BackgroundColor, sizeof(FourVectors));
	cudaMalloc(&d_TriangleColors, TriangleColors.size() * sizeof(Vector));
	cudaMalloc(&d_LightList, LightList.size() * sizeof(LightDesc_t));
	cudaMalloc(&d_output_buffer, width * height * sizeof(uint32_t));

	cudaMemcpy(d_BackgroundColor, &BackgroundColor, sizeof(FourVectors), cudaMemcpyHostToDevice);
	cudaMemcpy(d_TriangleColors, TriangleColors.data(), TriangleColors.size() * sizeof(Vector), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LightList, LightList.data(), LightList.size() * sizeof(LightDesc_t), cudaMemcpyHostToDevice);

	// Kernel launch
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
	RenderSceneKernel << <gridDim, blockDim >> >(
		width, height, stride, d_output_buffer, CameraOrigin, ULCorner, URCorner, LLCorner, LRCorner,
		lmode, d_BackgroundColor, d_TriangleColors, d_LightList, LightList.size());

	cudaMemcpy(output_buffer, d_output_buffer, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// Cleanup
	cudaFree(d_BackgroundColor);
	cudaFree(d_TriangleColors);
	cudaFree(d_LightList);
	cudaFree(d_output_buffer);
}

void RayTracingEnvironment::ComputeVirtualLightSources()
{
	// Implementation for computing virtual light sources
}

void RayTracingEnvironment::AddToRayStream(RayStream& s, Vector const& start, Vector const& end, RayTracingSingleResult* rslt_out)
{
	Vector delta = end;
	delta -= start;
	int msk = GetSignMask(delta);
	assert(msk >= 0);
	assert(msk < 8);
	int pos = s.n_in_stream[msk];
	assert(pos < 4);
	s.PendingRays[msk].origin.X(pos) = start.x;
	s.PendingRays[msk].origin.Y(pos) = start.y;
	s.PendingRays[msk].origin.Z(pos) = start.z;
	s.PendingRays[msk].direction.X(pos) = delta.x;
	s.PendingRays[msk].direction.Y(pos) = delta.y;
	s.PendingRays[msk].direction.Z(pos) = delta.z;
	s.PendingStreamOutputs[msk][pos] = rslt_out;
	if (pos == 3)
	{
		FlushStreamEntry(s, msk);
	}
	else
		s.n_in_stream[msk]++;
}

void RayTracingEnvironment::FinishRayStream(RayStream& s)
{
	for (int msk = 0; msk < 8; msk++)
	{
		int cnt = s.n_in_stream[msk];
		if (cnt)
		{
			// Fill in unfilled entries with dups of first
			for (int c = cnt; c < 4; c++)
			{
				s.PendingRays[msk].origin.X(c) = s.PendingRays[msk].origin.X(0);
				s.PendingRays[msk].origin.Y(c) = s.PendingRays[msk].origin.Y(0);
				s.PendingRays[msk].origin.Z(c) = s.PendingRays[msk].origin.Z(0);
				s.PendingRays[msk].direction.X(c) = s.PendingRays[msk].direction.X(0);
				s.PendingRays[msk].direction.Y(c) = s.PendingRays[msk].direction.Y(0);
				s.PendingRays[msk].direction.Z(c) = s.PendingRays[msk].direction.Z(0);
				s.PendingStreamOutputs[msk][c] = s.PendingStreamOutputs[msk][0];
			}
			FlushStreamEntry(s, msk);
		}
	}
}

inline void RayTracingEnvironment::FlushStreamEntry(RayStream& s, int msk)
{
	assert(msk >= 0);
	assert(msk < 8);
	float4 tmax = s.PendingRays[msk].direction.length();
	s.PendingRays[msk].direction.VectorNormalizeFast();
	Trace4Rays(s.PendingRays[msk], all_zeros, tmax, &s.Results[msk]);
	for (int oo = 0; oo < 4; oo++)
	{
		*s.PendingStreamOutputs[msk][oo] = s.Results[msk].Extract(oo);
	}
	s.n_in_stream[msk] = 0;
}

// The rest of the code and function implementations follow similar patterns
