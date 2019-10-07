#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>

__global__ void cuda_vecAdd(float *v1, float *v2, float *v3, int offset)
{
	int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
	v3[i] = v1[i] + v2[i];
}

int main(void)
{
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::time_point<Clock> TimePoint;
	typedef std::chrono::duration<float> Time;

	constexpr int n = 256*1000000;

	constexpr int nStreams = 32;
	constexpr int blockSize = 256;
	constexpr int streamSize = n / nStreams;
	constexpr int gridSize = streamSize / blockSize;

	printf("%d %d\n", streamSize, gridSize);

	TimePoint tstart, tstop;

	cudaStream_t streams[nStreams];

	float *v1 = nullptr, *v2 = nullptr, *v3 = nullptr;
	float *dv1 = nullptr, *dv2 = nullptr, *dv3 = nullptr;

	for(int i=0; i<nStreams; i++)
		cudaStreamCreate(&streams[i]);

	cudaMallocHost((void**)&v1, sizeof(float) * n);
	cudaMallocHost((void**)&v2, sizeof(float) * n);
	cudaMallocHost((void**)&v3, sizeof(float) * n);

	cudaMalloc((void**)&dv1, sizeof(float) * n);
	cudaMalloc((void**)&dv2, sizeof(float) * n);
	cudaMalloc((void**)&dv3, sizeof(float) * n);

	for(int i=0; i<n; i++)
	{
		v1[i] = (float)i;
		v2[i] = (float)-i+1;
	}

	for(int i=0; i<n; i++)
		v3[i] = 0.0f;

	tstart = Clock::now();

	for(int i=0; i<nStreams; i++)
	{
		int offset = i * streamSize;

		cudaMemcpyAsync(&dv1[offset], &v1[offset], sizeof(float) * streamSize, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&dv2[offset], &v2[offset], sizeof(float) * streamSize, cudaMemcpyHostToDevice, streams[i]);

		cuda_vecAdd<<<gridSize, blockSize, 0, streams[i]>>>(dv1, dv2, dv3, offset);

		cudaMemcpyAsync(&v3[offset], &dv3[offset], sizeof(float) * streamSize, cudaMemcpyDeviceToHost, streams[i]);
	}
	
	for(int i=0; i<nStreams; i++)
		cudaStreamSynchronize(streams[i]);
	
	tstop = Clock::now();

	float elaspedTimeGPU = std::chrono::duration_cast<Time>(tstop-tstart).count();

	float checkGPU = 0.0f;
	for(int i=0; i<n; i++)
		checkGPU += v3[i];

	printf("CheckGPU: %f -- TimeGPU: %fms\n", checkGPU, elaspedTimeGPU);

	cudaFree(dv1);
	cudaFree(dv2);
	cudaFree(dv3);

	cudaFreeHost(v1);
	cudaFreeHost(v2);
	cudaFreeHost(v3);

	for(int i=0; i<nStreams; i++)
		cudaStreamDestroy(streams[i]);

	return 0;
}