#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>

void vecAdd(float *v1, float *v2, float *v3, int n)
{
	for(int i=0; i<n; i++)
		v3[i] = v1[i] + v2[i];
}

int main(void)
{
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::time_point<Clock> TimePoint;
	typedef std::chrono::duration<float> Time;

	constexpr int n = 256*1000000;

	TimePoint tstart, tstop;

	float *v1, *v2, *v3;

	v1 = new float[n];
	v2 = new float[n];
	v3 = new float[n];

	for(int i=0; i<n; i++)
	{
		v1[i] = (float)i;
		v2[i] = (float)-i+1;
	}

	for(int i=0; i<n; i++)
		v3[i] = 0.0f;

	tstart = Clock::now();
	vecAdd(v1, v2, v3, n);
	tstop = Clock::now();

	float elaspedTimeCPU = std::chrono::duration_cast<Time>(tstop-tstart).count();

	float checkCPU = 0.0f;
	for(int i=0; i<n; i++)
		checkCPU += v3[i];

	printf("CheckCPU: %f -- TimeCPU: %fms\n", checkCPU, elaspedTimeCPU);

	delete[] v1;
	delete[] v2;
	delete[] v3;

	return 0;
}