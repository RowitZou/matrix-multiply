#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#define BLOCKSIZE 32
#define DEBUG 0
// Error code to check return values for CUDA calls
cudaError_t err = cudaSuccess;

// Print the matrix
void print(float *matrix, int n, int m)
{
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
            printf("%f ", matrix[n * i + j]);
        printf("\n");
    }
}

// Initialize the host matrices
void initMatrix(float *matrix, int n, int m)
{
    int size = n * m;
    for(int i = 0; i < size; ++i)
        matrix[i] = rand()/(float)RAND_MAX;
}

// Naive method to multiply two matrices
void mulMatrixNaive(float *matA, float *matB, float *matC, int m, int n, int o)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < o; ++j)
        {
            float sum = 0;
            for(int k = 0; k < n; ++k)
                sum += matA[n * i + k] * matB[o * k + j];
            matC[o * i + j] = sum;
        }
    }
}

__global__ void mulMatrixThreadKernel(const float *matA, const float *matB, float *matC, int m, int n, int o)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int column = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && column < o)
    {
        float sum = 0;

        for(int j = 0; j < n; ++j)
            sum += matA[row * n + j] * matB[j * o + column];

        matC[row * o + column] = sum;
    }
}

__global__ void mulMatrixBlockKernel(const float *matA, const float *matB, float *matC, int m, int n, int o)
{
    int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
    __shared__ float tmp1[BLOCKSIZE][BLOCKSIZE];
	__shared__ float tmp2[BLOCKSIZE][BLOCKSIZE];

    const int row = bx * blockDim.x + tx;
	const int column = by * blockDim.y + ty;

    float sum = 0;

    int ceiling = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    for (int k = 0; k < ceiling; ++k)
	{
		if (k * BLOCKSIZE + ty < n && row < m)
			tmp1[tx][ty] = matA[row * n + k * BLOCKSIZE + ty];
		else
			tmp1[tx][ty] = 0;

		if (k * BLOCKSIZE + tx < n && column < o)
			tmp2[tx][ty] = matB[(k * BLOCKSIZE + tx) * o + column];
		else
			tmp2[tx][ty] = 0;

		__syncthreads();

		for(int i = 0; i < BLOCKSIZE; ++i)
			sum += tmp1[tx][i] * tmp2[i][ty];

		__syncthreads();
	}

	if (row < m && column < o)
		matC[row * o + column] = sum;
}

void mulMatrixThread(const float *matA, const float *matB, float *matC, int m, int n, int o)
{
    float blockSize = BLOCKSIZE;
    float *A = NULL, *B = NULL, *C = NULL;

    // Allocate the device matrices
    err = cudaMalloc(&A, m * n * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&B, n * o * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&C, m * o * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input matrix A and B in host memory to the device input matrices in
    // device memory
    err = cudaMemcpy(A, matA, n * m * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(B, matB, n * o * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the matrix multiply CUDA Kernel
    dim3 threads(blockSize, blockSize, 1);
    dim3 blocks(ceil(m / blockSize), ceil(o / blockSize), 1);
    mulMatrixThreadKernel<<<blocks, threads>>>(A, B, C, m, n, o);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    err = cudaMemcpy(matC, C, m * o * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void mulMatrixBlock(const float *matA, const float *matB, float *matC, int m, int n, int o)
{
    float blockSize = BLOCKSIZE;
    float *A = NULL, *B = NULL, *C = NULL;

    // Allocate the device matrices
    err = cudaMalloc(&A, m * n * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&B, n * o * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&C, m * o * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input matrix A and B in host memory to the device input matrices in
    // device memory
    err = cudaMemcpy(A, matA, n * m * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(B, matB, n * o * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the matrix multiply CUDA Kernel
    dim3 threads(blockSize, blockSize, 1);
    dim3 blocks(ceil(m / blockSize), ceil(o / blockSize), 1);
    mulMatrixBlockKernel<<<blocks, threads>>>(A, B, C, m, n, o);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    err = cudaMemcpy(matC, C, m * o * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

bool compareMatrix(const float *A, const float *B, int n, int m)
{
    int size = n * m;
    for(int i = 0; i < size; ++i)
    {
        if(fabs(A[i] - B[i]) >= 1e-3){
            printf("%d %f %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    clock_t start, finish;
    double duration;
    int input = atoi(argv[1]);
	int n = input;
	int m = input;
	int o = input;

    srand((unsigned)time(NULL));

    // Allocate the host input and output matrices
    float *matA = (float *) malloc(m * n * sizeof(float));
    float *matB = (float *) malloc(n * o * sizeof(float));
    float *matCN = (float *) malloc(m * o * sizeof(float));
    float *matCT = (float *) malloc(m * o * sizeof(float));
    float *matCB = (float *) malloc(m * o * sizeof(float));

    if(matA == NULL || matB == NULL || matCN == NULL
            || matCT == NULL || matCB == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!");
        exit(EXIT_FAILURE);
    }

    initMatrix(matA, m, n);
    initMatrix(matB, n, o);
    initMatrix(matCN, m, o);
    initMatrix(matCT, m, o);
    initMatrix(matCB, m, o);

    start = clock();
    mulMatrixNaive(matA, matB, matCN, m, n, o);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Naive method costs %f seconds.\n", duration);

    start = clock();
    mulMatrixThread(matA, matB, matCT, m, n, o);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Thread method costs %f seconds.\n", duration);

    start = clock();
    mulMatrixBlock(matA, matB, matCB, m, n, o);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Block method costs %f seconds.\n", duration);

    if(compareMatrix(matCN, matCT, m, o) &&
        compareMatrix(matCN, matCB, m, o))
        printf("Test Passed\n");
    else
        printf("Test Failed\n");

    if(DEBUG)
    {
        print(matCN, m, o);
        printf("\n");
        print(matCT, m, o);
        printf("\n");
        print(matCB, m, o);
    }
    // Free host memory
    free(matA);
    free(matB);
    free(matCN);
    free(matCT);
    free(matCB);

    return 0;
}

