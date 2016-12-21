#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>

void check_upper();
void print_a();
void print_b();
void print_x();
void print_m();
void check_result();
void initialize_data();
void partial_pivoting(int row);
float* a_get(int row, int col);

pthread_barrier_t barrier;

float* a;
float* a_d;
float* b_d;
float* b;
float* x;
int n;
int thread_amount; 
int kek;

int BLOCK_SIZE;

__global__ void gauss_solve(int row, float* A, float* B, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	if (idx >= n || idx <= row || idy < row || idy >= n) return;
	
	__shared__ float m[32];
	__shared__ float tmp[32];
	
	m[tx] = A[idx*n+row]/A[row*n+row];
	tmp[ty] = A[row*n + idy];
	__syncthreads();
	
	A[idx*n+idy] -= m[tx] * tmp[ty];
	if (idy == n-1) B[idx]-= m[tx]*B[row];
} 

__global__ void part_pivot(int row, float* A, float* B, int n) {
	
	float ksave = -1.0f;
	int ki = 0;
	
	for (int col = row; col < n; col++) {
		if (abs(A[col*n + row]) > ksave) {
			ksave = abs(A[col*n + row]);
			ki = col;
		}
	}
	// Swap rows
	for (int col = row; col < n; col++) {
		float tmp = A[ki*n + col];
		A[ki*n + col] = A[row*n + col];
		A[row*n + col] = tmp;
	}
}

void gauss_elim() {
	size_t size = n * n * sizeof(float);
	cudaMalloc(&a_d, size);
	size_t size_b = n * sizeof(float);
	cudaMalloc(&b_d, size_b);
	
	cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, size_b, cudaMemcpyHostToDevice);		
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	size_t blocks_needed = ceil(n / (float)BLOCK_SIZE);
	dim3 numBlocks(blocks_needed, blocks_needed);
	
	for (int row = 0; row < n; row++) {
		
		part_pivot<<<dim3(1,1),dim3(1,1)>>>(row, a_d, b_d, n);
		//cudaDeviceSynchronize();
		
		gauss_solve<<<numBlocks, threadsPerBlock>>>(row, a_d, b_d, n);
		//cudaDeviceSynchronize();
		
	}
	cudaMemcpy(a, a_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, b_d, size_b, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
	cudaFree(b_d);
}

int main(int argc, char* argv[]) {
	
	if (argc != 3) {
		std::cout << "Invalid input. Number of arguments should be 2.\n";
		return 0;
	}
	n = atoi(argv[1]);
	BLOCK_SIZE = atoi(argv[2]);
	BLOCK_SIZE = std::min(BLOCK_SIZE, 32);
	
	initialize_data();
	
	// Get current system time
	struct timeval tp;
	gettimeofday(&tp, NULL);
	long int start = tp.tv_sec * 1000 + tp.tv_usec / 1000;

	
	gauss_elim();
		
	// Backwards solving
	for (int i = n-1; i >= 0; i--) {
		x[i] = b[i];
		for (int j = i+1; j < n; j++) {
			x[i]-= a[i*n + j]*x[j];
		}
		x[i]/=a[i*n + i];
	}
	
	gettimeofday(&tp, NULL);
	long int end = tp.tv_sec * 1000 + tp.tv_usec / 1000;	
	std::cout << "Execution time = " << end-start << " ms.\n";
	
	check_result();
	check_upper();
	
	free(a);
	delete[] b;
	delete[] x;
}

/**
 * Allocates memory and randomizes values for the matrix and vectors
 * */
void initialize_data() {
	a = (float*) malloc(n*n*sizeof(float));

	b = new float[n];
	x = new float[n];
	
	for (int i = 0; i < n; i++) {			
		for (int j = 0; j < n; j++) {
			a[i*n+j] = 1 + drand48() * 9;
		}	
		b[i] = drand48() * 9;
	}
}
/**
 * Checks and prints the final result by calculating the L2-norm of
 * Ax-b
 * */
void check_result() {
	float* r = new float[n];
	for (int i = 0; i < n; i++) {
		r[i] = 0;
		for (int j = 0; j < n; j++) {
			r[i] += a[i*n + j]*x[j];
		}
		r[i]-=b[i];
	}
	float result = 0;
	for (int i = 0; i < n; i++) {
		result += r[i]*r[i];
	}
	result = sqrt(result);
	std::cerr << "Error factor: " << result << ".\n";
}

/**
 * Prints the matrix A
 * */
void print_a() {
	std::cout << "A: \n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << (int)a[i*n + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";	
}

void print_b() {
	std::cout << "B: \n";	
	for (int i = 0; i < n; i++) {
		std::cout << b[i] << "\n";
	}
	std::cout << "\n\n";	
}

void print_x() {
	std::cout << "X: \n";
	for (int i = 0; i < n; i++) {
		std::cout << x[i] << "\n";
	}
	std::cout << "\n\n";	
}

void check_upper() {
	std::cout << "Check if upper: \n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if(((int)a[i*n+j]!=0)) 
				std::cout << "Value: " << (int)a[i*n + j] << " at " << i << "," << j << "\n";
		}
	}
}
