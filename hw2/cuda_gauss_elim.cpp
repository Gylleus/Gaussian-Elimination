#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <stdio.h>
//#include <cuda.h>

#define THREAD_AMOUNT_X 10
#define THREAD_AMOUNT_Y 10

void print_a();
void print_b();
void print_x();
void check_result();
void initialize_data();
void partial_pivoting(int row);

pthread_barrier_t barrier;

float* a;
float* a_d;
float* b;
float* x;
int n;
int thread_amount; 
int kek;

//const int THREADS_PER_BLOCK;

__global__ void gauss_add(float* m_values, int row, float** A) {
	int i = row + blockIdx.x * blockDim.x + threadIdx.x;
	int j = row + blockIdx.y * blockDim.y + threadIdx.y;

	
	A[i][j]-= m_values[i] * A[row][j];
} 

void gauss_elim() {
	size_t size = n * sizeof(float*);
	cudaError_t err = cudaMalloc(&a_d, size);
	printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
	size = n * sizeof(float);

	for (int i = 0; i < n; i++) {
		cudaError_t err = cudaMalloc(&a_d[i], size);
		printf("CUDA malloc A: %s\n",cudaGetErrorString(err));	
	}

	for (int row = 0; row < n; row++) {
		// Wait for all threads to catch up before next row begins
		print_a();
		partial_pivoting(row);
		
		float m_values[n-row];
				
		for (int k = row+1; k < n; k++) {
			float m = a[k][row]/a[row][row];
			m_values[k-row] = m;
			b[k]-= m*b[row];
		}
		
		dim3 threadsPerBlock(THREAD_AMOUNT_X, THREAD_AMOUNT_Y);
//		dim3 numBlocks((n-row)/threadsPerBlock.x,(n-row)/threadsPerBlock.y);
		dim3 numBlocks(1, 1);
//		std::cerr << "Blocks: " << numBlocks.x << '\n';
//		std::cerr << "Threads: " << threadsPerBlock.x << '\n';			
		gauss_add<<<numBlocks, threadsPerBlock>>>(m_values, row, a_d);
		/*
		// Gaussian elimination, rows divided into seperate threads
		for (int k = thread_row; k < std::round(tf + (n-row)/(float)thread_amount); k++) {
			float m = a[k][row]/a[row][row];
			for (int col = row; col < n; col++) {
				a[k][col]-= m*a[row][col];
			}
			b[k]-= m*b[row];
		} */
	}
} 

void partial_pivoting(int row) {
	float ksave = -1.0f;
	int ki = 0;
	// Find biggest element in current column
	for (int col = row; col < n; col++) {
		if (abs(a[col][row]) > ksave) {
			ksave = abs(a[col][row]);
			ki = col;
			
		}
	}
	// Swap rows
	for (int col = row; col < n; col++) {
		float tmp = a[ki][col];
		a[ki][col] = a[row][col];
		a[row][col] = tmp;
	}
	float tmp = b[ki];
	b[ki] = b[row];
	b[row] = tmp;
}


int main(int argc, char* argv[]) {
	
	if (argc != 3) {
		std::cout << "Invalid input. Number of arguments should be 2.\n";
		return 0;
	}
	n = atoi(argv[1]);
	thread_amount = atoi(argv[2]);
	thread_amount = std::min(thread_amount, n);
	
	initialize_data();
	
	print_b();
	print_x();

	gauss_elim();
		
	// Backwards solving
	for (int i = n-1; i >= 0; i--) {
		x[i] = b[i];
		for (int j = i+1; j < n; j++) {
			x[i]-= a[i][j]*x[j];
		}
		x[i]/=a[i][i];
	}
		
	check_result();
	print_b();
	print_x();
	
	// Free memory
	for (int i = 0; i < n; i++) {
		free(a[i]);
	}
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
			a[i][j] = 1 + drand48() * 10;
		}		
		b[i] = drand48() * 10;
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
			r[i] += a[i][j]*x[j];
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
			std::cout << (int)a[i][j] << " ";
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

void mat_get(int row, int col) {
	return a[row*n+col];
}

	int tid = threadIdx.x + threadIdx.y;
	if (tid >= n) return;
	__shared__ float max;
	__shared__ int ki;
	max = -1.0f;
	ki = 0;
	if (tid == row) {
	// Gick inte, inget bra sätt att hitta max på med flera trådar 
		// Find biggest element in current column
		for (int k = row; k < n; k++) {
			if (abs(A[k*n + row]) > max) {
				max = abs(A[k*n + row]);
				ki = k;
			}
		}
	}
	__syncthreads();
	
	float tmp = A[ki*n + tid];
	A[ki*n + tid] = A[row*n + tid];
	A[row*n + tid] = tmp;
	//A[row*n + tid]++;
		
	if (tid == row) {
		tmp = B[ki];
		B[ki] = B[row];
		B[row] = tmp;
	}
