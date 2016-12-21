#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

void print_a();
void check_result();
void initialize_data();
void partial_pivoting(int row);

float** a;
float* b;
float* x;
int n;
int thread_amount;

void gauss_elim() {
		int k = 0;
		for (int row = 0; row < n; row++) {
		partial_pivoting(row);
		// Parallel solving of current row begins
		#pragma omp parallel num_threads(thread_amount) 
		#pragma omp for private(k) schedule(dynamic)
				for (k = row+1; k < n; k++) {
					float m = a[k][row]/a[row][row];
					for (int col = row; col < n; col++) {
						a[k][col]-= m*a[row][col];
					}
					b[k]-= m*b[row];
				}
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

	// Get current system time
	struct timeval tp;
	gettimeofday(&tp, NULL);
	long int start = tp.tv_sec * 1000 + tp.tv_usec / 1000;
	
	// Start algorithm
	gauss_elim();
	
	// Backwards solving
	for (int i = n-1; i >= 0; i--) {
		x[i] = b[i];
		for (int j = i+1; j < n; j++) {
			x[i]-= a[i][j]*x[j];
		}
		x[i]/=a[i][i];
	}
	gettimeofday(&tp, NULL);
	long int end = tp.tv_sec * 1000 + tp.tv_usec / 1000;
	
	std::cout << "Execution time = " << end-start << " ms.\n";
	check_result();
	
	for (int i = 0; i < n; i++) {
		delete[] a[i];
	}
	delete[] a;
	delete[] b;
	delete[] x;
}

/**
 * Allocates memory and randomizes values for the matrix and vectors
 * */
void initialize_data() {
	a = new float*[n];
	for (int i = 0; i < n; i++) {
		a[i] = new float[n];
	}
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
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << (int)a[i][j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";	
}
