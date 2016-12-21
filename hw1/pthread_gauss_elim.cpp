#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <pthread.h>
#include <stdio.h>
#include <chrono>


void print_a();
void check_result();
void initialize_data();
void partial_pivoting(int row);
std::chrono::milliseconds get_time();

pthread_barrier_t barrier;

float** a;
float* b;
float* x;
int n;
int thread_amount; 

void* gauss_elim(void* s) {
	// Thread id passed as argument
	int t_id = *(int*) s;
	
	for (int row = 0; row < n; row++) {
		// Wait for all threads to catch up before next row begins
		pthread_barrier_wait(&barrier);
		if (t_id == 0) {
			partial_pivoting(row);
		}
		// Wait for first thread to complete partial pivoting
		pthread_barrier_wait(&barrier);
		
		// Destribute rows left
		float tf = row + t_id*(n-row)/(float)thread_amount;
		int thread_row = (int)tf;

		// Gaussian elimination, rows divided into seperate threads
		for (int k = thread_row; k < std::round(tf + (n-row)/(float)thread_amount); k++) {
			if (k <= row) {
				continue;
			}
			float m = a[k][row]/a[row][row];
			for (int col = row; col < n; col++) {
				a[k][col]-= m*a[row][col];
			}
			b[k]-= m*b[row];
		}
	}
	pthread_exit(0);
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
	
	print_a();
	
	std::chrono::milliseconds start = get_time();
	
	// Start algorithm
	pthread_barrier_init (&barrier, NULL, thread_amount);

	// Set thread data
	int thread_rows[thread_amount];
	pthread_t threads[thread_amount];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	
	// Assign the rows a thread will handle
	for (int i = 0; i < thread_amount; i++) {
		thread_rows[i] = i;
		pthread_create(threads+i, &attr, gauss_elim, (void*) (thread_rows+i));
	}
	
	// Wait for gaussian elimination to finish
	for (int i = 0; i < thread_amount; i++) {
		pthread_join(threads[i],NULL);
	}
	
	// Backwards solving
	for (int i = n-1; i >= 0; i--) {
		x[i] = b[i];
		for (int j = i+1; j < n; j++) {
			x[i]-= a[i][j]*x[j];
		}
		x[i]/=a[i][i];
	}
	std::chrono::milliseconds end = get_time();
	std::cout << "Execution time = " << (end-start).count() << " ms.\n";
		
	print_a();
	check_result();
	
	// Free memory
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

std::chrono::milliseconds get_time() {
	return std::chrono::duration_cast< std::chrono::milliseconds >(
    std::chrono::system_clock::now().time_since_epoch());
}
