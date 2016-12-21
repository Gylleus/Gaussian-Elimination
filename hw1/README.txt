README

How to run:

To run program, write "./[executable] [matrix size] [thread num]"

For the pthread implementation the executable name is 'pthread_gauss_elim'.

For the pthread implementation the executable name is 'openmp_gauss_elim'.


Example:

To run a 5000x5000 matrix with 4 cores using pthreads:

./pthread_gauss_elim 5000 4


Compilation:

To compile the programs simply type 'make'.

Upon recompiling the 'openmp_gauss_elim' will need to be deleted to recompile (sorry).
