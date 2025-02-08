#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

// Randomly initialize a matrix ()
// random == 0  ->  init 0
void matrix_init(float* mat, int64_t m, int64_t n, unsigned int seed){
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
		    mat[i*m+j] = (float)rand() / (float)RAND_MAX;
        }
    } 
}

// Initialize a matrix to all zeros
void matrix_init_zero(float* mat, int64_t m, int64_t n){
    memset(mat,0,sizeof(float)*m*n);
}

// Uniformly initialize a sparse matrix
// threshold(0-100) 
// sp <= threshold  -> mat[i] = 0
void matrix_init_sparse(float* mat, int64_t m, int64_t n, unsigned int seed, int threshold, std::vector<int>& index){
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
            if(rand()%100 < threshold){
                mat[i*m+j] = 0;
                continue;
            }
		    mat[i*m+j] = (float)rand() / (float)RAND_MAX;
            index.push_back(i*m+j);
        }
    } 
}

void print_matrix(float* mat, int64_t m, int64_t n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
		   std::cout<<mat[i*m+j]<<" ";
        }
        std::cout<<std::endl;
    } 
}