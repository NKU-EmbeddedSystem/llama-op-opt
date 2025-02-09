#include <arm_sve.h>
#include <time.h>

#include "common.h"

// // test matrix LARGE
// #define M 4096
// #define K 512
// #define N 4096

// test matrix TEST
#define M 4
#define K 4
#define N 4

float A[M*K];
float B[K*N];
float C[M*N];

// Simply mul mat two matrix
void matrix_mul_mat(float* matrix_a, float* matrix_b, float* matrix_c,int m, int k, int n){
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            for(int q=0;q<k;q++){
                matrix_c[i*n+j] += matrix_a[i*k+q] * matrix_b[q*n+j];
            }
        }
    }
}

int main(){
    std::map<int,std::vector<int>> index_row,index_col;
    matrix_init_sparse(A,M,K,666,20,index_row,index_col);
    matrix_init(B,K,N,666);
    matrix_init_zero(C,M,N);

    clock_t start_time, end_time, total_time;

    start_time = clock();
    
    // for(int i = 0;i<100;i++) {
        matrix_mul_mat(A, B, C, M, K, N);
    // } 

    end_time = clock();

    total_time = end_time - start_time;
    std::cout<<"Simple mul_mat took "<< (double)total_time / CLOCKS_PER_SEC/10000 << "seconds to execute."<<std::endl;
    print_matrix(A, M, K);
    print_matrix(B, K, N);
    print_matrix(C, M, N);
    return 0;
}