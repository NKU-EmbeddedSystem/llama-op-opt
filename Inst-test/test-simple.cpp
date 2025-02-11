#include <arm_sve.h>
#include <unistd.h>
#include <time.h>

#include "common.h"

#if defined(SUPER_LARGE)
// test matrix SUPER_LARGE
#define M 4096
#define K 512
#define N 4096

#elif defined(LARGE)
// test matrix LARGE
#define M 1024
#define K 1024
#define N 1024

#elif defined(MEDIUM)
// test matrix MEDIUM
#define M 512
#define K 512
#define N 512

#elif defined(SMALL)
// test matrix SMALL
#define M 256
#define K 256
#define N 256

#else
// test matrix TEST
#define M 16
#define K 16
#define N 16

#endif

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

int main(int argc, char* argv[]){
    int sp = -1;

    int opt;
    while ((opt = getopt(argc, argv, "s:")) != -1) {
        switch (opt) {
            case 's':
                sp = std::stoi(optarg); 
                break;
            default:
                std::cerr << "usage:  -s <sparsity>" << std::endl;
                return 1;
        }
    }
    if(sp == -1){
        std::cerr << "usage:  -s <sparsity>" << std::endl;
        return 1;
    }

    std::map<int,std::vector<int>> index_row,index_col;
    matrix_init_sparse(A,M,K,666,sp,index_row,index_col);
    matrix_init(B,K,N,888);
    matrix_init_zero(C,M,N);

    clock_t start_time, end_time, total_time;

    start_time = clock();
    for(int i = 0;i<5;i++) {
        matrix_mul_mat(A, B, C, M, K, N);
    } 
    end_time = clock();

    total_time = end_time - start_time;
    // print_matrix(A, M, K);
    // print_matrix(B, K, N);
    // print_matrix(C, M, N);
    std::cout<<"Simple mul_mat took "<< (double)total_time / CLOCKS_PER_SEC / 5 << " seconds to execute.  Sparsity: "<< sp <<std::endl;
    return 0;
}