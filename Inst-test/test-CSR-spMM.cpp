#include <arm_sve.h>
#include <omp.h>
#include <vector>
#include "common.h"
#include <cstdint>
#include <assert.h>
#include <unistd.h>


float B[K * N];
float C_single[M * N];
float C_multi[M * N];


// Single-core version of SpMM implementation
void spmm_sve_single(const CSRMatrix& A, const float* B, float* C,int m, int k, int n) {
    const int vl = svcntw();  // Get SVE vector length
    
    // Iterate through each row
    for (int i = 0; i < A.rows; i++) {
        const int row_start = A.row_ptrs[i];
        const int row_end = A.row_ptrs[i + 1];
        const int nnz_row = row_end - row_start;
        
        // Iterate through each column of result matrix
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            
            // Process elements that can be divided by SVE vector length
            int k = 0;
            for (; k + vl <= nnz_row; k += vl) {
                // Create predicate
                svbool_t pred = svptrue_b32();
                
                // Load non-zero element values
                svfloat32_t a_vec = svld1_f32(pred, &A.values[row_start + k]);
                
                // Use gather load to load elements from matrix B
                svint32_t indices = svld1_s32(pred, &A.col_indices[row_start + k]);
                svint32_t offsets = svmul_n_s32_x(pred, indices, n);
                offsets = svadd_n_s32_x(pred, offsets, j);
                svfloat32_t b_vec = svld1_gather_s32index_f32(pred, B, offsets);
                
                // Calculate dot product and accumulate
                svfloat32_t prod = svmul_f32_x(pred, a_vec, b_vec);
                sum += svaddv_f32(pred, prod);
            }
            
            // Process remaining elements
            for (; k < nnz_row; k++) {
                sum += A.values[row_start + k] * B[A.col_indices[row_start + k] * n + j];
            }
            
            C[i * n + j] = sum;
        }
    }
}

// Multi-core version of SpMM implementation
void spmm_sve_multi(const CSRMatrix& A, const float* B, float* C, int m, int k, int n, int num_threads) {
    #pragma omp parallel num_threads(num_threads)
    {
        const int vl = svcntw();  // Get SVE vector length
        
        // Use OpenMP for row-level parallelization
        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < A.rows; i++) {
            const int row_start = A.row_ptrs[i];
            const int row_end = A.row_ptrs[i + 1];
            const int nnz_row = row_end - row_start;
            
            // Iterate through each column of result matrix
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                
                // Process elements that can be divided by SVE vector length
                int k = 0;
                for (; k + vl <= nnz_row; k += vl) {
                    // Create predicate
                    svbool_t pred = svptrue_b32();
                    
                    // Load non-zero element values
                    svfloat32_t a_vec = svld1_f32(pred, &A.values[row_start + k]);
                    
                    // Use gather load to load elements from matrix B
                    svint32_t indices = svld1_s32(pred, &A.col_indices[row_start + k]);
                    svint32_t offsets = svmul_n_s32_x(pred, indices,n);
                    offsets = svadd_n_s32_x(pred, offsets, j);
                    svfloat32_t b_vec = svld1_gather_s32index_f32(pred, B, offsets);
                    
                    // Calculate dot product and accumulate
                    svfloat32_t prod = svmul_f32_x(pred, a_vec, b_vec);
                    sum += svaddv_f32(pred, prod);
                }
                
                // Method 1: Process remaining elements
                // Process remaining elements
                for (; k < nnz_row; k++) {
                    sum += A.values[row_start + k] * B[A.col_indices[row_start + k] * n + j];
                }

                // // // Method 2: use SVE flag to process remaining elements
                // // use SVE flag to process remaining elements
                // if (k < nnz_row) {
                //     // Create predicate for remaining elements
                //     svbool_t pred = svwhilelt_b32(k, nnz_row);
                    
                //     // Load non-zero element values
                //     svfloat32_t a_vec = svld1_f32(pred, &A.values[row_start + k]);
                    
                //     // Use gather load to load elements from matrix B
                //     svint32_t indices = svld1_s32(pred, &A.col_indices[row_start + k]);
                //     svint32_t offsets = svmul_n_s32_x(pred, indices, N);
                //     offsets = svadd_n_s32_x(pred, offsets, j);
                //     svfloat32_t b_vec = svld1_gather_s32index_f32(pred, B, offsets);
                    
                //     // Calculate dot product and accumulate
                //     svfloat32_t prod = svmul_f32_x(pred, a_vec, b_vec);
                //     sum += svaddv_f32(pred, prod);
                // }
                
                C[i * n + j] = sum;
            }
        }
    }
}

// // Usage example
// void example_usage() {
//     // Create an example CSR matrix
//     CSRMatrix A;
//     A.rows = 1000;
//     A.cols = 1000;
//     // Initialize CSR matrix values, col_indices and row_ptrs...
    
//     // Create dense matrix B and result matrix C
//     const int N = 100;  // Number of columns in B
//     std::vector<float> B(A.cols * N);
//     std::vector<float> C(A.rows * N, 0.0f);
    
//     // Use single-core version
//     spmm_sve_single(A, B.data(), C.data(), N);
    
//     // Use multi-core version
//     int num_threads = omp_get_max_threads();
//     spmm_sve_multi(A, B.data(), C.data(), N, num_threads);
// }
int main(int argc, char* argv[]) {
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

    if(sp == -1) {
        std::cerr << "usage:  -s <sparsity>" << std::endl;
        return 1;
    }

    // Get number of threads
    int num_threads = omp_get_max_threads();
    // int num_threads = 1;

    // Initialize matrices
    CSRMatrix A;


    // Initialize matrix data
    matrix_init(B, K, N, 888);
    matrix_init_zero(C_single, M, N);
    matrix_init_zero(C_multi, M, N);
    matrix_init_csr(A, M, K, 666, sp);  // Initialize sparse matrix in CSR format

    clock_t start_time, end_time, total_time;
    
    // Test single-core version
    start_time = clock();
    // for(int i = 0; i < 5; i++) {
        spmm_sve_single(A, B, C_single, M, K, N);
    // }
    end_time = clock();
    total_time = end_time - start_time;
    std::cout << "Single-core SpMM took " << (double)total_time / CLOCKS_PER_SEC 
              << " seconds to execute. Sparsity: " << sp << std::endl;

    // Test multi-core version
    start_time = clock();
    // for(int i = 0; i < 5; i++) {
        spmm_sve_multi(A, B, C_multi, M, K, N, num_threads);
    // }
    end_time = clock();
    total_time = end_time - start_time;
    std::cout << "Multi-core SpMM (" << num_threads << " threads) took " 
              << (double)total_time / CLOCKS_PER_SEC
              << " seconds to execute. Sparsity: " << sp << std::endl;

    // Optional: print result matrices for verification
    std::cout<<"A: "<<std::endl;
    print_CSR_matrix_dense(A);
    std::cout<<"B: "<<std::endl;
    print_matrix(B, K, N);
    std::cout<<"C: "<<std::endl;
    print_matrix(C_single, M, N);
    std::cout<<"--------------------------------"<<std::endl;
    print_matrix(C_multi, M, N);

    return 0;
}