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
    matrix_init_zero(C,m,n); 
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
            // 展开4次循环
            for (; k + 4*vl <= nnz_row; k += 4*vl) {
                // Create predicate
                svbool_t pred = svptrue_b32();
                
                // Load non-zero element values
                svfloat32_t a_vec0 = svld1_f32(pred, &A.values[row_start + k]);
                svfloat32_t a_vec1 = svld1_f32(pred, &A.values[row_start + k + vl]);
                svfloat32_t a_vec2 = svld1_f32(pred, &A.values[row_start + k + 2*vl]);
                svfloat32_t a_vec3 = svld1_f32(pred, &A.values[row_start + k + 3*vl]);
                
                // Use gather load to load elements from matrix B
                svint32_t indices0 = svld1_s32(pred, &A.col_indices[row_start + k]);
                svint32_t indices1 = svld1_s32(pred, &A.col_indices[row_start + k + vl]);
                svint32_t indices2 = svld1_s32(pred, &A.col_indices[row_start + k + 2*vl]);
                svint32_t indices3 = svld1_s32(pred, &A.col_indices[row_start + k + 3*vl]);

                svint32_t offsets0 = svmul_n_s32_x(pred, indices0, n);
                svint32_t offsets1 = svmul_n_s32_x(pred, indices1, n);
                svint32_t offsets2 = svmul_n_s32_x(pred, indices2, n);
                svint32_t offsets3 = svmul_n_s32_x(pred, indices3, n);

                offsets0 = svadd_n_s32_x(pred, offsets0, j);
                offsets1 = svadd_n_s32_x(pred, offsets1, j);
                offsets2 = svadd_n_s32_x(pred, offsets2, j);
                offsets3 = svadd_n_s32_x(pred, offsets3, j);

                svfloat32_t b_vec0 = svld1_gather_s32index_f32(pred, B, offsets0);
                svfloat32_t b_vec1 = svld1_gather_s32index_f32(pred, B, offsets1);
                svfloat32_t b_vec2 = svld1_gather_s32index_f32(pred, B, offsets2);
                svfloat32_t b_vec3 = svld1_gather_s32index_f32(pred, B, offsets3);
                
                // Calculate dot product and accumulate
                svfloat32_t prod0 = svmul_f32_x(pred, a_vec0, b_vec0);
                svfloat32_t prod1 = svmul_f32_x(pred, a_vec1, b_vec1);
                svfloat32_t prod2 = svmul_f32_x(pred, a_vec2, b_vec2);
                svfloat32_t prod3 = svmul_f32_x(pred, a_vec3, b_vec3);

                sum += svaddv_f32(pred, prod0);
                sum += svaddv_f32(pred, prod1);
                sum += svaddv_f32(pred, prod2);
                sum += svaddv_f32(pred, prod3);
            }

            // 处理剩余的完整向量
            for (; k + vl <= nnz_row; k += vl) {
                svbool_t pred = svptrue_b32();
                
                svfloat32_t a_vec = svld1_f32(pred, &A.values[row_start + k]);
                
                svint32_t indices = svld1_s32(pred, &A.col_indices[row_start + k]);
                svint32_t offsets = svmul_n_s32_x(pred, indices, n);
                offsets = svadd_n_s32_x(pred, offsets, j);
                svfloat32_t b_vec = svld1_gather_s32index_f32(pred, B, offsets);
                
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
    matrix_init_zero(C,m,n); 
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
                // 展开4次循环
                for (; k + 4*vl <= nnz_row; k += 4*vl) {
                    // Create predicate
                    svbool_t pred = svptrue_b32();
                    
                    // Load non-zero element values
                    svfloat32_t a_vec0 = svld1_f32(pred, &A.values[row_start + k]);
                    svfloat32_t a_vec1 = svld1_f32(pred, &A.values[row_start + k + vl]);
                    svfloat32_t a_vec2 = svld1_f32(pred, &A.values[row_start + k + 2*vl]);
                    svfloat32_t a_vec3 = svld1_f32(pred, &A.values[row_start + k + 3*vl]);
                    
                    // Use gather load to load elements from matrix B
                    svint32_t indices0 = svld1_s32(pred, &A.col_indices[row_start + k]);
                    svint32_t indices1 = svld1_s32(pred, &A.col_indices[row_start + k + vl]);
                    svint32_t indices2 = svld1_s32(pred, &A.col_indices[row_start + k + 2*vl]);
                    svint32_t indices3 = svld1_s32(pred, &A.col_indices[row_start + k + 3*vl]);

                    svint32_t offsets0 = svmul_n_s32_x(pred, indices0, n);
                    svint32_t offsets1 = svmul_n_s32_x(pred, indices1, n);
                    svint32_t offsets2 = svmul_n_s32_x(pred, indices2, n);
                    svint32_t offsets3 = svmul_n_s32_x(pred, indices3, n);

                    offsets0 = svadd_n_s32_x(pred, offsets0, j);
                    offsets1 = svadd_n_s32_x(pred, offsets1, j);
                    offsets2 = svadd_n_s32_x(pred, offsets2, j);
                    offsets3 = svadd_n_s32_x(pred, offsets3, j);

                    svfloat32_t b_vec0 = svld1_gather_s32index_f32(pred, B, offsets0);
                    svfloat32_t b_vec1 = svld1_gather_s32index_f32(pred, B, offsets1);
                    svfloat32_t b_vec2 = svld1_gather_s32index_f32(pred, B, offsets2);
                    svfloat32_t b_vec3 = svld1_gather_s32index_f32(pred, B, offsets3);
                    
                    // Calculate dot product and accumulate
                    svfloat32_t prod0 = svmul_f32_x(pred, a_vec0, b_vec0);
                    svfloat32_t prod1 = svmul_f32_x(pred, a_vec1, b_vec1);
                    svfloat32_t prod2 = svmul_f32_x(pred, a_vec2, b_vec2);
                    svfloat32_t prod3 = svmul_f32_x(pred, a_vec3, b_vec3);

                    sum += svaddv_f32(pred, prod0);
                    sum += svaddv_f32(pred, prod1);
                    sum += svaddv_f32(pred, prod2);
                    sum += svaddv_f32(pred, prod3);
                }

                // 处理剩余的完整向量
                for (; k + vl <= nnz_row; k += vl) {
                    svbool_t pred = svptrue_b32();
                    
                    svfloat32_t a_vec = svld1_f32(pred, &A.values[row_start + k]);
                    
                    svint32_t indices = svld1_s32(pred, &A.col_indices[row_start + k]);
                    svint32_t offsets = svmul_n_s32_x(pred, indices, n);
                    offsets = svadd_n_s32_x(pred, offsets, j);
                    svfloat32_t b_vec = svld1_gather_s32index_f32(pred, B, offsets);
                    
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
    int dist = -1;

    int opt;
    while ((opt = getopt(argc, argv, "s:d:")) != -1) {
        switch (opt) { 
            case 's':
                sp = std::stoi(optarg); 
                break;
            case 'd':
                dist = std::stoi(optarg); 
                break;
            default:
                std::cerr << "usage: -s <sparsity> -d <distance>" << std::endl;
                return 1;
        }
    }

    if(sp == -1 && dist == -1) {
        std::cerr << "usage: -s <sparsity> -d <distance>" << std::endl;
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
    if(sp != -1) {
        matrix_init_csr(A, M, K, 666, sp);  // Initialize sparse matrix in CSR format
    }
    else if(dist != -1) {
        matrix_init_csr_with_dist(A,M,K,666,dist);
    }
     else {
        std::cerr << "usage: -s <sparsity> -d <distance>" << std::endl;
        return 1;
    }

    clock_t start_time, end_time, total_time;
    
    // Test single-core version
    start_time = clock();
    for(int i = 0; i < 5; i++) {
        spmm_sve_single(A, B, C_single, M, K, N);
    }
    end_time = clock();
    total_time = end_time - start_time;
    std::cout << "Single-core SpMM took " << (double)total_time / CLOCKS_PER_SEC/5 
              << " seconds to execute. Sparsity: " << sp << " Distance: " << dist << std::endl;

    // Test multi-core version
    start_time = clock();
    for(int i = 0; i < 5; i++) {
        spmm_sve_multi(A, B, C_multi, M, K, N, num_threads);
    }
    end_time = clock();
    total_time = end_time - start_time;
    std::cout << "Multi-core SpMM (" << num_threads << " threads) took " 
              << (double)total_time / CLOCKS_PER_SEC/5
              << " seconds to execute. Sparsity: " << sp<< " Distance: " << dist  << std::endl;

    // Optional: print result matrices for verification
    // std::cout<<"A: "<<std::endl;
    // print_CSR_matrix_dense(A);
    // std::cout<<"B: "<<std::endl;
    // print_matrix(B, K, N);
    // std::cout<<"C: "<<std::endl;
    // print_matrix(C_single, M, N);
    // std::cout<<"--------------------------------"<<std::endl;
    // print_matrix(C_multi, M, N);

    // Check results
    check_result_CSR(A, B, C_single, M, K, N);
    check_result_CSR(A, B, C_multi, M, K, N);

    return 0;
}