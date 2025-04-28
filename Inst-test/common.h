#include <chrono>
#include <arm_sve.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <map> // fixme: use unordered_map


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


// Data structure for sparse matrix in CSR format
struct CSRMatrix {
    std::vector<float> values;     // Non-zero element values
    std::vector<int> col_indices;  // Column indices of non-zero elements
    std::vector<int> row_ptrs;     // Row pointers
    int rows;                      // Number of rows in matrix
    int cols;                      // Number of columns in matrix
};

// Randomly initialize a matrix ()
// random == 0  ->  init 0
void matrix_init(float* mat, int64_t m, int64_t n, unsigned int seed){
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
		    mat[i*n+j] = (float)rand() / (float)RAND_MAX;
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
void matrix_init_sparse(float* mat, int64_t m, int64_t n, unsigned int seed, int threshold, 
    std::map<int,std::vector<int>>& index_row, std::map<int,std::vector<int>>& index_col){
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
            if(rand()%100 < threshold){
                mat[i*n+j] = 0;
                continue;
            }
		    mat[i*n+j] = (float)rand() / (float)RAND_MAX;
            if(index_row.find(i)==index_row.end()) {
                index_row[i] = std::vector<int>();
                index_col[i] = std::vector<int>();
            }
            index_row[i].push_back(i*n+j);
            index_col[i].push_back(j*n);
        }
    }
    // // Print test
    // std::cout<<"index_row length:"<<index_row.size();
    // for(auto i : index_row){
    //     std::cout<<"row "<< i.first <<":[";
    //     for (int j = 0; j < i.second.size(); j ++) {
    //     std::cout<< i.second[j] <<" ";
    //     }
    //     std::cout<<"]"<<std::endl;
    // }
    // std::cout<<std::endl;
    // std::cout<<"index_col length:"<<index_col.size();
    // for(auto i : index_col){
    //     std::cout<<"row "<< i.first <<":[";
    //     for (int j = 0; j < i.second.size(); j ++) {
    //     std::cout<< i.second[j] <<" ";
    //     }
    //     std::cout<<"]"<<std::endl;
    // }
    // std::cout<<std::endl;
}



//initialize a sparse matrix with CSRMatrix
void matrix_init_csr(CSRMatrix& mat, int64_t m, int64_t n, unsigned int seed, int threshold) {
    srand(seed);
    
    // Initialize basic properties of CSR matrix
    mat.rows = m;
    mat.cols = n;
    mat.row_ptrs.resize(m + 1, 0);

    // Temporary storage for non-zero elements in each row
    std::vector<std::vector<float>> temp_values(m);
    std::vector<std::vector<int>> temp_col_indices(m);

    int nnz = 0; // Total number of non-zero elements

    // Generate random sparse matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (rand() % 100 >= threshold) {  // Non-zero element
                float val = (float)rand() / (float)RAND_MAX;
                temp_values[i].push_back(val);
                temp_col_indices[i].push_back(j);
                nnz++;
            }
        }
    }

    // Allocate storage space
    mat.values.reserve(nnz);
    mat.col_indices.reserve(nnz);

    // Build CSR format
    int current_pos = 0;
    mat.row_ptrs[0] = 0;
    
    for (int i = 0; i < m; i++) {
        // Copy current row data
        mat.values.insert(mat.values.end(), 
                         temp_values[i].begin(), 
                         temp_values[i].end());
        mat.col_indices.insert(mat.col_indices.end(), 
                             temp_col_indices[i].begin(), 
                             temp_col_indices[i].end());
        
        current_pos += temp_values[i].size();
        mat.row_ptrs[i + 1] = current_pos;
    }
}


void  print_vector(svfloat32_t vec){
    int vl = svcntw();
    const svbool_t pg_full = svptrue_b32();
    float* output = new float[vl];
    
    for(int i=0;i<vl;i++){
        output[i]=0;
    }
    svst1(pg_full, output, vec);
    std::cout<<"length:"<<vl<<" [";
    for(int i=0;i<vl;i++){
        std::cout<<output[i]<<",";
    }
    std::cout<<"]\n";

    delete []output;

}

void print_matrix(float* mat, int64_t m, int64_t n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
		   std::cout<<mat[i*n+j]<<" ";
        }
        std::cout<<std::endl;
    } 
}

// Print CSR matrix in dense format
void print_CSR_matrix_dense(const CSRMatrix& mat) {
    for(int i = 0; i < mat.rows; i++) {
        int row_start = mat.row_ptrs[i];
        int row_end = mat.row_ptrs[i+1];
        int curr_col = 0;
        
        // Iterate through each non-zero element in this row
        for(int j = row_start; j < row_end; j++) {
            // Print leading zeros
            while(curr_col < mat.col_indices[j]) {
                std::cout << "0 ";
                curr_col++;
            }
            // Print non-zero element
            std::cout << mat.values[j] << " ";
            curr_col++;
        }
        
        // Print remaining zeros in this row
        while(curr_col < mat.cols) {
            std::cout << "0 ";
            curr_col++;
        }
        std::cout << std::endl;
    }
}

// Check if matrix multiplication result is correct
void check_result(float* matrix_a, float* matrix_b, float* matrix_c, int m, int k, int n) {
    float* expected = new float[m * n]();
    
    // Calculate expected result
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int q = 0; q < k; q++) {
                sum += matrix_a[i * k + q] * matrix_b[q * n + j];
            }
            expected[i * n + j] = sum;
        }
    }

    // Compare results
    bool correct = true;
    float max_diff = 0.0f;
    for(int i = 0; i < m * n; i++) {
        float diff = std::abs(matrix_c[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 1e-4) {
            correct = false;
            break;
        }
    }

    if(correct) {
        std::cout << "Result correct! Maximum error: " << max_diff << std::endl;
    } else {
        std::cout << "Result incorrect! Maximum error: " << max_diff << std::endl;
    }

    delete[] expected;
}
// Check if matrix multiplication result is correct (CSR version)
void check_result_CSR(const CSRMatrix& matrix_a, float* matrix_b, float* matrix_c, int m, int k, int n) {
    float* expected = new float[m * n]();
    
    // Calculate expected result
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            // 遍历当前行的非零元素
            for(int q = matrix_a.row_ptrs[i]; q < matrix_a.row_ptrs[i+1]; q++) {
                int col = matrix_a.col_indices[q];
                sum += matrix_a.values[q] * matrix_b[col * n + j];
            }
            expected[i * n + j] = sum;
        }
    }

    // Compare results
    bool correct = true;
    float max_diff = 0.0f;
    for(int i = 0; i < m * n; i++) {
        float diff = std::abs(matrix_c[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 1e-4) {
            correct = false;
            break;
        }
    }

    if(correct) {
        std::cout << "Result correct! Maximum error: " << max_diff << std::endl;
    } else {
        std::cout << "Result incorrect! Maximum error: " << max_diff << std::endl;
    }

    delete[] expected;
}
