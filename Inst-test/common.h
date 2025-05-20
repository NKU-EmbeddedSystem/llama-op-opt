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
    std::vector<std::vector<int>>& index_row, std::vector<std::vector<int>>& index_col){
    srand(seed);
    
    // 初始化向量大小
    index_row.resize(m);
    index_col.resize(m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j ++) {
            if(rand()%100 < threshold){
                mat[i*n+j] = 0;
                continue;
            }
		    mat[i*n+j] = (float)rand() / (float)RAND_MAX;
            index_row[i].push_back(i*n+j);
            index_col[i].push_back(j*n);
        }
    }
}

// initialize a sparse matrix with pramas(float* mat, int64_t m, int64_t n, unsigned int seed, int threshold, std::vector<std::vector<float>>& value,std::vector<std::vector<int>>& index_col)
// A_value A_index_col
void matrix_init_sparse_vc(float* mat, int64_t m, int64_t n, unsigned int seed, int threshold, 
    std::vector<std::vector<float>>& value, std::vector<std::vector<int>>& index_col) {
    srand(seed);
    memset(mat, 0, sizeof(float)*m*n);
    
    for (int i = 0; i < m; i++) {
        value.push_back(std::vector<float>());
        index_col.push_back(std::vector<int>());
        for (int j = 0; j < n; j++) {
            if(rand()%100 < threshold){
                continue;
            }
            float val = (float)rand() / (float)RAND_MAX;
            mat[i*n+j] = val;
            value[i].push_back(val);
            index_col[i].push_back(j*n);
        }
    }
}

// initialize a sparse matrix with pramas(float* mat, int64_t m, int64_t n, unsigned int seed, int threshold ,std::vector<std::vector<int>>& index_col)
void matrix_init_sparse_c(float* mat, int64_t m, int64_t n, unsigned int seed, int threshold,
    std::vector<std::vector<int>>& index_col) {
    srand(seed);
    memset(mat, 0, sizeof(float)*m*n);
    
    for (int i = 0; i < m; i++) {
        index_col.push_back(std::vector<int>());
        for (int j = 0; j < n; j++) {
            if(rand()%100 < threshold){
                continue;
            }
            float val = (float)rand() / (float)RAND_MAX;
            mat[i*n+j] = val;
            index_col[i].push_back(j);
        }
    }
}


//初始化CSR稀疏矩阵
void matrix_init_csr(CSRMatrix& mat, int64_t m, int64_t n, unsigned int seed, int threshold) {
    srand(seed);
    
    mat.rows = m;
    mat.cols = n;
    mat.row_ptrs.resize(m + 1, 0);

    std::vector<float> values;
    std::vector<int> col_indices;
    
    for (int i = 0; i < m; i++) {
        mat.row_ptrs[i] = values.size();
        for (int j = 0; j < n; j++) {
            if (rand() % 100 >= threshold) {
                values.push_back((float)rand() / RAND_MAX);
                col_indices.push_back(j);
            }
        }
    }
    mat.row_ptrs[m] = values.size();
    
    mat.values = std::move(values);
    mat.col_indices = std::move(col_indices);
}

//初始化CSR稀疏矩阵,distance表示字节间隔
void matrix_init_csr_with_dist(CSRMatrix& mat, int64_t m, int64_t n, unsigned int seed, int distance) {
    srand(seed);
    
    mat.rows = m;
    mat.cols = n;
    mat.row_ptrs.resize(m + 1, 0);

    std::vector<float> values;
    std::vector<int> col_indices;
    int item_num = distance/4;

    for (int i = 0; i < m; i++) {
        mat.row_ptrs[i] = values.size();
        for (int j = 0; j < n; j++) {
            if (distance == 0 || (i*n+j)%item_num == 0) {
                values.push_back((float)rand() / RAND_MAX);
                col_indices.push_back(j);
            }
        }
    }
    mat.row_ptrs[m] = values.size();
    
    mat.values = std::move(values);
    mat.col_indices = std::move(col_indices);
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
