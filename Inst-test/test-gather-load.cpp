#include <arm_sve.h>
#include <time.h>
#include <assert.h>
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
void matrix_mul_mat_SVE(float* matrix_a, float* matrix_b, float* matrix_c,int m, int k, int n,
    std::map<int,std::vector<int>>& index_row,std::map<int,std::vector<int>>& index_col){
        
    assert(index_row.size()==index_col.size());
    
    int vl = svcntw();
    
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){            
            // mod : part that is not enough for vl
            int index_num = index_row[i].size();
            int vec_num = index_num/vl;
            int mod = index_num%vl;
            float res = 0.0f;

            const svbool_t pg_full = svptrue_b32();

            for(int vec_i=0;vec_i<vec_num;vec_i++){               
                // load vector_a from matrix_a
                svint32_t indices_a = svld1_s32(pg_full,index_row[i].data() + vec_i*vl);
                svfloat32_t va = svld1_gather_s32index_f32(pg_full, matrix_a, indices_a);
                // load vector_b from matrix_b
                svint32_t indices_b_base = svld1_s32(pg_full, index_col[i].data() + vec_i*vl);
                svint32_t indices_b = svadd_n_s32_z(pg_full, indices_b_base ,(int32_t)j);
                svfloat32_t vb = svld1_gather_s32index_f32(pg_full, matrix_b, indices_b);

                svfloat32_t vprod = svmul_f32_z(pg_full,va,vb);
                res += svaddv_f32(pg_full,vprod);
            }

            // final part
            if(mod){
                // set first mod elements to True
                const svbool_t pg = svwhilelt_b32(0,mod);
                // load vector_a from matrix_a
                svint32_t indices_a = svld1_s32(pg, index_row[i].data());
                svfloat32_t va = svld1_gather_s32index_f32(pg, matrix_a, indices_a);
                // load vector_b from matrix_b
                svint32_t indices_b_base = svld1_s32(pg, index_col[i].data());
                svint32_t indices_b = svadd_n_s32_z(pg, indices_b_base ,(int32_t)j);
                svfloat32_t vb = svld1_gather_s32index_f32(pg, matrix_b, indices_b);

                svfloat32_t vprod = svmul_f32_z(pg,va,vb);
                res += svaddv_f32(pg,vprod);
            }
            matrix_c[i*n+j] = res;
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
    
    for(int i = 0;i<100;i++) {
        matrix_mul_mat_SVE(A, B, C, M, K, N,index_row,index_col);
    } 

    end_time = clock();

    total_time = end_time - start_time;
    std::cout<<"Sparse gather mul_mat took "<< (double)total_time / CLOCKS_PER_SEC/10000 << "seconds to execute."<<std::endl;
    print_matrix(A, M, K);
    print_matrix(B, K, N);
    print_matrix(C, M, N);
    return 0;
}