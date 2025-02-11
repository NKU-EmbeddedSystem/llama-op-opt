#include <arm_sve.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
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
void matrix_mul_mat_SVE(float* matrix_a, float* matrix_b, float* matrix_c,int m, int k, int n,
    std::map<int,std::vector<int>> index_row,std::map<int,std::vector<int>> index_col){

    assert(index_row.size()==index_col.size());
    
    int vl = svcntw();
    // std::cout<< "vector register length: "<< vl <<" (x 32) bits"<<std::endl;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            assert(index_row[i].size()==index_col[i].size());            
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
                // print_vector(va);
                
                // load vector_b from matrix_b
                svint32_t indices_b_base = svld1_s32(pg_full, index_col[i].data() + vec_i*vl);
                svint32_t indices_b = svadd_n_s32_z(pg_full, indices_b_base ,(int32_t)j);
                svfloat32_t vb = svld1_gather_s32index_f32(pg_full, matrix_b, indices_b);
                // print_vector(vb);

                svfloat32_t vprod = svmul_f32_z(pg_full,va,vb);
                res += svaddv_f32(pg_full,vprod);
            }

            // final part
            if(mod){
                assert(index_row[i].size()==index_col[i].size());            
                // set first mod elements to True
                const svbool_t pg = svwhilelt_b32(0,mod);
                
                // load vector_a from matrix_a
                svint32_t indices_a = svld1_s32(pg, index_row[i].data());
                svfloat32_t va = svld1_gather_s32index_f32(pg, matrix_a, indices_a);
                // print_vector(va);

                // load vector_b from matrix_b
                svint32_t indices_b_base = svld1_s32(pg, index_col[i].data());
                svint32_t indices_b = svadd_n_s32_z(pg, indices_b_base ,(int32_t)j);
                svfloat32_t vb = svld1_gather_s32index_f32(pg, matrix_b, indices_b);
                // print_vector(vb);

                svfloat32_t vprod = svmul_f32_z(pg,va,vb);
                res += svaddv_f32(pg,vprod);
            }
            matrix_c[i*n+j] = res;
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

    // int vl = svcntw();
    // std::cout<< "vector register length: "<< vl <<" (x 32) bits"<<std::endl;

    std::map<int,std::vector<int>> index_row,index_col;
    matrix_init_sparse(A,M,K,666,sp,index_row,index_col);
    matrix_init(B,K,N,888);
    matrix_init_zero(C,M,N);

    clock_t start_time, end_time, total_time;

    start_time = clock();
    for(int i = 0;i<5;i++) {
        matrix_mul_mat_SVE(A, B, C, M, K, N,index_row,index_col);
    } 
    end_time = clock();

    total_time = end_time - start_time;
    // print_matrix(A, M, K);
    // print_matrix(B, K, N);
    // print_matrix(C, M, N);
    std::cout<<"Sparse gather mul_mat took "<< (double)total_time / CLOCKS_PER_SEC/5 << " seconds to execute. Sparsity: "<< sp <<std::endl;
    return 0;
}