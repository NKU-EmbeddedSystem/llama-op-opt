#include <arm_sve.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include "common.h"

float A[M*K];
float B[K*N];
float C[M*N];

// Simply mul mat two matrix
void matrix_mul_mat_SVE(float* matrix_a, float* matrix_b, float* matrix_c,int m, int k, int n,
    std::map<int,std::vector<int>> index_row,std::map<int,std::vector<int>> index_col){

    assert(index_row.size()==index_col.size());
    
    matrix_init_zero(matrix_c,m,n); 
    int vl = svcntw();
    const svbool_t pg_full = svptrue_b32();
    // std::cout<< "vector register length: "<< vl <<" (x 32) bits"<<std::endl;
    for(int i=0;i<m;i++){
        assert(index_row[i].size()==index_col[i].size());            
        // mod : part that is not enough for vl
        int index_num = index_row[i].size();
        int vec_num = index_num/vl;
        int mod = index_num%vl;
        for(int j=0;j<n;j++){
            float res = 0.0f;

            // 展开4次循环
            int vec_i = 0;
            for(; vec_i<vec_num-3; vec_i+=4){               
                // load vector_a from matrix_a
                svint32_t indices_a0 = svld1_s32(pg_full,index_row[i].data() + vec_i*vl);
                svint32_t indices_a1 = svld1_s32(pg_full,index_row[i].data() + (vec_i+1)*vl);
                svint32_t indices_a2 = svld1_s32(pg_full,index_row[i].data() + (vec_i+2)*vl);
                svint32_t indices_a3 = svld1_s32(pg_full,index_row[i].data() + (vec_i+3)*vl);

                svfloat32_t va0 = svld1_gather_s32index_f32(pg_full, matrix_a, indices_a0);
                svfloat32_t va1 = svld1_gather_s32index_f32(pg_full, matrix_a, indices_a1);
                svfloat32_t va2 = svld1_gather_s32index_f32(pg_full, matrix_a, indices_a2);
                svfloat32_t va3 = svld1_gather_s32index_f32(pg_full, matrix_a, indices_a3);
                
                // load vector_b from matrix_b
                svint32_t indices_b_base0 = svld1_s32(pg_full, index_col[i].data() + vec_i*vl);
                svint32_t indices_b_base1 = svld1_s32(pg_full, index_col[i].data() + (vec_i+1)*vl);
                svint32_t indices_b_base2 = svld1_s32(pg_full, index_col[i].data() + (vec_i+2)*vl);
                svint32_t indices_b_base3 = svld1_s32(pg_full, index_col[i].data() + (vec_i+3)*vl);

                svint32_t indices_b0 = svadd_n_s32_z(pg_full, indices_b_base0 ,(int32_t)j);
                svint32_t indices_b1 = svadd_n_s32_z(pg_full, indices_b_base1 ,(int32_t)j);
                svint32_t indices_b2 = svadd_n_s32_z(pg_full, indices_b_base2 ,(int32_t)j);
                svint32_t indices_b3 = svadd_n_s32_z(pg_full, indices_b_base3 ,(int32_t)j);

                svfloat32_t vb0 = svld1_gather_s32index_f32(pg_full, matrix_b, indices_b0);
                svfloat32_t vb1 = svld1_gather_s32index_f32(pg_full, matrix_b, indices_b1);
                svfloat32_t vb2 = svld1_gather_s32index_f32(pg_full, matrix_b, indices_b2);
                svfloat32_t vb3 = svld1_gather_s32index_f32(pg_full, matrix_b, indices_b3);

                svfloat32_t vprod0 = svmul_f32_z(pg_full,va0,vb0);
                svfloat32_t vprod1 = svmul_f32_z(pg_full,va1,vb1);
                svfloat32_t vprod2 = svmul_f32_z(pg_full,va2,vb2);
                svfloat32_t vprod3 = svmul_f32_z(pg_full,va3,vb3);

                res += svaddv_f32(pg_full,vprod0);
                res += svaddv_f32(pg_full,vprod1);
                res += svaddv_f32(pg_full,vprod2);
                res += svaddv_f32(pg_full,vprod3);
            }

            // 处理剩余的完整向量部分
            for(; vec_i<vec_num; vec_i++){               
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
                assert(index_row[i].size()==index_col[i].size());            
                // set first mod elements to True
                const svbool_t pg = svwhilelt_b32(0,mod);
                
                // load vector_a from matrix_a
                svint32_t indices_a = svld1_s32(pg, index_row[i].data() + vec_num*vl);
                svfloat32_t va = svld1_gather_s32index_f32(pg, matrix_a, indices_a);

                // load vector_b from matrix_b
                svint32_t indices_b_base = svld1_s32(pg, index_col[i].data() + vec_num*vl);
                svint32_t indices_b = svadd_n_s32_z(pg, indices_b_base ,(int32_t)j);
                svfloat32_t vb = svld1_gather_s32index_f32(pg, matrix_b, indices_b);

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
    matrix_init(B,K,N,888);
    matrix_init_zero(C,M,N);
    matrix_init_sparse(A,M,K,666,sp,index_row,index_col);

    clock_t start_time, end_time, total_time;

    start_time = clock();
    for(int i = 0;i<5;i++) {
        matrix_mul_mat_SVE(A, B, C, M, K, N,index_row,index_col);
    } 
    end_time = clock();

    total_time = end_time - start_time;
    // Optional: print result matrices for verification
    // std::cout<<"A: "<<std::endl;
    // print_matrix(A, M, K);
    // std::cout<<"B: "<<std::endl;
    // print_matrix(B, K, N);
    // std::cout<<"C: "<<std::endl;
    // print_matrix(C, M, N);
    // check_result(A, B, C, M, K, N);
    std::cout<<"Sparse gather mul_mat took "<< (double)total_time / CLOCKS_PER_SEC/5 << " seconds to execute. Sparsity: "<< sp <<std::endl;
    return 0;
}