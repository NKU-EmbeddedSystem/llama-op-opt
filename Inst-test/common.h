#include <chrono>
#include <arm_sve.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <map>
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