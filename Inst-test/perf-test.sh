RESULT_PATH=$PWD/res

if [ ! -d "res" ];then
  mkdir res
#   else
#   rm -r -f ./res
#   mkdir res
fi

sparsity=(0 10 20 30 40 50 60 70 80 90 100)

# if [ ! -d "./res/perf_gather_load_test.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-gather-load-TEST -s $sp >> ./res/perf_gather_load_test.txt
#     done
# fi

# if [ ! -d "./res/perf_simple_test.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-simple-TEST -s $sp >> ./res/perf_simple_test.txt
#     done
# fi

# if [ ! -d "./res/perf_simple_sve_test.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-simple-sve-TEST -s $sp >> ./res/perf_simple_sve_test.txt
#     done
# fi

# #=====================================================================================
# if [ ! -d "./res/perf_gather_load_small.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-gather-load-SMALL -s $sp >> ./res/perf_gather_load_small.txt
#     done
# fi

# if [ ! -d "./res/perf_simple_small.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-simple-SMALL -s $sp >> ./res/perf_simple_small.txt
#     done
# fi

# if [ ! -d "./res/perf_simple_sve_small.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-simple-sve-SMALL -s $sp >> ./res/perf_simple_sve_small.txt
#     done
# fi

# #=====================================================================================
# if [ ! -d "./res/perf_gather_load_medium.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-gather-load-MEDIUM -s $sp >> ./res/perf_gather_load_medium.txt
#     done
# fi

# if [ ! -d "./res/perf_simple_medium.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-simple-MEDIUM -s $sp >> ./res/perf_simple_medium.txt
#     done
# fi

# if [ ! -d "./res/perf_simple_sve_medium.txt" ];then
# for sp in "${sparsity[@]}"
#     do
#     ./build/test-simple-sve-MEDIUM -s $sp >> ./res/perf_simple_sve_medium.txt
#     done
# fi

#=====================================================================================
if [ ! -d "./res/perf_gather_load_large.txt" ];then
for sp in "${sparsity[@]}"
    do
    ./build/test-gather-load-LARGE -s $sp >> ./res/perf_gather_load_large.txt
    done
fi

if [ ! -d "./res/perf_simple_large.txt" ];then
for sp in "${sparsity[@]}"
    do
    ./build/test-simple-LARGE -s $sp >> ./res/perf_simple_large.txt
    done
fi

if [ ! -d "./res/perf_simple_sve_large.txt" ];then
for sp in "${sparsity[@]}"
    do
    ./build/test-simple-sve-LARGE -s $sp >> ./res/perf_simple_sve_large.txt
    done
fi

#=====================================================================================
if [ ! -d "./res/perf_gather_load_super_large.txt" ];then
for sp in "${sparsity[@]}"
    do
    ./build/test-gather-load-SUPER_LARGE -s $sp >> ./res/perf_gather_load_super_large.txt
    done
fi

if [ ! -d "./res/perf_simple_super_large.txt" ];then
for sp in "${sparsity[@]}"
    do
    ./build/test-simple-SUPER_LARGE -s $sp >> ./res/perf_simple_super_large.txt
    done
fi

if [ ! -d "./res/perf_simple_sve_super_large.txt" ];then
for sp in "${sparsity[@]}"
    do
    ./build/test-simple-sve-SUPER_LARGE -s $sp >> ./res/perf_simple_sve_super_large.txt
    done
fi