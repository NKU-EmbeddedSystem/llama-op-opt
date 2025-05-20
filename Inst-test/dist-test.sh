RESULT_PATH=$PWD/res

if [ ! -d "res" ];then
  mkdir res
  else
  rm -r -f ./res
  mkdir res
fi

distance=(0 32 64 128 256 512 1024 2048 4096 5120)

#=====================================================================================
if [ ! -d "./res/perf_gather_load_super_large.txt" ];then
for dist in "${distance[@]}"
    do
    ./build/test-CSR-spMM-LARGE -d $dist >> ./res/perf_CSR_spMM_large_dist.txt
    done
fi
