RESULT_PATH=$PWD/res

if [ ! -d "res" ];then
  mkdir res
  else
  rm -r -f ./res
  mkdir res
fi

touch ./res/perf_gather_load.txt
touch ./res/perf_simple.txt
touch ./res/perf_simple_sve.txt

sparsity=(0 10 20 30 40 50 60 70 80 90 100)

for sp in "${sparsity[@]}"
    do
    ./build/test-gather-load -s $sp >> ./res/perf_gather_load.txt
    done

for sp in $sparsity
    do
    ./build/test-simple -s $sp >> ./res/perf_simple.txt
    done

for sp in $sparsity
    do
    ./build/test-simple-sve -s $sp >> ./res/perf_simple_sve.txt
    done