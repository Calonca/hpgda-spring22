nsys profile --stats=true -t nvtx,cuda  -o srun ./bin/b -b ppr -I 4 -i 100 -t 128 -B 500 -g data/wikipedia.mtx
#cuda-memcheck ./bin/b -d -c -b ppr -I 4 -i 2 -t 256 -m 80 -B 40 |more
#nsight-sys

#usage
#srun <srun args> ncu -o <filename> <other Nsight args> <code> <args>

#sudo /usr/local/NVIDIA-Nsight-Compute/ncu ./bin/b -c -b ppr -I 0 -i 10 -t 1024 -B 10000 -g data/wikipedia.mtx

#sudo /usr/local/NVIDIA-Nsight-Compute/ncu --set detailed -o ncuProf/prof1 ./bin/b -b ppr -I 0 -i 1 -m 100 -t 512 -B 200000 -g data/wikipedia.mtx
