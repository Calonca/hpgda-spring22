#!/bin/bash

bin/b -c -b ppr -I 4 -i 5 -t 128 -B 4000 -m 80 -g data/wikipedia.mtx
#bin/b -c -b ppr -I 0 -i 1 -t 1024 -B 20 -g data/wikipedia.mtx

# Run vector sum;
#bin/b -d -c -n 100000000 -b vec -I 1 -i 30 -t 64;
#bin/b -d -c -n 100000000 -b vec -I 2 -i 30 -t 64;

# Run matrix multiplication;
#bin/b -d -c -n 1000 -b mmul -I 1 -i 30 -t 8;
#bin/b -d -c -n 1000 -b mmul -I 2 -i 30 -t 8 -B 14;