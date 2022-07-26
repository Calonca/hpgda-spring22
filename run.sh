#!/bin/bash
#Imprementations in order of speed:

### 1. MC Complete path ###
# Fastest implementation, uses the Monte Carlo complete path method
# Works well for the Wikipedia graph, not well for the California graph
# The number of walkers is decided by blocks*threads, reducing them will reduce the accuracy
# m is number of steps that each thread makes, reducing it will reduce the accuracy
# if a value less than 50 is given, it will be set to 50
bin/b -c -b ppr -I 4 -i 100 -t 128 -B 4000 -g data/wikipedia.mtx

### 2. Final Coo power method implementation ###
# Implementation without libraries, uses the power method and an Heuristic for early stopping
# Works well on both graphs given the proper heuristic parameters
#bin/b -c -b ppr -I 5 -i 100 -t 128 -B 4000 -g data/wikipedia.mtx

### 3. Fast coo with cuSparse ###
# It uses the power method, cuSparse for the coo, and cublas for the dot products
#bin/b -c -b ppr -I 3 -i 100 -t 128 -B 4000 -g data/wikipedia.mtx

### 4. Naive improved ###
# Cublas and CuSparse implementation
#bin/b -c -b ppr -I 2 -i 100 -t 128 -B 4000 -g data/wikipedia.mtx

### 5. Oldest naive implementation ###
# Our first implementation, uses the power method
#bin/b -c -b ppr -I 0 -i 100 -t 128 -B 4000 -g data/wikipedia.mtx

### 6. Cublas CuSparse naive ###
# Test implementation for Cublas and CuSparse, uses the power method with a bsr matrix
# Used to test comparisons between bsr, csr and coo
#bin/b -c -b ppr -I 1 -i 100 -t 128 -B 4000 -g data/wikipedia.mtx
