# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Use NVCC.
# Set the appropriate GPU architecture, check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
CXX=nvcc

#FLAGS = -O2 -std=c++14 -arch=sm_86 -lcublas -lcusparse -g -lcurand -G #debug
FLAGS = -Xptxas -O3, -Xcompiler -O3,-O3 --use_fast_math -std=c++14 -arch=sm_86 -lcublas -lcurand -lcusparse#optimized

#-v verbose compilation,
#-Xptxas means Specify options directly to ptxas, the PTX optimizing assembler.
#use -g or --debug for debug and -G or --device-debug for device debug
#About -G: If --dopt is not specified, then this option turns off all optimizations on device code. It is not intended for profiling; use --generate-line-info instead for profiling.

BIN_FOLDER=bin
SRC_FOLDER=src
FILES=${SRC_FOLDER}/main.cu ${SRC_FOLDER}/mmio.cpp ${SRC_FOLDER}/benchmark.cu
GPU_FILES=${SRC_FOLDER}/benchmarks/vector_sum.cu ${SRC_FOLDER}/benchmarks/matrix_multiplication.cu ${SRC_FOLDER}/benchmarks/personalized_pagerank.cu ${SRC_FOLDER}/benchmarks/pprImplementations/implementation.cu
IMP0=${SRC_FOLDER}/benchmarks/pprImplementations/naiveImplementation.cu
IMP1=${SRC_FOLDER}/benchmarks/pprImplementations/cublasCusparseNaiveImplementation.cu
IMP2=${SRC_FOLDER}/benchmarks/pprImplementations/improvedImplementation.cu
IMP3=${SRC_FOLDER}/benchmarks/pprImplementations/fastImplementation.cu
IMP4=${SRC_FOLDER}/benchmarks/pprImplementations/mcCompletePath.cu
IMP5=${SRC_FOLDER}/benchmarks/pprImplementations/finalImplementation.cu

.PHONY: all clean launch imp0 imp1 imp2 imp3 imp4 imp5

all:#Will compile all the implementations
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP0) $(IMP1) $(IMP2) $(IMP3) $(IMP4) $(IMP5) $(FLAGS) -DIMP0 -DIMP1 -DIMP2 -DIMP3 -DIMP4 -DIMP5 -o $(BIN_FOLDER)/b;

imp0:#The first implementation we made
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP0) $(FLAGS) -DIMP0 -o $(BIN_FOLDER)/b;

imp1:#Cusparse and cublas naive implementation, we used it to test csr vs coo vs bsr
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP1) $(FLAGS) -DIMP1 -o $(BIN_FOLDER)/b;


imp2:#Improved implementation
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP2) $(FLAGS) -DIMP2 -o $(BIN_FOLDER)/b;

imp3:#Fast implementation using cublas and cusparse coo
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP3) $(FLAGS) -DIMP3 -o $(BIN_FOLDER)/b;

imp4:#Monte Carlo CompletePath algorithm
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP4) $(FLAGS) -DIMP4 -o $(BIN_FOLDER)/b;

imp5:#Final implementation with our kernels
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(GPU_FILES) $(IMP5) $(FLAGS) -DIMP5 -o $(BIN_FOLDER)/b;

clean:
	rm $(BIN_FOLDER)/*;

launch:
	bash bin/b -d -c -b ppr -I 1 -i 30 -t 64 -d;
