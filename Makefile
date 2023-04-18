all: dirs seq omp cuda

dirs:
	mkdir -p bin
	mkdir -p out

seq: src/perlin_noise_seq.c
	gcc -g -Wall src/perlin_noise_seq.c -o bin/perlin_noise_seq -lm


omp: src/perlin_noise_omp.c
	gcc-12 -g -Wall src/perlin_noise_omp.c -o bin/perlin_noise_omp -lm -fopenmp

cuda: src/perlin_noise_cuda.cu
	nvcc -g src/perlin_noise_cuda.cu -o bin/perlin_noise_cuda -lm

run-all: run-seq run-omp run-cuda

run-seq: bin/perlin_noise_seq
	bin/perlin_noise_seq 

run-omp: bin/perlin_noise_omp
	bin/perlin_noise_omp 

run-cuda: bin/perlin_noise_cuda
	bin/perlin_noise_cuda 