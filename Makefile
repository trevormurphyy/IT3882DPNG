all: sequential omp

seq: src/perlin_noise_seq.c
	gcc -g -Wall src/perlin_noise_seq.c -o bin/perlin_noise_seq -lm


omp: src/perlin_noise_omp.c
	gcc-12 -g -Wall src/perlin_noise_omp.c -o bin/perlin_noise_omp -lm -fopenmp

run-all: run-seq run-omp

run-seq: bin/perlin_noise_seq
	bin/perlin_noise_seq

run-omp: bin/perlin_noise_omp
	bin/perlin_noise_omp