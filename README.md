# IT3882DPNG
2d Perlin Noise Generator

## Sequential
compile with 
```bash
gcc -g -Wall perlin_noise_seq.c -o perlin_noise_seq -lm
./perlin_noise_seq
```

## Openmp
```bash
gcc -g -Wall perlin_noise_omp.c -o perlin_noise_omp -lm -fopenmp
./perlin_noise_omp
```

## Using the Makefile

### Compile everything
```bash
make all
```

### Compile sequential program
```bash
make seq
```

### Compile parallel omp program
```bash
make omp
```

### Run everything
```bash
make run-all
```

### Run sequential program
```bash
make run-seq
```

### Run parallel omp program
```bash
make run-omp
```