# IT3882DPNG
2d Perlin Noise Generator using multiple implementations
- Sequential
- OpenMP
- CUDA

## Compile all
```bash
make
```
## Run all
```
make run-all
```
## Specific make targets:

### Compile
```
make seq
make omp
make cuda
```
### Run
```
make run-seq
make run-omp
make run-cuda
```

### Run on Expanse
1. Copy all files in directory to expanse
2. Load gpu and cuda modules: `module load gpu; module load cuda`
3. Compile all files: `make all`
3. Submit jobscript: `sbatch perlin_noise_job.sb`
