#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
#define WIDTH 10000
#define HEIGHT 10000
#define CHANNELS 3

// Interpolation function
__device__ float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

// Smoothstep function
__device__ float smoothstep(float t)
{
    return t * t * (3 - 2 * t);
}

// Generate random gradient vectors
__global__ void generate_gradients(float *grad_x, float *grad_y, int width, int height, unsigned long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        curandState state;
        curand_init(seed * (x * height + y), 0, 0, &state);
        int index = y * width + x;
        float angle = curand_uniform(&state) * 2 * M_PI;
        grad_x[index] = cos(angle);
        grad_y[index] = sin(angle);
    }
}

// Perlin noise function
__device__ float perlin(float x, float y, float *grad_x, float *grad_y, int width, int height)
{
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    float fx = x - ix;
    float fy = y - iy;

    ix = ix % width;
    iy = iy % height;

    int ix1 = (ix + 1) % width;
    int iy1 = (iy + 1) % height;

    float dot00 = grad_x[iy * width + ix] * fx + grad_y[iy * width + ix] * fy;
    float dot10 = grad_x[iy * width + ix1] * (fx - 1) + grad_y[iy * width + ix1] * fy;
    float dot01 = grad_x[iy1 * width + ix] * fx + grad_y[iy1 * width + ix] * (fy - 1);
    float dot11 = grad_x[iy1 * width + ix1] * (fx - 1) + grad_y[iy1 * width + ix1] * (fy - 1);

    float sx = smoothstep(fx);
    float sy = smoothstep(fy);

    float a = lerp(dot00, dot10, sx);
    float b = lerp(dot01, dot11, sx);

    return lerp(a, b, sy);
}

__global__ void render_image(unsigned char *data, float *grad_x, float *grad_y, int width, int height, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float value = perlin(x / scale, y / scale, grad_x, grad_y, width, height);

        value = (value + 1) / 2.0;
        unsigned char color = (unsigned char)(value * 255);

        int index = (y * width + x) * CHANNELS;
        data[index] = color;
        data[index + 1] = color;
        data[index + 2] = color;
    }
}

int main(int argc, char *argv[])
{
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate the data array on the device
    unsigned char *d_data;
    cudaMalloc((void **)&d_data, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char));

    float *d_grad_x;
    cudaMalloc((void **)&d_grad_x, WIDTH * HEIGHT * sizeof(float));
    float *d_grad_y;
    cudaMalloc((void **)&d_grad_y, WIDTH * HEIGHT * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start, 0);
    generate_gradients<<<gridSize, blockSize>>>(d_grad_x, d_grad_y, WIDTH, HEIGHT, time(NULL));
    cudaDeviceSynchronize();

    float scale = 10.0;
    render_image<<<gridSize, blockSize>>>(d_data, d_grad_x, d_grad_y, WIDTH, HEIGHT, scale);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CUDA time to generate image: %f ms\n", elapsedTime);

    // Copy the data from the device to the host
    unsigned char *h_data = (unsigned char *)malloc(WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char));
    cudaMemcpy(h_data, d_data, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("out/cuda.jpg", WIDTH, HEIGHT, CHANNELS, h_data, WIDTH * CHANNELS);

    // Free the allocated memory
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_grad_x);
    cudaFree(d_grad_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}