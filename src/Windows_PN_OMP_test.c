/*

gcc perlin_noise_omp.c -o perlin_noise_omp -lm -fopenmp
./perlin_noise_omp

*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants
#define WIDTH 10000
#define HEIGHT 10000
#define CHANNELS 3

// Interpolation function
float lerp(float a, float b, float t)
{
    return (a + (b - a) * t)*5;
}

// Smoothstep function
float smoothstep(float t)
{
    return t * t * (3 - 2 * t);
}

// Generate random gradient vectors
void generate_gradients(float *grad_x, float *grad_y, int width, int height) {
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();
        srand(seed);
        #pragma omp for
        for (int i = 0; i < width * height; i++) {
            float angle = (float) rand() / RAND_MAX * 2 * M_PI;
            grad_x[i] = cos(angle);
            grad_y[i] = sin(angle);
        }
    }
}

// Perlin noise function
float perlin(float x, float y, float *grad_x, float *grad_y, int width, int height)
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

int main(int argc, char *argv[])
{
    double startTime, elapsedTime;
    int x;
    int nThreads = 10;
    omp_set_num_threads(nThreads);

    // Allocate the data array on the heap
    unsigned char *data = (unsigned char *)malloc(WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char));

    float *grad_x = malloc(WIDTH * HEIGHT * sizeof(float));
    float *grad_y = malloc(WIDTH * HEIGHT * sizeof(float));

    startTime = omp_get_wtime();
    generate_gradients(grad_x, grad_y, WIDTH, HEIGHT);

    float scale = 2500.0;
#pragma omp parallel for num_threads(nThreads) private(x) 
    for (int y = 0; y < HEIGHT; y++)
    {
        for (x = 0; x < WIDTH; x++)
        {
            float value = perlin(x / scale, y / scale, grad_x, grad_y, WIDTH, HEIGHT);

            value = (value + 1) / 2.0;
            unsigned char color = (unsigned char)(value * 255);

            int index = (y * WIDTH + x) * CHANNELS;
            data[index] = color;
            data[index + 1] = color;
            data[index + 2] = color;
        }
    }

    elapsedTime = omp_get_wtime() - startTime;
    printf("Parallel time to generate image: %f\n", elapsedTime);

    stbi_write_jpg("output_omp.jpg", WIDTH, HEIGHT, CHANNELS, data, WIDTH * CHANNELS);

    // Free the allocated memory
    free(data);
    free(grad_x);
    free(grad_y);

    return 0;
}
