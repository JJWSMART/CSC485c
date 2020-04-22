#include <iostream>
#include <stdio.h>
#include <chrono>      // timing library
#include <cassert>     // assert()
#include <numeric>     // std::accumulate()

#define v 10 // num of vertex

/**************************************
 *      change the v while tesing     * 
 *        v = 10.txt                  *
 *        v = 1000.txt                *
 *        v = 3000.txt                *
 **************************************/
int matrix[v][v];

void graph(const char * filename);
void display(int * mat, int N);

/**
 * graph generator
*/
void graph(const char * filename)
{
   FILE * fp;
   fp = fopen(filename, "r");

   int row, col;
   for (row = 0; row < v; row ++){
       for (col = 0; col < v; col ++){
           fscanf(fp, "%d", &matrix[row][col]);
       }
   }
}

/**
 * showing graph
*/
void display(int * mat, int N)
{
    for (int i = 0; i < N; ++ i)
    {
       for (int j = 0; j < N; ++ j)
       {
           printf("%d ", mat[ i*N + j ]);
       }
       printf("\n");
    }
}

__global__ void FloydWarshall(int* mat, int k, int N)
{
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < N && j < N)
    {
        const unsigned int kj = k * N + j;
        const unsigned int ij = i * N + j;
        const unsigned int ik = i * N + k;

        if (mat[ik] != 0 && mat[kj] != 0)
        {
            int t1 = mat[ik] + mat[kj];
            int t2 = mat[ij];

            if (i == j) mat[ij] = 0;
            else if (mat[ij] == 0) mat[ij] = t1;
            else mat[ij] = (t1 < t2) ? t1 : t2;
        }
    }
}

__host__ void floyd_warshall_driver(int * m, int N, dim3 thread_per_block)
{
    int* cuda_m;
    int size = sizeof(int) * v * v;

    cudaMalloc((void**) &cuda_m, size);
    cudaMemcpy(cuda_m, m, size, cudaMemcpyHostToDevice);

    dim3 num_block(ceil(1.0*N/thread_per_block.x),
                   ceil(1.0*N/thread_per_block.y));

    for (int k = 0; k < v; ++ k) {
        FloydWarshall<<< num_block, thread_per_block >>>(cuda_m, k, v);
        cudaThreadSynchronize();
    }

    cudaMemcpy(m, cuda_m, size, cudaMemcpyDeviceToHost);

    display(m, v);

    cudaFree(cuda_m);
}

int main(int argc, char * argv[])
{
   /************************************************************************************************************************************/
    
   int deviceCount;
   cudaGetDeviceCount(&deviceCount);

   for (int i = 0; i < deviceCount; i++) {

        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);

        std::cout << "Device " << i << ": " << deviceProps.name << "\n"
                  << "SMs: " << deviceProps.multiProcessorCount << "\n"
                  << "Global mem: " << static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024) << "GB \n"
                  << "CUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n";
   }
   
   /* _______________________________________check out the answer comment the line above _______________________________________________*/

   const char * filename = argv[1];

   graph(filename);

   int * result = (int*)malloc(sizeof(int)*v*v);
   memcpy(result, matrix, sizeof(int)*v*v);

   auto const num_trials = 1u;
   auto const start_time = std::chrono::system_clock::now();

   floyd_warshall_driver(result, v, 100);

   auto const end_time = std::chrono::system_clock::now();
   auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

   // check out the correctness of the out put comment the line below out
   std::cout << "time: " << ( elapsed_time.count() / static_cast< float >( num_trials ) ) << " us" << std::endl;
   return 0;
}

