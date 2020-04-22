#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>      // timing library
#include <cassert>     // assert()
#include <numeric>     // std::accumulate()
#include <iostream>
#include "omp.h" // omp_set_num_threads

#define INF 99999
#define v 3500
#define THREAD_NUM 512

int num = 0;
int matrix [v][v];

void graph(const char * filename);
void display(int arr[], int s);

/* Graph struct for storing and calculating*/
void graph(const char * filename)
{
   FILE * fp;
   fp = fopen(filename, "r");

   int row, col;
   for (row = 0; row < v; row ++)
   {
       for (col = 0; col < v; col ++)
       {
           fscanf(fp, "%d", &matrix[row][col]);
           if (matrix[row][col] != 0)
           {
               num ++;
           }
       }
   }
}

void edge(int (*edges)[3])
{
   int e = 0;
   for (int i = 0; i < v; ++ i) {
      for (int j = 0; j < v; ++ j) {
         if (matrix[i][j] == 1  && e < num) {
            edges[e][0] = i;
            edges[e][1] = j;
            edges[e][2] = matrix[i][j];
            e ++;
         }
      }
   }
}

__global__
void parallel( int *d_edges, int *d_dist, int *d_p, size_t n )
{
    int const index = threadIdx.x + blockIdx.x * blockDim.x;

    if( index <= n )
    {
        int m = d_edges[index * 3];
        int n = d_edges[index * 3 + 1];
        int w = d_edges[index * 3 + 2];

        if (d_dist[m] != 99999  && d_dist[n] > d_dist[m] + w )
        {
           d_dist[n] = d_dist[m] + w;
        }
    }
}

void bellmanford (int (*edge)[3], int s)
{
   int V = v; // num of vertices
   int d[V]; // dist array

   #pragma omp parallel for
   for (int i = 0; i < V; ++ i) {
       d[i] = INF;
   }

   d[s] = 0;
   int *d_edges, *d_dist, * d_p;

   cudaMalloc(&d_p, sizeof(int) * V);
   cudaMalloc(&d_dist, sizeof(int) * V);
   cudaMalloc(&d_edges, sizeof(int) * num * 3);

   cudaMemcpy(d_edges, edge, sizeof(int) * num * 3, cudaMemcpyHostToDevice);

   int blocks = (num + THREAD_NUM - 1) / THREAD_NUM;

   for (int i = 1; i <= V - 1; ++ i) {
       cudaMemcpy(d_dist, d, sizeof(int) * V, cudaMemcpyHostToDevice);
       parallel<<<blocks, THREAD_NUM>>> (d_edges, d_dist, d_p, num);
       cudaMemcpy(&d, d_dist, sizeof(int) * v, cudaMemcpyDeviceToHost);
   }

   #pragma omp parallel for   
   for (int i = 0; i < num; ++ i)
   {
      int m = edge[i][0];
      int n = edge[i][1];
      int w = edge[i][2];

      if (d[m] != INF && d[m] > d[n] + w)
      {
          printf("Negtive weight cycle detected");
          return;
      }
   }

   //display(d, v);  
   cudaFree(d_dist);
   cudaFree(d_edges);
}

/**
 * display the graph
*/
void display (int arr[], int size)
{
     for (int i = 0; i < size; ++ i)
     {
        if (arr[i] == 99999) printf("I ");
        else printf("%d ", arr[i]);
     }
     printf("\n");
}


int main(int argc, char ** argv)
{

   int deviceCount;
   cudaGetDeviceCount(&deviceCount);
   for (int i = 0; i < deviceCount; i++)
   {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);

        std::cout << "Device " << i << ": " << deviceProps.name << "\n"
                  << "SMs: " << deviceProps.multiProcessorCount << "\n"
                  << "Global mem: " << static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024) << "GB \n"
                  << "CUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n";
   }

   const char * filename = argv[1];
   graph(filename);

   int edges[num][3];
   edge(edges);

   auto const trials = 1u;
   auto const start_time = std::chrono::system_clock::now();
   for (int i = 0; i < v; i ++)
   {
      bellmanford(edges, i);
   }
   auto const end_time = std::chrono::system_clock::now();
   auto const elapsed_time = std::chrono::duration_cast< std::chrono::microseconds >( end_time - start_time );
   std::cout << "time: "<< elapsed_time.count() / static_cast< float >( trials )<< " us" << std::endl;
   return 0;
}
              
