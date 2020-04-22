#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "math.h"      // sqrtf()
#include <chrono>      // timing library
#include <cassert>     // assert()
#include <numeric>     // std::accumulate()
#include <iostream>

#define BUFF 1000
#define INF 99999
#define v 3500  

int num = 0;
int matrix [v][v];

void graph(const char * filename);
void display(int arr[], int s);

/**
 * @para filename
*/
void graph(const char * filename)
{
   FILE * fp;
   fp = fopen(filename, "r");

   int row, col;
   for (row = 0; row < v; row ++) {
       for (col = 0; col < v; col ++) {
           fscanf(fp, "%d", &matrix[row][col]);
           if (matrix[row][col] != 0) { num ++; }
       }
   }
}

/**
 * @para edges
 * adding eddges
*/
void edge(int (*edges)[3]) {
   int e = 0;
   for (int i = 0; i < v; ++ i) {
      for (int j = 0; j < v; ++ j) {
         if (matrix[i][j] != 0  && e < num)
         {
            edges[e][0] = i;
            edges[e][1] = j;
            edges[e][2] = matrix[i][j];
            e ++;
         }
      }
   }
}


/**
 * bellmanford algorithm
*/
void bellmanford (int (*edge)[3], int s)
{
   int tV = v; // num of vertices
   int rE = num; // num of edges
   int d[tV];

   for (int i = 0; i < tV; ++ i) {
       d[i] = INF;
   }

   d[s] = 0;
   int m, n, w;

   for (int i = 1; i <= tV-1; ++ i) {
      for (int j = 0; j < rE; ++ j){
          m = edge[j][0];
          n = edge[j][1];
          w = edge[j][2];
          if (d[m] != INF && d[n] > d[m] + w) {
             d[n] = d[m] + w;
          }
      }
   }


   for (int i = 0; i < num; ++ i) {
       m = edge[i][0];
       n = edge[i][1];
       w = edge[i][2];

       if (d[m] != INF && d[m] > d[n] + w) {
          printf("Negtive weight cycle detected");
          return;
       }
   }

   //display(d, v);
}

/**
 * @para 1d array
*/
void display (int arr[], int size)
{
     int i;
     for (i = 0; i < size; ++ i)
     {
        if (arr[i] == 99999){ printf("I "); }
        else{ printf("%d ", arr[i]); }
     }
     printf("\n");
}

int main(int argc, char * argv[])
{
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
   const char * filename = argv[1];
   graph(filename);

   int edges[num][3];
   edge(edges);

   auto const start_time = std::chrono::system_clock::now();

   for (int i = 0; i < v; i ++) bellmanford(edges, i);

   auto const end_time = std::chrono::system_clock::now();
   auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );
   std::cout << "time: " << ( elapsed_time.count() / static_cast< float >( 1u ) ) << " us" << std::endl;

   return 0;
}

