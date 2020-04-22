#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>      // timing library
#include <cassert>     // assert()
#include <numeric>     // std::accumulate()
#include <iostream>

/**************************************
 *      change the v while tesing     * 
 *        v = 10.txt                  *
 *        v = 1000.txt                *
 *        v = 3000.txt                *
 **************************************/
#define v 10 

int num = 0;
int dist[v][v];

void graph(const char * filename);
void printSolution(int dist[][v]);
void FloydWarshall(int dist[][v]);

/*
 * generate graph
*/
void graph(const char * filename)
{
   FILE * fp;
   fp = fopen(filename, "r");

   int row, col;
   for (row = 0; row < v; row ++)
   {
      for (col = 0; col < v; col ++)
      {
         fscanf(fp, "%d", &dist[row][col]);
         if (row != col && dist[row][col] == 0 )
         {
            dist[row][col] = 99999;
         }
      }
   }
}

/*
 * sequential folydwarshall algorithm
*/
void FloydWarshall(int dist[][v])
{
   int i, j, k;

   for (k = 0; k < v; ++ k)
   {
      for (i = 0; i < v; ++ i)
      {
         for (j = 0; j < v; ++ j)
         {
            if (dist[i][k] + dist[k][j] < dist[i][j])
            {
               dist[i][j] = dist[i][k] + dist[k][j];
            }
         }
      }
   }
   printSolution(dist);
}

/*
 * print out the solution
*/
void printSolution(int dist[][v])
{
    for (int i = 0; i < v; i++)
    {
        for (int j = 0; j < v; j++)
        {
           if (dist[i][j] == 99999)
           {
              printf ("%d ", 0);
           }
           else
           {
              printf ("%d ", dist[i][j]);
           }
        }
        printf("\n");
    }
}

int main(int argc, char * argv[])
{
   /* ----------------------------------------------------------------------------------------------------------------------------------*/
   
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
   
   /* _______________________________________check out the answer comment the line above _______________________________________________*/

   const char * filename = argv[1];
   graph(filename);

   /* check the correctness of the result comment out this line */
   auto const num_trials = 1u;
   auto const start_time = std::chrono::system_clock::now();

   FloydWarshall(dist);

   auto const end_time = std::chrono::system_clock::now();
   auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

   /* check the correctness of the result comment out this line */
   std::cout << "time: " << ( elapsed_time.count() / static_cast< float >( num_trials ) ) << " us" << std::endl; 
   return 0;
}

