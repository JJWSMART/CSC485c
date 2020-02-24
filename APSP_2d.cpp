#include <iostream>
#include <chrono>
#include <list>
#include <iterator>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <ctime>
#include <random>
#include <limits.h>
using namespace std;

// void ShortestDistance(int arr [4][4], int v);

void BFS (int arr[1000][1000], int src, int v, int dist[])
{
    list<int> queue;
    bool visited[v];
    
    for (int i = 0; i < v; i++) {
        visited[i] = false;
        dist[i] = INT_MAX;
    }
    
    visited[src] = true;
    dist[src] = 0;
    queue.push_back(src);
    
    while (!queue.empty()) {
        int u = queue.front();
        queue.pop_front();
        for (int i = 0; i < v; i += 1) {
            if (arr[u][i] == 1) {
                if (visited[i] == false) {
                    visited[i] = true;
                    dist[i] = dist[u] + 1;
                    queue.push_back(i);
                }
            }
        }
    }
}

void ShortestDistance(int arr [1000][1000], int v)
{
    int dist [v];
    for(int i = 0; i < v; i += 1) {
        BFS(arr, i, v, dist);
        for (int k = 0; k < v; k += 1)
        {
            cout << dist[k] << " ";
        }
        cout << "\n";
    }
}

void CreateGraph(int size, int edge_size, std::vector< std::vector<int> > &graph)
{
    // std::cout << "nnnnnnn" << std::endl;
    // std::uniform_int_distribution<> u(0,10);
    // std:cout << 0;
    // default_random_engine e(2147483646); 
    std::cout << 2;
    for (int i = 0; i < edge_size; i++) {
            int m = rand() % size;
            int n = rand() % size;
            int k = 1000;
        while (graph[m][n] == 1 || m == n) {
            std::cout << m << " " << n << std::endl;
            // srand(k);
            m = rand() % size;
            n = rand() % size;
            k++;
        }
        graph[m][n] = 1;
        graph[n][m] = 1;
    }
    for (int i = 0; i< size; i++) {
        if (std::accumulate(graph[i].begin(), graph[i].end(), 0) == 0) {
            int m = rand() % size;
            graph[m][i] = 1;
            graph[i][m] = 1;
        }
    }
}

int main (){

	srand(0);
	int v = 1000;
	std::vector< std::vector<int> > array(v);
    for (int i = 0; i < v; i++) {
        vector<int> x(v, 0);
        array[i] = x;
    }
    
    CreateGraph(v, 10000, array);

    int num [1000][1000];
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            num[i][j] = array[i][j];
        }
    }
    
    auto const start_time = std::chrono::system_clock::now();
    ShortestDistance(num, v);
    
    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Excution time: " << elapsed_time.count() << "us" << std::endl;
    
    return 0; 
}
