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
using namespace std; 

class Graph 
{ 
    int V;    // No. of vertices
    list<int> *adj;    
    public:
        Graph(int V);  // Constructor
        void addEdge(int v, int w);
        void BFS(int s, int dist[]);
}; 
  
Graph::Graph(int V) 
{ 
    this->V = V; 
    adj = new list<int>[V]; 
} 
  
void Graph::addEdge(int v, int w){
    adj[v].push_back(w); // Add w to v’s list.
    adj[w].push_back(v); // Add v to w’s list.
} 
  
void Graph::BFS(int s, int dist[])
{
    bool *visited = new bool[V]; 
    for(int i = 0; i < V; i++) visited[i] = false;
    
    list<int> queue; 
    dist[s] = 0;
    visited[s] = true; 
    queue.push_back(s); 
  
    list<int>::iterator i; 
  
    while(!queue.empty()) 
    {
        s = queue.front();
        queue.pop_front(); 
  
        for (i = adj[s].begin(); i != adj[s].end(); ++i) 
        { 
            if (!visited[*i]) 
            { 
                visited[*i] = true;
                dist[*i] = dist[s] + 1;
                queue.push_back(*i); 
            } 
        } 
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


  
// Driver program to test methods of graph class 
int main() 
{ 
    srand(0);
    int v = 1000;
    Graph g(v);

    std::vector< std::vector<int> > array(v);
    for (int i = 0; i < v; i++) {
        vector<int> x(v, 0);
        array[i] = x;
    }

    std::cout << " ppppp" << std::endl;

    CreateGraph(v, 10000, array);

    for (int i = 0; i< v; ++i) {
        for (int j = i; j < v; j++) {
            std::cout << array[i][j] << " ";
            if (array[i][j] == 1) {
                g.addEdge(i,j);
            }
        }
        std::cout << std::endl;
    }

    int dist[v];
    auto const start_time = std::chrono::system_clock::now();
    
    for (int k = 0; k < v; k++){
        g.BFS(k, dist);
        for (int j = 0; j < v; j++){
            cout << dist[j] << " ";
        }
        cout << "\n";
    }

    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Excution time: " << elapsed_time.count() << "us" << std::endl;
  
    return 0; 
} 
