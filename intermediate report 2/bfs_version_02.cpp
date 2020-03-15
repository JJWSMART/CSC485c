#include <iostream>
#include <chrono>
#include <list>
#include <iterator>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <ctime>
#include <random>
#include <string>
#include <cstring>
#include <omp.h>    // for multi-core parallelism
#include <queue>

using namespace std;

class Graph {
    int V;
    vector<int> *adj;
    public:
        /* Constructor */
        Graph(int V);
        void addEdge(int v, int w);
        void BFS(int s, int dist[]);
};

Graph::Graph(int V)
{
    this->V = V;
    adj = new vector<int>[V];
}


void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
    adj[w].push_back(v); // Add v to w’s list.
}

void Graph::BFS(int s, int dist[])
{
    
    dist[s] = 0;
    queue<int> t_queue;
    
    bool *visited = new bool[V];
    for(int i = 0; i < V; i++) { visited[i] = false; }
    
    visited[s] = true;
    t_queue.push(s);
    
    
    while(!t_queue.empty()) {
        
        s = t_queue.front();   /* Queue's front */
        t_queue.pop();
        int m = adj[s].size();
        queue<int> queue[m];
        
        #pragma omp parallel for num_threads( 2 )
        for( int i = 0; i < m; ++ i ) {
            auto const th_id = omp_get_thread_num();
            if (!visited[adj[s][i]]) {
                #pragma omp critical
                {
                    visited[adj[s][i]] = true;
                    dist[adj[s][i]] = dist[s] + 1;
                }
                queue[th_id].push(adj[s][i]);
            }
        }
        
        for (int i = 0; i < m; ++ i)
        {
            while (!queue[i].empty()){
                int tmp = queue[i].front();
                queue[i].pop();
                if (!arr[tmp]) {
                    arr[tmp] = true;
                    t_queue.push(tmp);
                }
            }
        }
        
    }
}

int main() {
    int v = 1000;
    Graph g(v);
    
    /* Initialize the two-dimensional array */
    std::vector< std::vector<int> > array(v);
    for (int i = 0; i < v; i++) {
        vector<int> x(v, 0);
        array[i] = x;
    }

    /* Transfer the two-dimensional array into adjancy list */
    int idx = 0;
    for (std::string str; std::getline(std::cin, str);) {
        std::string buf; /* Buffer String */
        std::stringstream ss(str); /* Insert The String Into A Stream */
        std::vector<std::string> tokens;
        while (ss >> buf){ tokens.push_back(buf); }
        for (int i = idx; i < v; ++ i) {
            if (tokens[i] == "1") { g.addEdge(idx, i); }
        }
        idx++;
    }
    
    int dist [v][v] = { 99999 };
    /* start couting time */
    auto const start_time = std::chrono::system_clock::now();
    for (int k = 0; k < v; k++) { g.BFS(k, dist [k]); }
    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Excution time: " << elapsed_time.count() << "us" << std::endl;
    
    return 0;
}


