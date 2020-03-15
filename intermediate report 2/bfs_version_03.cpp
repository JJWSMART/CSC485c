#include <iostream>
#include <chrono>
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
        Graph(int V); // Constructor 
        void addEdge(int v, int w);
        void BFS(int s, int dist[]);
        vector<int> getNodes(int v);
};

Graph::Graph(int V)
{
    this->V = V;
    adj = new vector<int>[V];
}

vector<int> Graph::getNodes(int v)
{
    return(adj[v]);
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
    adj[w].push_back(v); // Add v to w’s list.
}

void Graph::BFS(int s, int dist[])
{
    /* dist array of nodes */
    dist[s] = 0;
    
    /* initialization of Queue */
    queue<int> t_queue;
    
    /***********************
     * num_threads         *
     * 1u                  *
     * 2u                  *
     * 4u                  *
     * 8u                  *
     ***********************/
    const size_t num_threads = 2u;
    
    /* initialize the visited array */
    bool *visited = new bool[V];
    for (int i = 0; i < V; i++) { visited[i] = false; }
    
    /* visited array */
    visited[s] = true;
    t_queue.push(s);
    
    /* Initialization of each thread of V vertex */
    bool v_arr [ num_threads ] [ V ];
    int v_dist [ num_threads ] [ V ];
    
    /* Boolean array of arr */
    bool arr[V] = { false };
    arr[s] = true;
    
    while(!t_queue.empty()) {
        s = t_queue.front();
        t_queue.pop();
        int adj_ns = adj[s].size();
        queue<int> queue[num_threads];

        for(auto i = 0llu; i < num_threads; i++) {
            for(int j = 0; j < V; j++) {
                v_arr[i][j] = visited[j];
                v_dist[i][j] = dist[j];
            }
        }
        
        #pragma omp parallel for num_threads( num_threads )
        for( int i = 0; i < adj_ns; ++ i ) {
            auto const th_id = omp_get_thread_num();
            int ad = adj[s][i];
            if (!v_arr[th_id][ad]) {
                v_arr[th_id][ad] = true;
                v_dist[th_id][ad] = dist[s] + 1;
                queue[th_id].push(ad);
            }
        }
        
        for (auto i = 0llu; i < num_threads; ++ i) {
            while (!queue[i].empty()){
                int tmp = queue[i].front();
                queue[i].pop();
                if (!arr[tmp]) {
                    arr[tmp] = true;
                    t_queue.push(tmp);
                }
            }
            for (auto k = 0; k < V; ++ k) {
                if (v_arr[i][k]) visited[k] = true;
                if (v_dist[i][k] != 99999){
                    dist[k] = v_dist[i][k];
                }
            }
        }
    }
}

int main() {
    
    // command line vertice
    int v = 1000;
    
    Graph g(v);
    
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
    /* start couting time */
    int dist [v][v] = { 99999 };
    auto const start_time = std::chrono::system_clock::now();
    for (int k = 0; k < v; k++) { g.BFS(k, dist[k]); }
    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Excution time: " << elapsed_time.count() << "us" << std::endl;
    return 0;
}
