#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
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

// Basic BFS algorithm
void Graph::BFS(int s, int dist[])
{
    // initialization of dist & queue
    dist[s] = 0;
    queue<int> t_queue;
    // bool visited array
    bool *visited = new bool[V];
    for (int i = 0; i < V; i++) { visited[i] = false; }
    // queue.not empty
    visited[s] = true;
    t_queue.push(s);
    while(!t_queue.empty())
    {
        s = t_queue.front();
        t_queue.pop();
        int num = adj[s].size();
        for( int i = 0; i < num; ++ i ) { //adjancent list
            int ad = adj[s][i];
            if (!visited[ad]) {
                visited[ad] = true;
                dist[ad]= dist[s] + 1;
                t_queue.push(ad);
            }
        }
    }
}

int main() {
    /* number of vertices */
    int v = 1000;
    Graph g(v);
    /* Distance 2-d array */
    int dist [v][v] = { 99999 };
    
    /* Transfer the 2-dimensional array into adjancy list */
    int idx = 0;
    for (std::string str; std::getline(std::cin, str);) {
        std::string buf; // Buffer String
        std::stringstream ss(str); // Insert the String into A Stream
        std::vector<std::string> tokens;
        while (ss >> buf){ tokens.push_back(buf); }
        for (int i = idx; i < v; ++ i)
            if (tokens[i] == "1") { g.addEdge(idx, i); }
        idx++;
    }
    /* ******************************************************* *
     * @input file: less_1000_input.txt / more_1000_input.txt  *
     * Exprienment Set Up to 1u                                *
     * Exprienment Set Up to 2u                                *
     * Exprienment Set Up to 4u                                *
     * Exprienment Set Up to 8u                                *
     * ******************************************************* */
    const size_t num_threads = 4u;
    auto const start_time = std::chrono::system_clock::now();
    #pragma omp parallel for num_threads( num_threads )
    for (int k = 0; k < v; k++)
    {
        g.BFS(k, dist[k]);
    }
    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Excution time: " << elapsed_time.count() << "us" << std::endl;
    
    return 0;
}

