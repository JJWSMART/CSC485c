#include <iostream>
#include <list>
#include <iterator>
#include <vector>
#include <random>

using namespace std;

/*
 * Purpose:
 *      Create Graph
 *
 * Pre-conditions:
 *      None.
 *
 * Returns:
 *      None
 *
*/

void CreateGraph(int size, int edge_size, std::vector< std::vector<int> > &graph) {
        /* create a 2d graph */
    for (int i = 0; i < edge_size; i++) {
                /* hash a position 1 */
            int m = rand() % size;
            /* hash a position 2 */
            int n = rand() % size;
            int k = 100;
            /* rehashing number if exits*/
            while (graph[m][n] == 1 || m == n) {
                m = rand() % size;
                n = rand() % size;
                k++;
        }
        graph[m][n] = 1;
        graph[n][m] = 1;
    }
    /* ensure the graph assgin a number */
    for (int i = 0; i< size; i++) {
        if (std::accumulate(graph[i].begin(), graph[i].end(), 0) == 0) {
            int m = rand() % size;
            graph[m][i] = 1;
            graph[i][m] = 1;
        }
    }
}

/*
 * Purpose:
 *      Print Graph
 *
 * Pre-conditions:
 *      None.
 *
 * Returns:
 *      None
 *
*/
void PrintGraph (int size, std::vector< std::vector<int>> &graph)
{
        for ( int i = 0llu; i < size; ++ i ){
                for (int j = 0llu; j < size; ++ j){
                        cout << graph[i][j] << " ";
                }
                cout << "\n";
        }
}


/*
 * Purpose:
 *      > redirect the graph to as a input file
 *
 * Pre-conditions:
 *      None.
 *
 * Returns:
 *      None
 *
*/
int main()
{
    srand(0);
    int v = 10;
    std::vector<std::vector<int>> array(v);

    for (int i = 0; i < v; ++ i)
    {
        vector<int> x(v, 0);
        array[i] = x;
    }

    CreateGraph(v, 40, array);

    PrintGraph (10, array);

    return 0;
}

