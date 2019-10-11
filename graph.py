import networkx as nx
import pickle

class Graph():
    """Class used for building graph using networkx.
        Its usage assumption is every alternate node will be of same type/class
    """

    def __init__(self):
        self._graph = nx.Graph()

    def add_node(self, node: str, c_type: str):
        """Add a node to the graph of an attribute type (binary: for example (title, skill))
        
        Arguments:
            node {str} -- name of the node
            c_type {str} -- type of the node (atrribute)
        """
        self._graph.add_node(node, type=c_type)

    def add_nodes(self, nodes: list, c_types: str):
        """Add nodes to the graph of an(or different) attribute type (binary: for example (title, skill))
        
        Arguments:
            nodes {list} -- list of nodes in string format
            c_types {str} -- Either a string to denote every node of same type OR list of string for each node
        """
        # decide to assign same type or different based of datatype
        if not isinstance(c_types, list):
            self._graph.add_nodes_from(nodes, type=c_types)
        else:
            self._graph.add_nodes_from(nodes)
            for i,key in enumerate(nodes):
                self._graph.nodes[key]['type'] = c_types[i]

    def add_edge(self, node1: str, node2: str, weight=1):
        """Add edges between nodes
        
        Arguments:
            node1 {str} -- name of the one node
            node2 {str} -- name of another node
        
        Keyword Arguments:
            weight {int} -- weight of the edge (default: {1})
        """
        if not self._graph.has_edge(node1, node2):
            self._graph.add_edge(node1, node2, weight=weight)
        else:
            # TODO: update the heuristic. Currently assign weights based on pre-defined heuristics
            old_weight = self._graph[node1][node2]['weight']
            if old_weight < weight:
                self._graph[node1][node2]['weight'] = weight

    def nearest_neighbor(self, node: str, number=1) -> list:
        """Find the neighboring nodes of a given node. Sort the neighbors accoring to the edge weights and return top k (number <= Max neighbors)
        
        Arguments:
            node {str} -- name of the node
        
        Keyword Arguments:
            number {int} -- number of nodes to return: 0 < number < Max neighbors (default: {1})
        
        Returns:
            list -- list of neighbors in string format
        """
        neighbors = self._graph[node]
        if neighbors:
            sorted_neighbor = [x for x,_ in sorted(neighbors.items(), key=lambda x: x[1]['weight'], reverse=True)]

            # return top k
            if number < len(sorted_neighbor):
                return sorted_neighbor[0:number]
            else:
                return sorted_neighbor           

    def next_neighbor(self, node: str, number=1) -> list:
        """Find the next neighbors (nodes after one hop) excluding itself( no self-loop)
        Sort the next neighbors accoring to the cummulative edge weights and return top k (number <= Max next-neighbors)
        
        Arguments:
            node {str} -- name of the node
        
        Keyword Arguments:
            number {int} -- number of nodes to return: 0 < number < Max next-neighbors (default: {1})
        
        Returns:
            list -- list of neighbors in string format
        """
        neighbors = self.nearest_neighbor(node,number=len(self._graph.nodes))
        if neighbors:
            next_neigh = []
            weights_2 = []
            for v in neighbors:
                weight_1 = self._graph[node][v]['weight']
                nx_n = self.nearest_neighbor(v, number=len(self._graph.nodes))
                # remove query node
                if node in nx_n:
                    nx_n.remove(node)
                if nx_n:
                    next_neigh.extend(nx_n)
                    # add first hop edge weight
                    weights_2.extend([self._graph[i][v]['weight']+weight_1 for i in nx_n])

            if len(next_neigh) > 0:
                sorted_next_neigh = [x for _,x in sorted(zip(weights_2, next_neigh), key=lambda pair: pair[0], reverse=True)]
                # return top k 
                if number < len(sorted_next_neigh):
                    return sorted_next_neigh[0:number]
                else:
                    return sorted_next_neigh
    
    @property
    def graph(self):
        return self._graph
    
    @property
    def nodes(self):
        return self._graph.nodes


if __name__ == "__main__":
    
    G = Graph()
    G.add_node(1,'one')

    node = 1
    nodes = [2,3,4,5]
    att = ['two', 'two', 'one','one']

    G.add_node(node,'one')
    G.add_nodes(nodes, att)

    G.add_edge(1,2,0.8)
    G.add_edge(1,2,1)

    G.add_edge(1,3,0.9)
    G.add_edge(2,4,0.5)
    G.add_edge(3,5,0.7)
    G.add_edge(3,6,0.3)

    G.next_neighbor(1,2)

    print('Here')


