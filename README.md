# PyG: How to Evaluate the Solubility of a Molecule with Zinc thorugh Graph Neural Networks

![My Image](https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true)

<img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true" width=50% height=50%>

#### Getting started

Here we are going the describe the dataset, nodes and edges, maybe some of graph theory, and that we are going to use PyG



#### GNN 
https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications#:~:text=Graph%20Neural%20Networks%20(GNNs)%20are,and%20graph%2Dlevel%20prediction%20tasks

In computer science, a graph is a data structure consisting of two components: nodes (vertices) and edges. A graph G can be defined as G = (V, E), where V is the set of nodes, and E are the edges between them. (insert graph of a molecule or a simple graph)

Graph Neural Networks (GNNs) are a class of deep learning methods designed to perform inference on data described by graphs. GNNs are neural networks that can be directly applied to graphs, and provide an easy way to do node-level, edge-level, and graph-level prediction tasks.GNNs can do what Convolutional Neural Networks (CNNs) failed to do. It’s very difficult to perform CNN on graphs because of the arbitrary size of the graph, and the complex topology, which means there is no spatial locality. 

There’s also unfixed node ordering. If we first labeled the nodes A, B, C, D, E, and the second time we labeled them B, D, A, E, C, then the inputs of the matrix in the network will change. Graphs are invariant to node ordering, so we want to get the same result regardless of how we order the nodes.





#### Understand the data & Graph theory
A graph is used to model pairwise relations (edges) between objects (nodes). A single graph in PyG is described by an instance of torch_geometric.data.Data, which holds the following attributes by default:

-  data.x: Node feature matrix with shape [num_nodes, num_node_features]

-  data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

-  data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

-  data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

-  data.pos: Node position matrix with shape [num_nodes, num_dimensions]

For instace, "Data(x=[29, 1], edge_index=[2, 64], edge_attr=[64], y=[1])", which is the first observation of the dataset, stands for 29 nodes and 64 edges. Edge_index is a matrix od two rows and 64 colums, being each column the indeces of the nodes that are connected. For every edge, we need to define two index tuples to account for both directions of a edge.





Coordinate format: https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs

#### Formulate the problem

1. The graph itself and the labels for each node
2. The edge data in the Coordinate Format (COO)
3. Embeddings or numerical representations for the nodes


#### 



#### 
