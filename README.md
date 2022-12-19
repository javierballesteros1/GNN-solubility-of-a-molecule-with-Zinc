# PyG: How to Evaluate the Solubility of a Molecule with Zinc through Graph Neural Networks

<p align="center">
<img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true" width=50% height=50%>
</p>

PyG (PyTorch Geometric) is a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. It consists of various methods for deep learning on graphs and other irregular structures, also known as geometric deep learning.

## Getting started: the Data

Here we are going the describe the dataset, nodes and edges, maybe some of graph theory, and that we are going to use PyG

### What is a Graph?
Following the explanation in this [page](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications#:~:text=Graph%20Neural%20Networks%20(GNNs)%20are,and%20graph%2Dlevel%20prediction%20tasks), a graph is a data structure consisting of two components: nodes and edges. A graph G can be defined as G = (V, E), where V is the set of nodes, and E are the edges between them. 


### Zinc dataset
The Zinc dataset, whose documentation can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ZINC) contains 220,011 molecules, which are represented in graphs. For each molecule, the degree of solubility is provided. The task is to predict this solubility with Graph Regression.

First of all, let's explore the data. A molecule is a group of two or more atoms held together by attractive forces known as chemical bonds. In other words, the nodes are the atoms and the edges are the chemical bonds between the atoms. Let's see a molecule given in the data:

<p align="center">
<img src="https://github.com/javierballesteros1/GNN-solubility-of-a-molecule-with-Zinc/blob/main/images/molecule.png" >
</p>

This particular molecule has the following structure `Data(x=[18, 1], edge_index=[2, 36], edge_attr=[36], y=[1])`. Let's explain each component:

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


## Graph Neural Network
Graph Neural Networks (GNNs) are a class of deep learning methods designed to perform inference on data described by graphs. GNNs are neural networks that can be directly applied to graphs, and provide an easy way to do node-level, edge-level, and graph-level prediction tasks.GNNs can do what Convolutional Neural Networks (CNNs) failed to do. It’s very difficult to perform CNN on graphs because of the arbitrary size of the graph, and the complex topology, which means there is no spatial locality. 

There’s also unfixed node ordering. If we first labeled the nodes A, B, C, D, E, and the second time we labeled them B, D, A, E, C, then the inputs of the matrix in the network will change. Graphs are invariant to node ordering, so we want to get the same result regardless of how we order the nodes.





#### 
