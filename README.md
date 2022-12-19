# PyG: How to Evaluate the Solubility of a Molecule with Zinc through Graph Neural Networks

<p align="center">
<img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true" width=40% height=40%>
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

-  `x=[18, 1]`: `x` is the node feature matrix with shape [num_nodes, num_node_features]. For this particular molecule, we have 18 nodes.

-  `edge_index=[2, 36]`: it shows the graph connectivity in COO format with shape [2, num_edges], being each column the indeces of the nodes that are connected. In the example, there are 36/2 = 18 edges, as for every edge, we need to define two index tuples to account for both directions of a edge.

-  `edge_attr=[36]`: feature matrix with shape [num_edges, num_edge_features]

-  `y=[1]`: Target to train against. In our example, [1] is the dimension: the degree of solubility of each molecule (it is a scalar). 

### Batching of graphs
Here I have to add a brief explanation and the plot from the google colab number 3

### Why do we need Graph Neural Networks?

As we have seen, if we want to classify images we might choose Convolutional Neural Networks (CNN). Images are basically matrices of pixel values, which are **ordered**. Each image has a top and bottom, left and right. In the same spirit, in all flavours of recurrent neural networks (RNN) we pass in vectors that reflect measurements from a time series or a sequence.

Recall how a convolutional layer works: we slide the convolutional operator window across a two-dimensional image, and we compute some function over that sliding window. Then, we pass it through many layers. How would we apply this layer on a graph? It would be very difficult and ambiguous, as graphs do not have a natural order or reference point. Thus there does not exist a top and a bottom or a left and right. This is different compared to the type of data we can explain with linear regression, CNNs or RNNs. A picture consists of a regular lattice. RNNs learn sequences of well ordered vectors.

It’s very difficult to perform CNN on graphs because of the arbitrary size of the graph, and the complex topology, which means there is no spatial locality. 

## Graph Neural Network

Graph Neural Networks (GNNs) are a class of deep learning methods designed to perform inference on data described by graphs. GNNs are neural networks that can be directly applied to graphs, and provide an easy way to do node-level, edge-level, and graph-level prediction tasks.GNNs can do what Convolutional Neural Networks (CNNs) failed to do. It’s very difficult to perform CNN on graphs because of the arbitrary size of the graph, and the complex topology, which means there is no spatial locality. 

There’s also unfixed node ordering. If we first labeled the nodes A, B, C, D, E, and the second time we labeled them B, D, A, E, C, then the inputs of the matrix in the network will change. Graphs are invariant to node ordering, so we want to get the same result regardless of how we order the nodes.

#### Message passing layers

#### Readout layer






