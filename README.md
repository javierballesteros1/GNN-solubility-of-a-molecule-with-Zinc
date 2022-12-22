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
The Zinc dataset, whose documentation can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ZINC), contains 220,011 molecules, which are represented in graphs. For each molecule, the degree of solubility is provided. The task is to predict this solubility with Graph Regression.

First of all, let's explore the data. A molecule is a group of two or more atoms held together by attractive forces known as chemical bonds. In other words, the nodes are the atoms and the edges are the chemical bonds between the atoms. Let's see a molecule given in the data:

<p align="center">
<img src="https://github.com/javierballesteros1/GNN-solubility-of-a-molecule-with-Zinc/blob/main/images/molecule.png" >
</p>

This particular molecule has the following structure `Data(x=[18, 1], edge_index=[2, 36], edge_attr=[36], y=[1])`. Let's explain each component:

-  `x=[18, 1]`: `x` is the node feature matrix with shape [num_nodes, num_node_features]. For this particular molecule, we have 18 nodes. For each node, we see there are 1 node feature. That is, each node is described by a scalar. Find below this representation (actually it is the transpose, as it should be a column vector).

`tensor([[0], 
        [1],
        [0],
        [0],
        [4],
        [0],
        [0],
        [1],
        [2],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [0],
        [0]])`

-  `edge_index=[2, 36]`: it shows the graph connectivity in COO format with shape [2, num_edges], being each column the indeces of the nodes that are connected. In the example, there are 36/2 = 18 edges, as for every edge, we need to define two index tuples to account for both directions of a edge. Find below this matrix. 

`tensor([[ 0,  1,  1,  2,  2,  3,  3,  4,  4,  4,  5,  6,  6,  6,  7,  8,  8,  8,
          9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17],
          [ 1,  0,  2,  1,  3,  2,  4,  3,  5,  6,  4,  4,  7,  8,  6,  6,  9, 10,
          8,  8, 11, 10, 12, 17, 11, 13, 12, 14, 13, 15, 16, 14, 14, 17, 11, 16]])`
          
We can observe that indeed the first two columsn are $(0,1)^T$ and $(1,0)^T$, meaning that these two nodes are connected in both directions. Thus, as this happens for all the nodes, we can check that the graph in undirected.

-  `edge_attr=[36]`: feature matrix with shape [num_edges, num_edge_features]. We see below the one-row vector, so each edge is defined by one feature (a scalar).

`tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2])`

-  `y=[1]`: Target to train against. In our example, [1] is the dimension: the degree of solubility of each molecule (it is a scalar). 

`tensor([0.4907])`

This value is the target to be predicted.

### Batching of graphs
As explained in the documentation of PyG in [Google Colab 3. Graph Classification with Graph Neural Networks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html), similar to what is done in the image or language domain, by rescaling or padding each example into a set of equally-sized shapes, and examples are then grouped in an additional dimension, in graph analysis a good idea is to batch the graphs before inputting them into a Graph Neural Network to guarantee full GPU utilization. The length of this dimension is then equal to the number of examples grouped in a mini-batch and is typically referred to as the batch_size.

Therefore, PyG opts for another approach to achieve parallelization across a number of examples. Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple isolated subgraphs), and node and target features are simply concatenated in the node dimension:

<p align="center">
<img src="https://github.com/javierballesteros1/GNN-solubility-of-a-molecule-with-Zinc/blob/main/images/batch.png" >
</p>

### Why do we need Graph Neural Networks?

As we have seen, if we want to classify images we might choose Convolutional Neural Networks (CNN). Images are basically matrices of pixel values, which are **ordered**. Each image has a top and bottom, left and right. In the same spirit, in all flavours of recurrent neural networks (RNN) we pass in vectors that reflect measurements from a time series or a sequence.

Recall how a convolutional layer works: we slide the convolutional operator window across a two-dimensional image, and we compute some function over that sliding window. Then, we pass it through many layers. How would we apply this layer on a graph? It would be very difficult and ambiguous, as graphs do not have a natural order or reference point. Thus there does not exist a top and a bottom or a left and right. This is different compared to the type of data we can explain with linear regression, CNNs or RNNs. A picture consists of a regular lattice. RNNs learn sequences of well ordered vectors.

It’s very difficult to perform CNN on graphs because of the arbitrary size of the graph, and the complex topology, which means there is no spatial locality. Thus, we need to define Graph Neural Networks.

## Graph Neural Network Architecture

We are going to briefly explain the architecture of Graph Neural Networks (GNNs) based on this [page](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications#:~:text=Graph%20Neural%20Networks%20(GNNs)%20are,and%20graph%2Dlevel%20prediction%20tasks) and this [page](https://en.wikipedia.org/wiki/Graph_neural_network). GNNs are a class of deep learning methods designed to perform inference on data described by graphs. GNNs are neural networks that can be directly applied to graphs, and provide an easy way to do node-level, edge-level, and graph-level prediction tasks.  

The architecture of a generic GNN implements the following fundamental layers: Message passing layers and Readout layer.

#### Message passing layers (or Permuatation equivariant layers)

This layer maps a representation of a graph into an updated representation of the same graph. In the literature, permutation equivariant layers are implemented via pairwise message passing between graph nodes. Intuitively, in a message passing layer, nodes update their representations by aggregating the messages received from their immediate neighbours. As such, each message passing layer increases the receptive field of the GNN by one hop.

We are going to put a bit more formally this explanation (following [this](https://en.wikipedia.org/wiki/Graph_neural_network)). Let ${\displaystyle G=(V,E)}$ be a graph, where ${\displaystyle V}$ is the node set and ${\displaystyle E}$ is the edge set. Let ${\displaystyle N_{u}}$ be the neighbourhood of some node ${\displaystyle u\in V}$. Additionally, let ${\displaystyle \mathbf {x} _{u}}$ be the features of node ${\displaystyle u\in V}$, and ${\displaystyle \mathbf {e} _{uv}}$ be the features of edge ${\displaystyle (u,v)\in E}$. Then, the outputs of message passing layer are node representations ${\displaystyle \mathbf {h} _{u}}$ for each node ${\displaystyle u\in V}$ in the graph: 


$$
{\displaystyle \mathbf {h} _{u}=\phi \left(\mathbf {x_{u}} ,\bigoplus _{v\in N_{u}}\psi (\mathbf {x} _{u},\mathbf {x} _{v},\mathbf {e} _{uv})\right)}
$$

where ${\displaystyle \phi }$  and ${\displaystyle \psi }$  are differentiable functions, and ${\displaystyle \bigoplus }$ is a permutation invariant aggregation operator that can accept an arbitrary number of inputs (e.g., element-wise sum, mean, or max). In particular, ${\displaystyle \phi }$ is the usual activation function that we usually see (ReLU will be used in this case-study)

In the literature we sometimes find that these layers are known to perform node embedding. For instance, [here](https://towardsdatascience.com/node-embeddings-for-beginners-554ab1625d98#:~:text=Node%20embeddings%20are%20a%20way,in%20machine%20learning%20prediction%20tasks.), we find a really good explanation of the reason behind carrying out this kind of layer. As it is said in this blog: node embeddings are a way of representing nodes as vectors, capturing the topology of the network relying on a notion of similarity.

#### Readout layer (or Global pooling layer)

A readout layer provides fixed-size representation of the whole graph. In other words, a readout layer collects all node representations in a graph to form a graph representation (source [here](https://torchdrug.ai/docs/notes/layer.html#:~:text=A%20readout%20layer%20collects%20all,every%20node%20in%20the%20graph.)). The global pooling layer must be permutation invariant, such that permutations in the ordering of graph nodes and edges do not alter the final output. Examples include element-wise sum, mean or maximum.

### GNN Architecture for the Zinc data

Find below the architecture that we found to have associated the lowest loss
```
class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels_1, hidden_channels_2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_1)
        self.conv3 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv4 = GCNConv(hidden_channels_2, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings through Message passing layers 
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        g_pool = global_mean_pool(x, batch)
        return g_pool
```
The parameters `hidden_channels_1` and `hidden_channels_2` were also manually tuned, and we got to the following GNN

```
GCN(  
  (conv1): GCNConv(1, 4)  
  (conv2): GCNConv(4, 4)  
  (conv3): GCNConv(4, 4)  
  (conv4): GCNConv(4, 1)  
)
```
However, we would like to mention that we also tried other GNN with parameters
*   `hidden_channels_1 = 2` `hidden_channels_2 = 4` 
*   `hidden_channels_1 = 4` `hidden_channels_2 = 2` 
*   `hidden_channels_1 = 2` `hidden_channels_2 = 8` 

but their performance was not as good as the model selected. 

### GNN Training
In the plot below we can see how the GNN is performing as we are training it with backward propagation. That is, in each epoch we are updating the weights so the GNN's loss decreases until it reaches a point where it stabilizes. 
<p align="center">
<img src="https://github.com/javierballesteros1/GNN-solubility-of-a-molecule-with-Zinc/blob/main/images/losstrainval.png" >
</p>

We are plotting how the GNN performs on both the train and the validation set.

### GNN Test
For the test set, we got a loss of 

### References

[1]: [Graph Neural Network and Some of GNN Applications: Everything You Need to Know](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications#:~:text=Graph%20Neural%20Networks%20(GNNs)%20are,and%20graph%2Dlevel%20prediction%20tasks). 

[2]: [TORCH_GEOMETRIC.DATASETS](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ZINC). 

[3]: [PYG: COLAB NOTEBOOKS AND VIDEO TUTORIALS](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)

[4]: [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network)

[5]: [Node embeddings for Beginners](https://towardsdatascience.com/node-embeddings-for-beginners-554ab1625d98#:~:text=Node%20embeddings%20are%20a%20way,in%20machine%20learning%20prediction%20tasks.)

[6]: [Graph Neural Network Layers](https://torchdrug.ai/docs/notes/layer.html#:~:text=A%20readout%20layer%20collects%20all,every%20node%20in%20the%20graph.)

[7]: [A Beginner’s Guide to Graph Neural Networks Using PyTorch Geometric — Part 1](https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742)



