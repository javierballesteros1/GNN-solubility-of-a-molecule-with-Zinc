# Geometric Deep Learning

## Graph Neural Network: how to evaluate the solubility of a molecule with Zinc

#### Getting started

Here we are going the describe the dataset, nodes and edges, maybe some of graph theory, and that we are going to use PyG




#### Understand the data
A graph is used to model pairwise relations (edges) between objects (nodes). A single graph in PyG is described by an instance of torch_geometric.data.Data, which holds the following attributes by default:

-  data.x: Node feature matrix with shape [num_nodes, num_node_features]

-  data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

-  data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

-  data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

-  data.pos: Node position matrix with shape [num_nodes, num_dimensions]





Coordinate format: https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs

#### Formulate the problem

1. The graph itself and the labels for each node
2. The edge data in the Coordinate Format (COO)
3. Embeddings or numerical representations for the nodes


#### 



#### 
