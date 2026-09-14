# HDBSCAN Examples
This document describes some examples of using HDBSCAN with [TreeSwift](https://github.com/TreeSwift).

## Import HDBSCAN
There are a handful of Python implementations of HDBSCAN.

### CPU (scikit-learn)
The Python scikit-learn library provides a [CPU implementation of HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

```python
from sklearn.cluster import HDBSCAN
```

### Faster CPU
The Python `fast_hdbscan` library provides a [fast multi-core CPU implementation of HDBSCAN](https://github.com/TutteInstitute/fast_hdbscan).

```python
from fast_hdbscan import HDBSCAN
```

### GPU
NVIDIA's cuML library provides a [GPU-accelerated implementation of HDBSCAN](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.hdbscan.HDBSCAN/).

```python
from cuml.cluster import HDBSCAN
```

## Compute HDBSCAN Clustering Tree
The function below computes a complete HDBSCAN clustering tree: `df` is a `DataFrame`, and `labels` is a `list` of `str` in which `labels[i]` is the label of the `i`-th row of `df`.

```python
def compute_hdbscan_tree(df, labels=None):
    model = HDBSCAN(min_cluster_size=2, min_samples=1)
    model.fit(df)
    linkage_matrix = model.single_linkage_tree_.to_numpy()
    root, nodes = to_tree(linkage_matrix, rd=True)
    tree = read_tree_scipy(root)
    for node in tree.traverse_preorder():
        node.scipy_id = node.label
        if node.is_leaf() and labels is not None:
            node.label = labels[node.label]
    return tree
```

## Clustering by Cutting at Height
A simple approach to compute clusters from an HDBSCAN clustering tree is to cut the tree at some height *h* above the leaves (the tree is ultrametric) and output (the leaves of) all resulting subtrees as clusters.

```python
def cluster_cut_height(tree, h):
    heights = {node:h for node, h in tree.heights()}
    clusters = list()
    to_visit = Queue()
    to_visit.put(tree.root)
    while not to_visit.empty():
        node = to_visit.get()
        if node.is_leaf():
            clusters.append([node])
        elif (heights[node] < h) and ((node.parent is None) or (heights[node.parent] >= h)):
            clusters.append([curr for curr in node.traverse_preorder()])
        else:
            for child in node.children:
                to_visit.put(child)
    return clusters
```

This approach is parametric: the user needs to know the desired height *h* at which to cut. One can select *h* by determining the number of clusters that would exist at any arbitrary value of *h* in a single traversal of the nodes of the tree, sorted in ascending distance from the root:

```python
def compute_clusters_vs_height(tree, include_singletons=False):
    tree_height = tree.height()
    num_clusters = dict()
    curr_num_clusters = 0
    for root_dist, node in tree.traverse_rootdistorder(ascending=False):
        if node.is_leaf():
            if include_singletons:
                curr_num_clusters += 1
        else: # internal node
            if include_singletons:
                curr_num_clusters -= 1
            else:
                num_leaf_children = sum(child.is_leaf() for child in node.children)
                if num_leaf_children == 0:
                    curr_num_clusters -= 1
                elif num_leaf_children == 2:
                    curr_num_clusters += 1
        num_clusters[tree_height-root_dist] = curr_num_clusters
    return num_clusters
```

## Excess of Mass (EOM) Clustering
[Excess of Mass (EOM)](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html#extract-the-clusters) selects clusters by finding branches that remain sufficiently persistent across desnity levels:

1. Calculate $\lambda_{b}\left(u\right)=\frac{1}{h\left(u\right)}$ for all nodes $u$, where $h\left(u\right)$ is the height of node $u$
2. Calculate $L\left(u\right)$ for all nodes $u$, where $L\left(u\right)$ is the number of leaves in the subtree rooted at $u$
3. Calculate $S\left(u\right)$ for all nodes $u$, where $S\left(u\right)=\sum_{c}{L\left(u\right)\left(\lambda_{b}\left(c\right)-\lambda_{b}\left(u\right)\right)}$ over all children $c$ of $u$

This approach is the [default clustering method in HDBSCAN](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering): this reimplementation is so it's easy to deviate from it / easily explore alternatives.

```python
def cluster_eom(tree, min_cluster_size=2, max_cluster_size=float('inf')):
    # calculate heights and lambda_birth values
    root_dist = dict()
    lambda_birth = dict()
    height = {node:h for node, h in tree.heights()}
    for node in tree.traverse_preorder():
        if node.is_root():
            root_dist[node] = 0
            lambda_birth[node] = 0.0
        else:
            root_dist[node] = root_dist[node.parent] + node.edge_length
            if node.is_leaf() or height[node] <= 0:
                lambda_birth[node] = float('inf')
            else:
                lambda_birth[node] = 1.0 / height[node]

    # calculate subtree sizes and stability values, and perform EOM
    subtree_size = dict()
    stability = dict()
    best = dict()
    select = dict()
    for node in tree.traverse_postorder():
        if node.is_leaf():
            subtree_size[node] = 1
            stability[node] = 0.0
            best[node] = 0.0
            select[node] = False
        else:
            subtree_size[node] = sum(subtree_size[child] for child in node.children)
            stability[node] = sum(subtree_size[child] * (lambda_birth[child] - lambda_birth[node]) for child in node.children)
            children_score = sum(best[child] for child in node.children)
            if (stability[node] >= children_score) and (min_cluster_size <= subtree_size[node] <= max_cluster_size):
                best[node] = stability[node]
                select[node] = True
            else:
                best[node] = children_score
                select[node] = False

    # extract clusters
    clusters = list()
    to_visit = Queue()
    to_visit.put(tree.root)
    while not to_visit.empty():
        node = to_visit.get()
        if select[node]:
            clusters.append([curr for curr in node.traverse_leaves()])
        else:
            for child in node.children:
                to_visit.put(child)
    return clusters
```

## Save Annotated Newick Tree
Annotate each leaf with its cluster number (`0` through `len(clusters)-1`) and other useful metrics, and save as a Newick file, which can be visualized using [Taxonium](https://taxonium.org).

```python
# initialize every node's node_params dict
for node in tree.traverse_preorder():
    node.node_params = dict()

# label leaves with their cluster numbers
node_to_clusters = dict()
for cluster_num, cluster in enumerate(clusters):
    for node in cluster:
        node.node_params['cluster'] = cluster_num
        node_to_clusters[node] = {cluster_num}
for node in tree.traverse_postorder(leaves=False):
    node_to_clusters[node] = {cluster_num for child in node.children for cluster_num in node_to_clusters[child]}
    node.node_params['cluster'] = '_'.join(str(cluster_num) for cluster_num in sorted(node_to_clusters[node]))

# calculate summary statistics of the clustering features
for feature in clustering_features:
    key_prefix = f'summary_{feature}'
    for node in tree.traverse_postorder():
        if node.is_leaf():
            node.node_params[feature] = df.item(node.scipy_id, feature)
            for s in ['min', 'max', 'sum']:
                node.node_params[f'{key_prefix}_{s}'] = node.node_params[feature] # delete later
            node.node_params['num_points'] = 1
        else:
            for s, f in [('min',min), ('max',max), ('sum',sum)]:
                k = f'{key_prefix}_{s}'
                node.node_params[k] = f(child.node_params[k] for child in node.children)
            node.node_params['num_points'] = sum(child.node_params['num_points'] for child in node.children)
            node.node_params[f'{key_prefix}_mean'] = node.node_params[f'{key_prefix}_sum'] / node.node_params['num_points']

# clean up nodes and write Newick file
for node in tree.traverse_preorder():
    if node.is_leaf():
        to_delete = [k for k in node.node_params if k.startswith('summary_')] + ['num_points']
    else:
        to_delete = [k for k in node.node_params if k.endswith('_sum')]
    for k in to_delete:
        del node.node_params[k]
tree.write_tree_newick('hdbscan_tree_annotated.nwk.gz')
```
