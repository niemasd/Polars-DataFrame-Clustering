#! /usr/bin/env python3
'''
Clustering trees from Polars DataFrame
'''

# imports
from hdbscan import HDBSCAN # use `from cuml.cluster import HDBSCAN` for GPU-accelerated
from matplotlib import ticker, use
from polars import DataFrame
from polars.selectors import numeric
from scipy.cluster.hierarchy import linkage, to_tree
import matplotlib.pyplot as plt

# set things up
use('Agg')

# compute hierarchical clustering tree
def compute_hierarchical_tree(df:DataFrame, label_col:str|None=None, method:str='ward', metric:str='euclidean'):
    labels = list(range(df.height)) if label_col is None else df.get_column(label_col).to_list()
    linkage_matrix = linkage(df.select(numeric()).to_numpy(), method=method, metric=metric)
    root, nodes = to_tree(linkage_matrix, rd=True)
    for i in range(len(labels)):
        nodes[i].label = labels[i]
    return linkage_matrix, root, nodes

# compute HDBSCAN clustering tree
def compute_hdbscan_tree(df:DataFrame, label_col:str|None=None, metric:str='euclidean', leaf_size=1):
    labels = list(range(df.height)) if label_col is None else df.get_column(label_col).to_list()
    model = HDBSCAN(min_cluster_size=2, min_samples=1, metric=metric, leaf_size=leaf_size)
    model.fit(df.select(numeric()).to_numpy())
    linkage_matrix = model.single_linkage_tree_.to_numpy()
    root, nodes = to_tree(linkage_matrix, rd=True)
    for i in range(len(labels)):
        nodes[i].label = labels[i]
    return linkage_matrix, root, nodes

# compute number of clusters vs. height (i.e., distance from leaves)
def compute_clusters_vs_height(nodes, include_singletons=False):
    num_clusters = dict()
    curr_num_clusters = 0
    for node in sorted(nodes, key=lambda x:x.dist):
        if node.is_leaf():
            if include_singletons:
                curr_num_clusters += 1
        else: # internal node
            if include_singletons:
                curr_num_clusters -= 1
            else:
                num_leaf_children = int(node.left.is_leaf()) + int(node.right.is_leaf())
                if num_leaf_children == 0:
                    curr_num_clusters -= 1
                elif num_leaf_children == 2:
                    curr_num_clusters += 1
        num_clusters[node.dist] = curr_num_clusters
    return num_clusters

# run small example
if __name__ == "__main__":
    # define small example dataset
    df = DataFrame({
        'label': ['A', 'B', 'C', 'X', 'Y', 'Z'],
        'value': [ 0 ,  1 ,  2 ,  96,  98, 100],
    })
    print(df)

    # hierarchical clustering
    print("=== Hierarchical Clustering ===")
    linkage_matrix, root, nodes = compute_hierarchical_tree(df, label_col='label')
    print("Number of Clusters by Height (no singletons):")
    num_clusters_height = sorted(compute_clusters_vs_height(nodes, include_singletons=False).items())
    x = list(); y = list()
    for height, num_clusters in num_clusters_height:
        print(f"- {height}\t{num_clusters}")
        x.append(height); y.append(num_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Height")
    ax.set_ylabel("Number of Clusters")
    fig.savefig("hierarchical_clusters_by_height.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Number of Clusters by Height (with Singletons):")
    num_clusters_height_singletons = sorted(compute_clusters_vs_height(nodes, include_singletons=True).items())
    x = list(); y = list()
    for height, num_clusters in num_clusters_height_singletons:
        print(f"- {height}\t{num_clusters}")
        x.append(height); y.append(num_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Height")
    ax.set_ylabel("Number of Clusters")
    fig.savefig("hierarchical_clusters_by_height_singletons.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # HDBSCAN clustering
    print("=== HDBSCAN Clustering ===")
    linkage_matrix, root, nodes = compute_hdbscan_tree(df, label_col='label')
    print("Number of Clusters by Height (no singletons):")
    num_clusters_height = sorted(compute_clusters_vs_height(nodes, include_singletons=False).items())
    x = list(); y = list()
    for height, num_clusters in num_clusters_height:
        print(f"- {height}\t{num_clusters}")
        x.append(height); y.append(num_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Height")
    ax.set_ylabel("Number of Clusters")
    fig.savefig("hdbscan_clusters_by_height.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Number of Clusters by Height (with Singletons):")
    num_clusters_height_singletons = sorted(compute_clusters_vs_height(nodes, include_singletons=True).items())
    x = list(); y = list()
    for height, num_clusters in num_clusters_height_singletons:
        print(f"- {height}\t{num_clusters}")
        x.append(height); y.append(num_clusters)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Height")
    ax.set_ylabel("Number of Clusters")
    fig.savefig("hdbscan_clusters_by_height_singletons.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
