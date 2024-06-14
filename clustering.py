import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
import argparse
import json

RANDOM_STATE = 42

def load_data(path):
    """
    Load data from a JSON file and extract sequences.

    Parameters:
    path (str): The path to the JSON file.

    Returns:
    list: A list of lists where each inner list is a sequence of data.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    sequences = list(data.values())

    return sequences # this is a list of lists


def apply_truncation(sequences, truncation_strategy):
    truncated_sequences = []
    truncation_length = min(len(s) for s in sequences)

    if truncation_strategy == 'right':
        for seq in sequences:
            truncated_sequences.append(seq[:truncation_length])
    
    else:
        pass

    return np.array(truncated_sequences), truncation_length


def apply_kmeans(X, k):
    """
    Apply K-means clustering to the dataset.

    Parameters:
    X (numpy.ndarray): A 2D array where each row is a data sample.
    k (int): The number of clusters.

    Returns:
    tuple:
        labels (numpy.ndarray): Cluster labels for each data sample.
        centroids (numpy.ndarray): Coordinates of cluster centroids.
        distances (numpy.ndarray): Distances of each sample to its nearest centroid.
    """
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(X)

    # assign a cluster to each data sample + select centroid
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_

    # compute distances to centroid (to be used later to rank labels)
    _, distances = pairwise_distances_argmin_min(X, centroids)

    return labels, centroids, distances


def plot_clustering(X, labels, centroids, distances):
    """
    Plot the clustering results and display cluster centroids and distances.

    Parameters:
    X (numpy.ndarray): A 2D array where each row is a data sample.
    labels (numpy.ndarray): Cluster labels for each data sample.
    centroids (numpy.ndarray): Coordinates of cluster centroids.
    distances (numpy.ndarray): Distances of each sample to its nearest centroid.
    """
    # TSNE to plot data in a 2D space
    combined = np.vstack([X, centroids])
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    combined_2d = tsne.fit_transform(combined)
    X_2d = combined_2d[:X.shape[0]]
    centroids_2d = combined_2d[X.shape[0]:]

    # plot clustering in 2D space
    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, marker='o', edgecolor='k')
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100)
    plt.title(f"Clustered Data (k={k}, tuncation strategy={trunc_strategy}, truncation_length={trunc_length})")
    # plt.show()

    # save the plot
    plt.savefig(f'./plots/clusters_{k}_{trunc_strategy}_{trunc_length}.png', dpi=500, bbox_inches='tight')

    # show distances to centroidss
    print("Cluster centroids:\n", centroids)
    print("\nDistances to the nearest centroid for each sample:\n", distances)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The path to TPSD dataset (json format is expected)', default='./data/tpsd_seq_dict.json', type=str)
    parser.add_argument('-t', '--truncation', help="Select the trucation strategy to perform on TPSD sequences. Can be either 'right', meaning naive right truncation, or 'smart', meaning taking care of keeping reapeting patterns.", default='right', type=str)
    parser.add_argument('-k', '--kmeans', help='The number of desired clusters. Default is 3.', default=5, type=int)

    args = parser.parse_args()
    path = args.path
    trunc_strategy = args.truncation
    k = args.kmeans

    sequences = load_data(path)
    X, trunc_length = apply_truncation(sequences, trunc_strategy)
    labels, centroids, distances = apply_kmeans(X, k)
    plot_clustering(X, labels, centroids, distances)