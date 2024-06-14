import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
import argparse
import json
import mplcursors

# import from other scripts in the repo
from truncation import apply_truncation_strategy, find_repeated_patterns

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
    labels = list(data.keys())
    print(len(sequences))

    return sequences, labels


def apply_truncation(sequences, truncation_strategy):
    truncated_sequences = []

    if truncation_strategy == 'right':
        truncation_length = min(len(s) for s in sequences)
    
    elif trunc_strategy == 'smart':
        length_if_applied = []
        for seq in sequences:
            repeated_patterns = find_repeated_patterns(seq)
            truncated_seq_with_strategy = apply_truncation_strategy(seq, repeated_patterns)
            l = len(truncated_seq_with_strategy)
            length_if_applied.append(l)
        truncation_length = min(length_if_applied)

    for seq in sequences:
        res_seq = seq[:truncation_length] if len(seq) > truncation_length else seq
        truncated_sequences.append(res_seq)

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


def interactive_plot_clustering(X, song_titles, labels, centroids, distances, k, trunc_strategy, trunc_length):
    """
    Plot the clustering results and display cluster centroids and distances.

    Parameters:
    X (numpy.ndarray): A 2D array where each row is a data sample.
    song_titles (list of str): List of song titles corresponding to each data sample.
    labels (numpy.ndarray): Cluster labels for each data sample.
    centroids (numpy.ndarray): Coordinates of cluster centroids.
    distances (numpy.ndarray): Distances of each sample to its nearest centroid.
    k (int): Number of clusters.
    trunc_strategy (str): Truncation strategy used.
    trunc_length (int): Truncation length.
    """
    # Combine data samples and centroids for TSNE transformation
    combined = np.vstack([X, centroids])
    tsne = TSNE(n_components=2, random_state=42)  # Use a fixed random state for reproducibility
    combined_2d = tsne.fit_transform(combined)
    X_2d = combined_2d[:X.shape[0]]
    centroids_2d = combined_2d[X.shape[0]:]

    # Plot clustering in 2D space
    plt.figure()
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')

    # Add centroids to the plot
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100, label='Centroids')

    # Set plot title and labels
    plt.title(f"Clustered Data (k={k}, truncation strategy={trunc_strategy}, truncation length={trunc_length})")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.legend()

    # Function to format tooltip text
    def format_tooltip(sel):
        ind = sel.target.index
        if ind < len(X):
            return f"Song: {song_titles[ind]}\nCluster: {labels[ind]}\nDistance to centroid: {distances[ind]:.2f}"
        else:
            return f"Centroid {ind - len(X) + 1}"

    # Enable interactive tooltips using mplcursors
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(format_tooltip(sel)))

    # Save the plot
    plt.savefig(f'./plots/clusters_{k}_{trunc_strategy}_{trunc_length}.png', dpi=500, bbox_inches='tight')

    # Show distances to centroids
    print("Cluster centroids:\n", centroids)
    print("\nDistances to the nearest centroid for each sample:\n", distances)

    # Display the plot
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The path to TPSD dataset (json format is expected)', default='./data/tpsd_seq_dict.json', type=str)
    parser.add_argument('-t', '--truncation', help="Select the truncation strategy to perform on TPSD sequences. Can be either 'right', meaning naive right truncation, or 'smart', meaning taking care of keeping reapeting patterns.", default='right', type=str) # there is an issue with the smart strategy, sometimes the algorithm cannot terminate
    parser.add_argument('-k', '--kmeans', help='The number of desired clusters. Default is 3.', default=3, type=int)
    parser.add_argument('-i', '--interactive', help='Whether the plot should be interactive, meaning we have the reference of each data point (song) by putting the mouse on it.', action='store_true')

    args = parser.parse_args()
    path = args.path
    trunc_strategy = args.truncation
    k = args.kmeans
    interactive = args.interactive

    sequences, names = load_data(path)
    X, trunc_length = apply_truncation(sequences, trunc_strategy)
    labels, centroids, distances = apply_kmeans(X, k)
    plot_clustering(X, labels, centroids, distances)
    
    if interactive:
        interactive_plot_clustering(X, names, labels, centroids, distances, k, trunc_strategy, trunc_length)