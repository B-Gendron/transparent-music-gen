# Transparent Music Generation Research Project

## Quickstart
```
git clone https://github.com/B-Gendron/transparent-music-gen.git 
cd transparent-music-gen
pip install matplotlib numpy scikit-learn
```

## How to run clustering

### Single run

Simply by calling the dedicated python file:

```
python3 clustering.py [-h] [-p PATH] [-t TRUNCATION] [-k KMEANS] [-i]
```

The parameters are the following:
```
  -p PATH, --path PATH  The path to TPSD dataset (json format is expected)
  -t TRUNCATION, --truncation TRUNCATION
                        Select the truncation strategy to perform on TPSD sequences. Can be either 'right', meaning naive right truncation, or 'smart',
                        meaning taking care of keeping reapeting patterns.
  -k KMEANS, --kmeans KMEANS
                        The number of desired clusters. Default is 3.
  -i, --interactive     Whether the plot should be interactive, meaning we have the reference of each data point (song) by putting the mouse on it.
```

### Batched runs

Once in the repository, check for the shell script `run_clustering.sh`. Make sure it is executable, otherwise run:

```
chmod +x run_clustering.sh
```

Then, get the clustering results of k-means algorithm with $k \in \[ 3, 10\]$ using:

```
./run_clustering.sh
```

## Sequence truncation algorithm

The script `truncation.py` contains the code to truncate a sequence of chord without the risk of getting rid of repeating patterns. Here is an example output:

```
Initial sequence of chords: [1, 12, 3, 5, 2, 3, 5, 2, 7, 7, 8, 10]
Truncated sequence: [1, 12, 3, 5, 2, 3, 5, 2, 7, 7]
Repeated Patterns: [[2], [3, 5, 2], [5, 2], [7]]
```