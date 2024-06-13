# Transparent Music Generation Research Project

### Quickstart
```
git clone https://github.com/B-Gendron/transparent-music-gen.git 
cd transparent-music-gen
pip install matplotlib numpy scikit-learn
```

### Sequence truncation algorithm

The script `truncation.py` contains the code to truncate a sequence of chord without the risk of getting rid of repeating patterns. Here is an example output:

```
Initial sequence of chords: [1, 12, 3, 5, 2, 3, 5, 2, 7, 7, 8, 10]
Truncated sequence: [1, 12, 3, 5, 2, 3, 5, 2, 7, 7]
Repeated Patterns: [[2], [3, 5, 2], [5, 2], [7]]
```