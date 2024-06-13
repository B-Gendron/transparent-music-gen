def build_suffix_array(seq):
    suffixes = sorted([(seq[i:], i) for i in range(len(seq))])
    suffix_arr = [suffix[1] for suffix in suffixes]
    return suffix_arr

def build_lcp_array(seq, suffix_arr):
    n = len(seq)
    rank = [0] * n
    lcp = [0] * n
    
    for i, suffix_index in enumerate(suffix_arr):
        rank[suffix_index] = i
    
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_arr[rank[i] - 1]
            while (i + h < n and j + h < n and seq[i + h] == seq[j + h]):
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    
    return lcp

def find_repeated_patterns(seq):
    suffix_arr = build_suffix_array(seq)
    lcp = build_lcp_array(seq, suffix_arr)
    repeated_patterns = []
    
    for i in range(1, len(lcp)):
        if lcp[i] > 0:
            pattern = seq[suffix_arr[i]:suffix_arr[i] + lcp[i]]
            repeated_patterns.append(pattern)
    
    return repeated_patterns


def apply_truncation_strategy(seq, repeated_patterns):
    # the patterns are retrieve in the order of the sequence so we can do that since we do right truncation
    last_pattern = repeated_patterns[-1]
    pattern_length = len(last_pattern)

    # deduce the bound by searching for the last occurence of the last pattern
    right_truncation_bound = 0
    for i in range(len(seq) - pattern_length + 1):
        if seq[i:i + pattern_length] == last_pattern:
            right_truncation_bound = i + pattern_length

    return seq[:right_truncation_bound]


# test
seq = [1, 12, 3, 5, 2, 3, 5, 2, 7, 7, 8, 10]
repeated_patterns = find_repeated_patterns(seq)
truncated_sequence = apply_truncation_strategy(seq, repeated_patterns)

print(f"Initial sequence of chords: {seq}")
print(f"Truncated sequence: {truncated_sequence}")
print(f"Repeated Patterns: {repeated_patterns}")
