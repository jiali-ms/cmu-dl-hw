import numpy as np
from collections import defaultdict

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''

def init_paths(symbols, prob):
    path_symbol_score, path_blank_score = defaultdict(int), defaultdict(int)

    path = ''
    path_blank_score[path] = prob[0]

    for i in range(len(symbols)):
        path = symbols[i]
        path_symbol_score[path] = prob[i+1]  # index 0 is blank

    return path_blank_score, path_symbol_score


def prune(path_blank_score, path_symbol_score, beam_width):
    # merge the score in two list, find the beam width
    scores = list(path_blank_score.values()) + list(path_symbol_score.values())
    cutoff = sorted(scores, reverse=True)[:beam_width][-1]

    pruned_path_blank_score = defaultdict(int)
    pruned_path_symbol_score = defaultdict(int)

    # prune the path by deleting from dict
    for path, score in path_blank_score.items():
        if score >= cutoff:
            pruned_path_blank_score[path] = score

    for path, score in path_symbol_score.items():
        if score >= cutoff:
            pruned_path_symbol_score[path] = score

    return pruned_path_blank_score, pruned_path_symbol_score


def extend_blank(path_blank_score, path_symbol_score, prob):
    updated_path_blank_score = defaultdict(int)

    # add blank symbol to path doesn't change path
    for path, score in path_blank_score.items():
        updated_path_blank_score[path] += score * prob[0]

    for path, score in path_symbol_score.items():
        updated_path_blank_score[path] += score * prob[0]

    return updated_path_blank_score


def extend_symbol(path_blank_score, path_symbol_score, symbols, prob):
    updated_path_symbol_score = defaultdict(int)

    # for every symbol, extend path
    for path, score in path_blank_score.items():
        for i in range(len(symbols)):
            new_path = path + symbols[i]
            updated_path_symbol_score[new_path] += score * prob[i+1]  # index 0 is blank

    for path, score in path_symbol_score.items():
        for i in range(len(symbols)):
            c = symbols[i]
            new_path = path if c == path[-1] else path + c
            updated_path_symbol_score[new_path] += score * prob[i+1]  # index 0 is blank

    return updated_path_symbol_score


def merge_path(path_blank_score, path_symbol_score):
    merged_path = path_symbol_score

    for path, score in path_blank_score.items():
        merged_path[path] += score

    return merged_path


def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    path = ''
    score = 1.0

    for i in range(y_probs.shape[1]):
        prob = y_probs[:, i, -1]
        idx = np.argmax(prob)
        symbol = '' if idx ==0 else SymbolSets[idx-1]
        path = path if path and symbol == path[-1] else path + symbol
        score = score * prob[idx]

    return path, score


##############################################################################

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search :-)
    for i in range(y_probs.shape[1]):
        print(y_probs[:,i,-1])

    # init paths, here path is a structure of path(str): score(int)
    new_path_blank_score, new_path_symbol_score = init_paths(SymbolSets, y_probs[:, 0, :])

    for i in range(1, y_probs.shape[1]):
        prob = y_probs[:, i, -1]
        path_blank_score, path_symbol_score = prune(new_path_blank_score, new_path_symbol_score, beam_width=BeamWidth)
        new_path_blank_score = extend_blank(path_blank_score, path_symbol_score, prob)
        new_path_symbol_score = extend_symbol(path_blank_score, path_symbol_score, SymbolSets, prob)

    merged_path = merge_path(new_path_blank_score, new_path_symbol_score)

    path = sorted(merged_path.items(), key=lambda x: -x[1])[0]

    return path[0], merged_path
