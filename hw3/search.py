import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)

    # return (forward_path, forward_prob)
    raise NotImplementedError



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

    # return (bestPath, mergedPathScores)
    raise NotImplementedError


if __name__ == "__main__":
    # # test greedy search when batch_size=1
    SymbolSets = ['a', 'b', 'c', 'd']
    y_prob = np.asarray([
        [[0.1], [0.1], [0.2], [0.6]],
        [[0.1], [0.1], [0.5], [0.1]],
        [[0.3], [0.1], [0.1], [0.1]],
        [[0.4], [0.1], [0.1], [0.1]],
        [[0.1], [0.6], [0.1], [0.1]],
    ])  # (lens sym+1, seq_lens, batch_size)=(5, 4, 1)
    # the forward path is :['c', 'd', 'a', '']
    # the forward probability is:[0.4, 0.6, 0.5, 0.6]
    forward_path, forward_prob = GreedySearch(SymbolSets, y_prob)
    print("forward path is:", forward_path, "\nforward probability is:", forward_prob)

    # todo:test greedy search when batch_size > 1
    # SymbolSets = ['a', 'b', 'c']
    # y_prob = np.asarray([
    #     [[0.1, 0.2], [0.2, 0.1], [0.1, 0.2], [0.2, 0.5], [0.5, 0.3]],
    #     [[0.3, 0.3], [0.2, 0.4], [0.6, 0.3], [0.5, 0.1], [0.1, 0.2]],
    #     [[0.1, 0.4], [0.4, 0.2], [0.2, 0.4], [0.1, 0.05], [0.1, 0.25]],
    #     [[0.5, 0.1], [0.2, 0.3], [0.1, 0.1], [0.2, 0.35], [0.3, 0.25]],
    # ])  # (lens sym+1, seq_lens, batch_size)=(4, 5, 2)
    #

    # test beam search when batch_size=1
    SymbolSets = ['a', 'b', 'c', 'd']
    y_prob = np.asarray([
        [[0.1], [0.1], [0.2], [0.6]],
        [[0.1], [0.1], [0.5], [0.1]],
        [[0.3], [0.1], [0.1], [0.1]],
        [[0.4], [0.1], [0.1], [0.1]],
        [[0.1], [0.6], [0.1], [0.1]],
    ])  # (lens sym+1, seq_lens, batch_size)=(5, 4, 1)
    beam_width = 2
    forward_path, forward_prob = BeamSearch(SymbolSets, y_prob, beam_width)
    print("forward path is:", forward_path, "\nforward probability is:", forward_prob)




