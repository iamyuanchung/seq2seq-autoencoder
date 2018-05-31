import math

import numpy as np


def naive_encoder(seq, vec_len):
    assert seq.shape[1] == 13
    assert vec_len % 13 == 0
    n_splits = vec_len / 13
    n_step = seq.shape[0] / n_splits
    seq_out = np.array([])
    for i in xrange(n_splits - 1):
        seq_out = np.append(seq_out, np.mean(seq[i * n_step: (i + 1) * n_step, :], axis=0))
    seq_out = np.append(seq_out, np.mean(seq[(n_splits - 1) * n_step:, :], axis=0))
    return seq_out


if __name__ == '__main__':
    seq = np.random.random((9, 13))
    # print str(seq) + '\n'

    print naive_encoder(seq, 13)
    print naive_encoder(seq, 52)
    # print naive_encoder(seq, 169)
