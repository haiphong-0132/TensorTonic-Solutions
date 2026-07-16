import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return positional encoding of shape (seq_len, d_model).

    Even indices (0,2,4,...) use sin.
    Odd indices (1,3,5,...) use cos.
    If d_model is odd, the last column is sin.
    """
    pe = np.zeros((seq_len, d_model), dtype=float)

    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (base ** (2 * (i // 2) / d_model))

            if i % 2 == 0:
                pe[pos, i] = np.sin(angle)
            else:
                pe[pos, i] = np.cos(angle)

    return pe