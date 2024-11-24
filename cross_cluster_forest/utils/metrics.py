import numpy as np

def absnorm(vec, max_norm=False):
    """
    Normalize vector using max or min value.

    Parameters:
    -----------
    vec : array-like
        Input vector to normalize
    max_norm : bool
        If True, use maximum value for normalization
        If False, use minimum value

    Returns:
    --------
    numpy.ndarray : Normalized vector
    """
    vec = np.array(vec).ravel()
    sgn = np.sign(vec)
    vec = np.abs(vec)

    mvec = np.max(vec) if max_norm else np.min(vec)
    vec = np.abs(vec - mvec)

    sum_vec = np.sum(vec)
    if sum_vec == 0:
        return np.zeros_like(vec)
    return sgn * (vec / sum_vec)