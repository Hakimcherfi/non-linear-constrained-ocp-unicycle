import numpy as np
from numpy import linalg

def null_space(A, eps=1e-10):
    """
    Returns an orthonormal basis of the null space of A, made of vectors whose singular values
    are close to zero (smaller than eps)
    """
    u,s,v = np.linalg.svd(A)
    m = A.shape[0]
    n = A.shape[1]
    if m<n:
        s = np.concatenate([s,np.zeros((n-m,))])
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, v, axis=0)
    return np.transpose(null_space)

def orth(A,eps=1e-10):
    """
    Returns an orthonormal basis of the range space of A, made of vectors whose singular values
    are strictly greater than eps
    """
    u,s,v = np.linalg.svd(A)
    notnull_mask = (s>eps)
    return np.compress(notnull_mask,u,axis=1)