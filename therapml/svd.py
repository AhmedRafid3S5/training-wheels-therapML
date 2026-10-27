from typing import Any, Dict
import numpy as np

def truncated_svd(matrix, rank):
    """ Manually implemented SVD Func"""

    m,n = matrix.shape
    AtA = np.dot(matrix.T,matrix)

    eigenvalues, eigenvectors = np.linalg.eig(AtA)

    # sigma = sqrt(lambda), so eigenvalues must be clipped to non negative values
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = eigenvectors[:, sorted_indices]

    #Finding Sigma S
    singular_values = np.sqrt(eigenvalues)
    S = singular_values[:rank]

    #Truncated V upto rank no. of cols
    V_k = V[:,:rank]
    V_t = V_k.T # V Transpose

    #Computing U

    U = np.zeros((m, rank))
    delta = 1e-10 # to avoid zero error
    for i in range(rank):
        if S[i] > 0.0:
            U[:,i] = np.dot(matrix,V_k[:,i])/ S[i]
        else:
            # for rank deficient / zero singular values, use cononical vector as placeholder
            u = np.zeros(m)
            u[i % m ] = 1.0
            U[:,i] = u

    # low rank reconstruction
    reconstructed = np.dot(U * S, V_t)

    return {
        "U" : U,
        "S" : S,
        "Vt": V_t,
        "reconstructed": reconstructed
    }



