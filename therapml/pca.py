from typing import Any, Dict
import numpy as np


def pca(data, k):

    """data as 2D array of n_samples, n_features ( n >> k)"""

    mean = np.mean(data, axis=0) # axis = 0 means along the column, gives feature mean
    data_centered = data - mean
    cov_matrix = np.cov(data_centered, rowvar=False) # set rowvar to False treats cols as individual features
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    #ONLY TAKE REAL PART
    #Coz Eigenvalues can be imaginary as well
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    #Sort eigenvalues in descending order
    sorted_idices = np.argsort(eigenvalues)[::-1] # [::-1] forces descending order
    top_k_indixes = sorted_idices[:k]

    #Only keep top K eigenvectors and values'
    top_eigenvalues = eigenvalues[top_k_indixes]
    B = eigenvectors[:,top_k_indixes] #B is the required projection matrix
    components = B.T

    #projected coords onto lower dim space
    projected = np.dot(data_centered,B) 
    reconstructed = np.dot(projected,components) + mean # mean takes the centreted data to the affine space (offset from zero)

    return {
        "projected"     : projected,
        "components"    : components,
        "eigenvalues"   : top_eigenvalues,
        "mean"          : mean,
        "reconstructed" : reconstructed,
    }