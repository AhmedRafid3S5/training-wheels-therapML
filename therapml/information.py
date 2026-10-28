from typing import Any, Dict
import numpy as np
from numpy.typing import NDArray

class Entropy :
    def __init__(self, data: np.ndarray):
        pass

    @staticmethod
    def shanonEntropy(P: NDArray):
        """Avg num of bits required to convey the distribution"""
        #NOTE to self: log2(0.0) produces -inf
        #              only compute logs for non zero values
        P_nonzero = P[P > 0]
        if len(P_nonzero) == 0:
            return 0.0
        bits = np.log2(P_nonzero)
        return float(-np.sum(P_nonzero*bits))

    @staticmethod
    def klDivergence(P: NDArray, Q: NDArray):

        mask = P > 0 #returns a boolean mask ex: P = [1,2,-1,3] mask = [T,T,F,T]
        # boolean mask indexing 
        p = P[mask]
        q = Q[mask] 

        #for cases where P > 0 and Q == 0, return inf
        if np.any(q == 0 ):
            return float("inf")

        log_ratio = np.log2(p / q)
        return float(np.sum(p*log_ratio))

    @staticmethod
    def cross_entropy(P: NDArray, Q: NDArray):
        mask= P > 0
        p = P[mask]
        q = Q[mask]

        if np.any(q == 0):
            return float("inf")

        return float(-np.sum(p*np.log2(q)))

    @staticmethod
    def mutual_information(joint: NDArray):
        # the table itself is P(x,y), where joint[xi][yi] gives P(X=xi, Y=yi)
        px = joint.sum(axis=1) #sum across columns, columns have y values
        py = joint.sum(axis=0) #sum across rows, rows have x values

        px_py = np.outer(px,py) #product p(x)p(y) for denominator

        mask = joint > 0
        p_xy = joint[mask]
        p_marginal_prod = px_py[mask]

        return float(np.sum(p_xy*np.log2(p_xy/ p_marginal_prod)))
    