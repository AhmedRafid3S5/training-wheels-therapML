from typing import Any, Dict
import numpy as np
import math

def grad_des_mle(data, lr, steps):
    """Fit a 1-D Gaussian to *data* via gradient-descent MLE."""
    #initial values  
    N = len(data)

    # NOTE: Initialization with 0 and 1.0 is a bad choice and can lead to vanishing gradients
    # in such cases, no amount of steps will bring the mu and sig cloes to the ground values
    # always initialize with a small sample mean and std dev
    mu = np.mean(data[:(N//4)])
    sig = np.std(data[:(N//4)])

    for i in range(steps):
        # Note to self: dont be stingy on parenthesis for expressions
        delL_mu = -(1.0/ (N * (sig **2)))* np.sum(data - mu)
        delL_sig = (1.0/sig) - (1.0/ (N * (sig**3)))*np.sum((data-mu)**2)

        #updates
        mu = mu - lr*delL_mu
        sig = sig - lr*delL_sig

        #clip sigma to >0
        sig = max(sig, 1e-6)

    return {
        "mu" : mu,
        "sigma2" : (sig**2)
    }

def compare_closed_form(lr, steps, mu, sig, N):

    #generate mock gaussian data 
    data = np.random.normal(loc=mu, scale=sig, size=N)

    #closed form solution
    true_mu = np.mean(data)
    true_sig = np.std(data)

    #solutions via grad descent
    sol = grad_des_mle(data, lr, steps)
    aprox_mu = sol["mu"]
    aprox_sig = math.sqrt(sol["sigma2"])

    print(f"Approximate mu: {aprox_mu:.4f}, Approximate sig: {aprox_sig:.4f}")
    print(f"Closed form mu: {true_mu:.4f}, Closed form sig: {true_sig:.4f}")
    print(f"mu error: {abs(aprox_mu - true_mu):.4f}, sig error: {abs(aprox_sig - true_sig):.4f}")


if __name__ == "__main__":
    #hyperparams
    lr = 0.01
    steps = 50

    #true params
    mu = 5.0
    sig = 2.0
    N = 10000

    compare_closed_form(lr, steps, mu, sig, N)
