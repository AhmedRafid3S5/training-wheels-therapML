from typing import Any, Dict
import numpy as np
from numpy.typing import NDArray

class Optimizers:
    def __init__(self) -> None:
        pass

    @staticmethod
    def sgd(X: NDArray, y: NDArray, lr: float, n_steps: int):
        """Vanilla SGD on linear regression"""

        # ensure proper shape of input and reference output
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y,dtype=np.float64).reshape(-1)
        N,D = X.shape

        #D = num of features of X
        #No of weights = no of features = D
        weights = np.zeros(D, dtype=np.float64)
        loss_history: list[float] = []

        #initial loss
        initial_residuals = X @ weights - y
        initial_loss = float(0.5*(np.mean(initial_residuals**2)))
        #loss_history.append(initial_loss)

        for _ in range(n_steps):
            #SGD random sampling
            idx = np.random.randint(0,N)
            x_i = X[idx]
            y_i = y[idx]

            error = np.dot(x_i,weights) - y_i
            grad  = error * x_i # (X_T*W - y)*X_T
            weights = weights - lr*grad #weight update via a single sampled value x_i

            #compute MSE loss
            residual = X @ weights - y
            cur_mse_loss = float(0.5 * (np.mean(residual ** 2)))
            loss_history.append(cur_mse_loss)

        return {
            "weights" : weights,
            "loss_history" : loss_history,
        }

    @staticmethod
    def sgd_momentum(X: NDArray, y: NDArray, lr: float, momentum: float, n_steps: int):
            """Vanilla SGD on linear regression"""
    
            # ensure proper shape of input and reference output
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y,dtype=np.float64).reshape(-1)
            N,D = X.shape
    
            #D = num of features of X
            #No of weights = no of features = D
            weights = np.zeros(D, dtype=np.float64)
            loss_history: list[float] = []
    
            #initial loss
            initial_residuals = X @ weights - y
            initial_loss = float(0.5*(np.mean(initial_residuals**2)))
            #loss_history.append(initial_loss)
            V_w = 0.0

    
            for _ in range(n_steps):
                #SGD random sampling
                idx = np.random.randint(0,N)
                x_i = X[idx]
                y_i = y[idx]
    
                error = np.dot(x_i,weights) - y_i
                grad  = error * x_i # (X_T*W - y)*X_T
                V_w = momentum*V_w + grad
                weights = weights - lr*V_w #weight update via a single sampled value x_i
    
                #compute MSE loss
                residual = X @ weights - y
                cur_mse_loss = float(0.5 * (np.mean(residual ** 2)))
                loss_history.append(cur_mse_loss)
    
            return {
                "weights" : weights,
                "loss_history" : loss_history,
            }

    @staticmethod
    def sgd_adam_fit(X: NDArray, y: NDArray, lr: float, n_steps: int):
            """Vanilla SGD on linear regression"""
    
            # ensure proper shape of input and reference output
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y,dtype=np.float64).reshape(-1)
            N,D = X.shape
    
            #D = num of features of X
            #No of weights = no of features = D
            weights = np.zeros(D, dtype=np.float64)
            loss_history: list[float] = []
    
            #initial loss
            theta = np.zeros(D, dtype=np.float64)
            m = np.zeros_like(theta)
            v = np.zeros_like(theta)
            beta1 = 0.9
            beta2 = 0.999
            eps   = 1e-8

    
            for t in range(1,n_steps + 1):
                #SGD random sampling
                idx = np.random.randint(0,N)
                x_i = X[idx]
                y_i = y[idx]
    
                error = np.dot(x_i,weights) - y_i
                grad  = error * x_i # (X_T*W - y)*X_T

                #update 1st and 2nd momentum
                m = beta1*m + (1.0 - beta1)*grad
                v = beta2*v + (1.0 - beta2)*(grad**2)

                #compute bias-corrected estimates
                m_hat = m / (1-(beta1**t))
                v_hat = v / (1-(beta2**t))

                #update weights 
                weights = weights - (lr / (np.sqrt(v_hat) + eps)) * m_hat
                #compute MSE loss
                residual = X @ weights - y
                cur_mse_loss = float(0.5 * (np.mean(residual ** 2)))
                loss_history.append(cur_mse_loss)
    
            return {
                "weights" : weights,
                "loss_history" : loss_history,
            }




        