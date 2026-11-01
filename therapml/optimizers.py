from typing import Any, Dict, List
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


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
        weight_history: list[np.ndarray] = [weights.copy()]

        for _ in range(n_steps):
            #SGD random sampling
            idx = np.random.randint(0,N)
            x_i = X[idx]
            y_i = y[idx]

            error = np.dot(x_i,weights) - y_i
            grad  = error * x_i # (X_T*W - y)*X_T
            weights = weights - lr*grad #weight update via a single sampled value x_i
            weight_history.append(weights.copy())

            #compute MSE loss
            residual = X @ weights - y
            cur_mse_loss = float(0.5 * (np.mean(residual ** 2)))
            loss_history.append(cur_mse_loss)

        return {
            "weights" : weights,
            "loss_history" : loss_history,
            "weight_history": np.asarray(weight_history),
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
            weight_history: list[np.ndarray] = [weights.copy()]
            weight_history: list[np.ndarray] = [weights.copy()]
            V_w = np.zeros(D, dtype=np.float64)

            for _ in range(n_steps):
                #SGD random sampling
                idx = np.random.randint(0,N)
                x_i = X[idx]
                y_i = y[idx]
    
                error = np.dot(x_i,weights) - y_i
                grad  = error * x_i # (X_T*W - y)*X_T
                V_w = momentum*V_w + grad
                weights = weights - lr*V_w #weight update via a single sampled value x_i
                weight_history.append(weights.copy())
    
                #compute MSE loss
                residual = X @ weights - y
                cur_mse_loss = float(0.5 * (np.mean(residual ** 2)))
                loss_history.append(cur_mse_loss)
    
            return {
                "weights" : weights,
                "loss_history" : loss_history,
                "weight_history": np.asarray(weight_history),
            }

    @staticmethod
    def sgd_adam_fit(X: NDArray, y: NDArray, lr: float, n_steps: int, lr_decay = False, rate = 0.40):
            """Vanilla SGD on linear regression"""
    
            # ensure proper shape of input and reference output
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y,dtype=np.float64).reshape(-1)
            N,D = X.shape
    
            #D = num of features of X
            #No of weights = no of features = D
            weights = np.zeros(D, dtype=np.float64)
            loss_history: list[float] = []
            weight_history: list[np.ndarray] = [weights.copy()]
    
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
                weight_history.append(weights.copy())
                #compute MSE loss
                residual = X @ weights - y
                cur_mse_loss = float(0.5 * (np.mean(residual ** 2)))
                loss_history.append(cur_mse_loss)

                if lr_decay == True and t >= n_steps * 0.80:
                     lr = lr * rate
    
            return {
                "weights" : weights,
                "loss_history" : loss_history,
                "weight_history": np.asarray(weight_history),
            }





def plot_optimization_analysis(
    X: np.ndarray, 
    y: np.ndarray, 
    weight_histories: Dict[str, np.ndarray], 
    loss_histories: Dict[str, List[float]]
):
    """
    Plots the 2D loss landscape with optimizer trajectories and a loss convergence chart.
    
    Args:
        X: Design matrix of shape (N, 2) for a 2D weight landscape.
        y: Target values of shape (N,).
        weight_histories: Dictionary mapping optimizer names to arrays of shape (steps, 2).
        loss_histories: Dictionary mapping optimizer names to lists of loss values.
    """
    fig, (ax_landscape, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Determine spatial bounds for the landscape grid based on all trajectories
    all_weights = np.vstack(list(weight_histories.values()))
    w0_min, w0_max = all_weights[:, 0].min(), all_weights[:, 0].max()
    w1_min, w1_max = all_weights[:, 1].min(), all_weights[:, 1].max()
    
    # Add padding to the bounds
    pad0, pad1 = (w0_max - w0_min) * 0.2, (w1_max - w1_min) * 0.2
    w0_range = np.linspace(w0_min - pad0, w0_max + pad0, 100)
    w1_range = np.linspace(w1_min - pad1, w1_max + pad1, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    
    # 2. Compute Mean Squared Error across the grid
    Loss_Surface = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w = np.array([W0[i, j], W1[i, j]])
            residuals = X @ w - y
            Loss_Surface[i, j] = np.mean(residuals**2) / 2.0
            
    # 3. Plot Landscape Contours
    cp = ax_landscape.contour(W0, W1, Loss_Surface, levels=25, cmap='viridis', alpha=0.7)
    ax_landscape.clabel(cp, inline=True, fontsize=8)
    
    # 4. Overlay Trajectories
    colors = ['crimson', 'dodgerblue', 'darkorange']
    for idx, (name, weights) in enumerate(weight_histories.items()):
        color = colors[idx % len(colors)]
        # Plot path
        ax_landscape.plot(weights[:, 0], weights[:, 1], label=name, 
                          color=color, linewidth=2, alpha=0.8)
        # Plot start and end points
        ax_landscape.scatter(weights[0, 0], weights[0, 1], color=color, marker='s', s=50)
        ax_landscape.scatter(weights[-1, 0], weights[-1, 1], color=color, marker='*', s=150)
        
    ax_landscape.set_title("2D Loss Landscape & Optimizer Trajectories")
    ax_landscape.set_xlabel(r"$w_0$")
    ax_landscape.set_ylabel(r"$w_1$")
    ax_landscape.legend()
    
    # 5. Plot Convergence Speed (Loss vs. Iteration)
    for idx, (name, losses) in enumerate(loss_histories.items()):
        color = colors[idx % len(colors)]
        ax_loss.plot(range(len(losses)), losses, label=name, color=color, linewidth=2)
        
    ax_loss.set_title("Convergence Speed (Loss vs. Iteration)")
    ax_loss.set_xlabel("Iteration Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log') # Log scale often helps visualize the tail end of convergence
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is in sys.path so we can import from tests
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 1. Import dataset generator from test_optimizers
    from tests.phase_1.part_1.test_optimizers import _make_regression

    # 2. Generate 2D regression data (d=2 for 2D landscape visualization)
    X, y, w_true = _make_regression(n=500, d=2, seed=42)
    print(f"True weights: {w_true}")

    # 3. Run optimizers
    n_steps = 2000

    print("Running Vanilla SGD...")
    res_sgd = Optimizers.sgd(X, y, lr=0.001, n_steps=n_steps)
    print(f"Vanilla SGD final weights: {res_sgd['weights']}")

    print("Running SGD + Momentum...")
    res_momentum = Optimizers.sgd_momentum(X, y, lr=0.001, momentum=0.9, n_steps=n_steps)
    print(f"SGD + Momentum final weights: {res_momentum['weights']}")

    # learning rate decay does make adam less noisy overal.
    print("Running Adam...")
    res_adam = Optimizers.sgd_adam_fit(X, y, lr=0.01, n_steps=n_steps,lr_decay=True)
    print(f"Adam final weights: {res_adam['weights']}")

    # 4. Prepare histories for plotting
    weight_histories = {
        "Vanilla SGD": res_sgd["weight_history"],
        "SGD + Momentum": res_momentum["weight_history"],
        "Adam": res_adam["weight_history"],
    }

    loss_histories = {
        "Vanilla SGD": res_sgd["loss_history"],
        "SGD + Momentum": res_momentum["loss_history"],
        "Adam": res_adam["loss_history"],
    }

    # 5. Plot analysis
    plot_optimization_analysis(X, y, weight_histories, loss_histories)

        