import gzip
import os
import urllib.request
from typing import Any, Dict
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_mnist_with_progress(data_dir: str = ".mnist_data"):
    """Download and load MNIST with a real-time tqdm progress bar."""
    os.makedirs(data_dir, exist_ok=True)
    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    files = {
        "images": ("train-images-idx3-ubyte.gz", 16),
        "labels": ("train-labels-idx1-ubyte.gz", 8),
    }

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    data = {}
    for key, (filename, offset) in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            downloaded = False
            for base_url in mirrors:
                try:
                    url = base_url + filename
                    with DownloadProgressBar(
                        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {filename}"
                    ) as t:
                        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)
                    downloaded = True
                    break
                except Exception:
                    continue
            if not downloaded:
                # Fallback to fetch_openml if mirrors fail
                print("Mirrors unreachable, falling back to fetch_openml...")
                mnist = fetch_openml("mnist_784", version=1, parser="auto")
                return mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

        with gzip.open(filepath, "rb") as f:
            if key == "images":
                data[key] = np.frombuffer(f.read(), dtype=np.uint8, offset=offset).reshape(-1, 784).astype(np.float64)
            else:
                data[key] = np.frombuffer(f.read(), dtype=np.uint8, offset=offset).astype(int)

    return data["images"], data["labels"]

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
    discarded_indices = sorted_idices[k:]

    #Only keep top K eigenvectors and values'
    top_eigenvalues = eigenvalues[top_k_indixes]
    discarded_eigenvalues = eigenvalues[discarded_indices]
    B = eigenvectors[:,top_k_indixes] #B is the required projection matrix
    components = B.T

    #projected coords onto lower dim space
    projected = np.dot(data_centered,B) 
    reconstructed = np.dot(projected,components) + mean # mean takes the centreted data to the affine space (offset from zero)

    return {
        "projected"     : projected,
        "components"    : components,
        "eigenvalues"   : top_eigenvalues,
        "discarded_eigenvalues" : discarded_eigenvalues,
        "mean"          : mean,
        "reconstructed" : reconstructed,
    }

if __name__ == "__main__":
    # 1. Load MNIST Dataset with progress bar
    # Using 10,000 samples for efficient computation.
    print("Fetching MNIST data...")
    X_all, y_all = load_mnist_with_progress()
    X = X_all[:10000]
    y = y_all[:10000]
  
    # 2. Visualize 10,000 digits projected onto 2 principal components
    _pca = pca(X, k=2)
    projected_2d = _pca["projected"]

    # measuring the reconstruction error
    ks = [2,10,50,100,784]

    for _k in ks:
      hdpca = pca(X, k = _k )
      discarded_eigenvalues = hdpca["discarded_eigenvalues"]
      J = np.sum(discarded_eigenvalues)
      print(f"For k={_k} , reconstruction error J = {J} ")
  
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projected_2d[:, 0], projected_2d[:, 1], c=y, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Digit Label')
    plt.title('MNIST: 10,000 Digits Embedded in a 2D Principal Subspace')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.show()