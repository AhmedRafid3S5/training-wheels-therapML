from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt

def truncated_svd(matrix, rank):
    """ Manually implemented SVD Func"""

    m,n = matrix.shape
    AtA = np.dot(matrix.T,matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(AtA)

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



if __name__ == "__main__" :
  img_path = "therapml/grayscale-image.png"
  img = plt.imread(img_path)

  if img.ndim == 3:
    img_2d = np.mean(img[..., :3], axis=2)
  else:
    img_2d = img

  #convert to 2D array
  A = np.asarray(img_2d, dtype=np.float64)

  #compression at various ranks'

  ranks = [5, 10, 20, 50, 100]
  errors = []
  reconstructions = {}

  for r in ranks:
      result = truncated_svd(A,r)
      A_hat = result["reconstructed"]

      error = np.linalg.norm(A - A_hat, ord="fro")
      errors.append(error)
      reconstructions[r] = A_hat

  # -------------------------------------------------------------
  # 3. Plotting: Reconstructions & Frobenius Error Curve
  # -------------------------------------------------------------
  fig = plt.figure(figsize=(15, 8))  

  # Subplot: Original Image
  ax_orig = plt.subplot(2, 4, 1)
  ax_orig.imshow(A, cmap="gray")
  ax_orig.set_title(f"Original ({A.shape[0]}x{A.shape[1]})")
  ax_orig.axis("off")  

  # Subplots: Reconstructed Images at each rank
  for idx, r in enumerate(ranks):
    ax = plt.subplot(2, 4, idx + 2)
    # Clip values to valid display range [0, 255]
    clipped_img = np.clip(reconstructions[r], 0, 255)
    ax.imshow(clipped_img, cmap="gray")
    ax.set_title(f"Rank {r}\nError: {errors[idx]:.1f}")
    ax.axis("off")  

  # Subplot: Frobenius Norm Error vs. Rank
  ax_curve = plt.subplot2grid((2, 4), (1, 2), colspan=2, fig=fig)
  ax_curve.plot(ranks, errors, marker="o", color="crimson", linewidth=2)
  ax_curve.set_xlabel("Rank (r)")
  ax_curve.set_ylabel(r"$\|A - \hat{A}_{(r)}\|_F$ (Frobenius Norm)")
  ax_curve.set_title("Reconstruction Error vs. Rank")
  ax_curve.grid(True, linestyle="--", alpha=0.6)  

  plt.tight_layout()
  plt.show()
