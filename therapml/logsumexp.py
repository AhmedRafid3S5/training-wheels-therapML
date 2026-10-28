import numpy as np
from numpy.typing import NDArray

def logsumexp(x: NDArray):
    c = np.max(x)

    return float(c + np.log(np.sum((np.exp(x - c)))))

print("1. UNDERFLOW DEMONSTRATION & FIX")

n_samples = 100
prob = 1e-4
probabilities = np.full(n_samples, prob)

direct_product = 1.0
underflow_step = None

for i, p in enumerate(probabilities, start=1):
  direct_product *= p
  if direct_product == 0.0 and underflow_step is None:
    underflow_step = i

print(f"Total multiplications attempted: {n_samples}")
print(
    f"Direct multiplication underflowed to 0.0 at multiplication step"
    f" #{underflow_step}"
)
print(f"Final Direct Product Value: {direct_product}")

# If we try to take log(product) after underflow:
try:
  print(f"log(direct_product): {np.log(direct_product)}")  # Evaluates to -inf
except Exception as e:
  print(f"Error: {e}")

# --- The Fix: Log Space Summation ---
# log(prod_i p_i) = sum_i log(p_i)
log_probabilities = np.log(probabilities)
log_total = np.sum(log_probabilities)

print("\n--- The Fix (Operating in Log-Space) ---")
print(f"Log probability sum: {log_total:.4f}")
print(f"Exact theoretical value: 10^({log_total / np.log(10):.1f})")

logits = np.array([1000.0, 1001.0, 1002.0])

print(f"Logits: {logits}")
try:
  naive_exps = np.exp(logits)
  print(f"Naive exp(logits): {naive_exps}")
  naive_lse = np.log(np.sum(naive_exps))
  print(f"Naive log(sum(exp(logits))): {naive_lse}")  # inf
except Exception as e:
  print(f"Error during naive calculation: {e}")

# --- Fix: Stabilized Log-Sum-Exp ---

c = np.max(logits)
shifted_logits = logits - c  # [-2.0, -1.0,  0.0]
stable_lse = c + np.log(np.sum(np.exp(shifted_logits)))

print("\n--- The Fix (Subtracting max value c = 1002.0) ---")
print(f"Shifted exponents (logits - c): {shifted_logits}")
print(f"exp(shifted_logits): {np.exp(shifted_logits)}")
print(f"Numerically Stable Log-Sum-Exp: {stable_lse:.6f}")
