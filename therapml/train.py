"""
Training Script for Mini-Llama on TinyShakespeare
=================================================

Requirements implemented:
  1. Model: MiniLlama (decoder-only transformer with GQA, RoPE, RMSNorm, SwiGLU, and weight tying).
  2. Optimizer: Custom MyAdam (AdamW) with decoupled weight decay.
  3. Gradient Clipping: torch.nn.utils.clip_grad_norm_ with max_norm=1.0.
  4. Learning Rate Schedule: Linear warmup (first 10% of steps) + Cosine decay (remaining 90%).
  5. Mixed-Precision Training: torch.autocast and GradScaler.
  6. Precision Benchmark: Compares training throughput (steps/sec) in FP32 vs. BF16/FP16.
  7. Experiment Logging: Logs loss, learning rate, gradient norms, and generated text samples
     to Weights & Biases (W&B) if available, with clean local console and CSV logging fallback.

Code style:
  - Step-by-step logic without compressed one-liners.
  - Descriptive variable names and thorough line-by-line comments.
"""

import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our custom components
from therapml.miniLLaMa import MiniLlama
from therapml.adamw import MyAdam
from therapml.tokenizer import BasicTokenizer

# Optional Weights & Biases logger
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# 1. Configuration & Hyperparameters
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "tinyshakespeare.txt"

# Model Architecture (Compact Mini-Llama for fast and effective training)
VOCAB_SIZE = 1000          # Vocabulary size (256 raw bytes + 740 merges + 4 specials)
CONTEXT_LENGTH = 128       # Maximum sequence length (tokens per sample)
D_MODEL = 256              # Embedding / hidden representation dimension
NUM_LAYERS = 4             # Number of stacked Transformer blocks
NUM_HEADS = 8              # Number of query attention heads
NUM_KV_HEADS = 2           # Number of key/value heads for Grouped Query Attention (4:1 ratio)
D_FF = 688                 # Hidden dimension for the SwiGLU Feed-Forward Network
ROPE_THETA = 10000.0       # Base frequency for Rotary Positional Embeddings

# Training Parameters
BATCH_SIZE = 16            # Number of sequences per training step
TOTAL_STEPS = 1000         # Total optimization steps
MAX_LR = 1e-3              # Peak learning rate reached at end of warmup
MIN_LR = 1e-4              # Minimum learning rate at end of cosine decay
WARMUP_FRAC = 0.1          # 10% of total steps for linear warmup
WEIGHT_DECAY = 0.01        # Decoupled weight decay for AdamW
MAX_NORM = 1.0             # Maximum gradient L2 norm for clipping

# Evaluation and Logging Intervals
EVAL_INTERVAL = 50         # Evaluate validation loss every 50 steps
EVAL_STEPS = 10            # Number of batches to average for validation loss
SAMPLE_INTERVAL = 200      # Generate sample text every 200 steps
BENCHMARK_SPEED = True     # Run FP32 vs mixed-precision speed comparison before training
USE_WANDB = False          # Set to True to log to Weights & Biases (if installed and logged in)


# =============================================================================
# 2. Learning Rate Scheduler
# =============================================================================

def get_learning_rate(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_frac: float = 0.1) -> float:
    """
    Computes the learning rate for the current step using:
      - Linear Warmup for the first `warmup_frac` of steps (from 0 up to max_lr).
      - Cosine Decay for the remaining steps (from max_lr down to min_lr).
    """
    # Number of warmup steps (e.g. 10% of 1000 = 100 steps)
    warmup_steps = int(warmup_frac * total_steps)

    if step < warmup_steps:
        # Phase 1: Linear warmup from 0.0 to max_lr
        current_lr = max_lr * (step / warmup_steps)
        return current_lr
    elif step <= total_steps:
        # Phase 2: Cosine decay from max_lr down to min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = min_lr + cosine_factor * (max_lr - min_lr)
        return current_lr
    else:
        # Past total_steps: maintain minimum learning rate
        return min_lr


# =============================================================================
# 3. Dataset Preparation & Batching
# =============================================================================

def prepare_data(text_path: Path, vocab_size: int):
    """
    Loads text from file, trains a BasicTokenizer, and splits tokens into train/val sets.
    """
    print(f"Loading raw text from: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Loaded {len(raw_text):,} characters.")

    # Train a BPE tokenizer with special tokens
    print("Training BPE tokenizer...")
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    tokenizer = BasicTokenizer()

    # Train on the first 50,000 characters to build the vocabulary
    train_slice = raw_text[:50_000]
    tokenizer.train(train_slice, vocab_size=vocab_size, special_tokens=special_tokens)
    print(f"Tokenizer trained with vocab size: {len(tokenizer.vocab)}")

    # Encode the entire corpus into token IDs
    print("Encoding corpus into token IDs...")
    all_tokens = tokenizer.encode(raw_text)
    print(f"Total encoded tokens: {len(all_tokens):,}")

    # Split into Train (90%) and Validation (10%) sets
    split_index = int(0.9 * len(all_tokens))
    train_tokens = torch.tensor(all_tokens[:split_index], dtype=torch.long)
    val_tokens = torch.tensor(all_tokens[split_index:], dtype=torch.long)

    print(f"Train split: {len(train_tokens):,} tokens")
    print(f"Val split:   {len(val_tokens):,} tokens")

    return tokenizer, train_tokens, val_tokens


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device: str):
    """
    Samples a random batch of sequences of length `context_length`.
    Returns:
      x: Inputs of shape (batch_size, context_length)
      y: Next-token targets of shape (batch_size, context_length)
    """
    # Pick random starting indices in the dataset
    max_start_idx = len(data) - context_length - 1
    # By picking a random start index and blinding carving out exactly context_length tokens
    # , we guarantee that every single sequence in the batch is perfectly rectangular.
    random_starts = torch.randint(low=0, high=max_start_idx, size=(batch_size,))

    # Collect input sequences and corresponding target sequences (shifted by 1 token)
    input_chunks = []
    target_chunks = []
    for start in random_starts:
        x_chunk = data[start : start + context_length]
        y_chunk = data[start + 1 : start + context_length + 1]
        input_chunks.append(x_chunk)
        target_chunks.append(y_chunk)

    # Stack into 2D tensors of shape (batch_size, context_length)
    x = torch.stack(input_chunks).to(device)
    y = torch.stack(target_chunks).to(device)

    return x, y


# =============================================================================
# 4. Text Generation Utility
# =============================================================================

def generate_sample(
    model: nn.Module,
    tokenizer: BasicTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    context_length: int = 128,
    device: str = "cpu",
) -> str:
    """
    Autoregressively generates text starting from a given prompt string.
    """
    model.eval()

    # Encode prompt to token IDs
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [ord("A")]

    current_ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if length exceeds context_length
            context_window = current_ids[-context_length:]

            # Convert to tensor: shape (1, seq_len)
            input_tensor = torch.tensor([context_window], dtype=torch.long, device=device)

            # Forward pass through model
            logits = model(input_tensor)

            # Get logits of the very last token position: shape (vocab_size,)
            last_token_logits = logits[0, -1, :]

            # Apply temperature scaling and sample
            if temperature == 0.0:
                # Greedy choice: pick highest logit
                next_token_id = int(torch.argmax(last_token_logits).item())
            else:
                # Temperature sampling
                scaled_logits = last_token_logits / temperature
                probabilities = F.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probabilities, num_samples=1).item())

            current_ids.append(next_token_id)

    # Decode back into readable string
    generated_text = tokenizer.decode(current_ids)
    return generated_text


# =============================================================================
# 5. Mixed-Precision Speed Benchmark (FP32 vs. BF16 / FP16)
# =============================================================================

def compare_training_speed(
    model: nn.Module,
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    device: str,
    num_steps: int = 20,
):
    """
    Measures and compares training throughput (steps/sec) in:
      1. Standard FP32 (Full Precision)
      2. Mixed Precision (BF16 or FP16 via torch.autocast)
    """
    print("\n" + "=" * 65)
    print("  MIXED-PRECISION SPEED BENCHMARK (FP32 vs. Mixed-Precision)")
    print("=" * 65)

    # Determine available mixed precision format
    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        dtype_name = "BF16" if amp_dtype == torch.bfloat16 else "FP16"
    else:
        amp_dtype = torch.bfloat16
        dtype_name = "BF16 (CPU)"

    # Optimizer for benchmarking
    test_optimizer = MyAdam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda" and amp_dtype == torch.float16))

    # Helper function to benchmark a specific precision mode
    def benchmark_mode(use_amp: bool, label: str):
        model.train()

        # Warmup steps (untimed)
        for _ in range(3):
            test_optimizer.zero_grad()
            x, y = get_batch(data, batch_size, context_length, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            test_optimizer.step()

        # Sync GPU clocks if CUDA is used
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed steps
        start_time = time.perf_counter()
        for _ in range(num_steps):
            test_optimizer.zero_grad()
            x, y = get_batch(data, batch_size, context_length, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            test_optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        steps_per_sec = num_steps / elapsed
        print(f"  {label:<22}: {steps_per_sec:>8.2f} steps/sec ({elapsed:>6.3f}s total)")
        return steps_per_sec

    # Run FP32 (Full precision)
    speed_fp32 = benchmark_mode(use_amp=False, label="FP32 (Full Precision)")

    # Run Mixed Precision (Autocast)
    speed_amp = benchmark_mode(use_amp=True, label=f"Autocast ({dtype_name})")

    speedup = speed_amp / speed_fp32
    print(f"\n  Speedup from Mixed Precision: {speedup:.2f}x")
    print("=" * 65 + "\n")


# =============================================================================
# 6. Main Training Loop
# =============================================================================

def train():
    # ── 1. Device Selection ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = (amp_dtype == torch.float16)  # FP16 needs GradScaler to prevent underflow
    else:
        device = "cpu"
        amp_dtype = torch.bfloat16
        use_scaler = False

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Autocast Dtype: {amp_dtype}")

    # ── 2. Data Preparation ──────────────────────────────────────────────────
    tokenizer, train_tokens, val_tokens = prepare_data(DATA_PATH, vocab_size=VOCAB_SIZE)

    # ── 3. Model Instantiation ───────────────────────────────────────────────
    print("\nInstantiating MiniLlama model...")
    model = MiniLlama(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        tie_weights=True, 
        use_MHA_GELU=True # Weight tying between embedding and output head
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # ── 4. Mixed-Precision Speed Benchmark ────────────────────────────────────
    if BENCHMARK_SPEED:
        compare_training_speed(
            model=model,
            data=train_tokens,
            context_length=CONTEXT_LENGTH,
            batch_size=BATCH_SIZE,
            device=device,
            num_steps=15,
        )

    # ── 5. Optimizer, Scaler, and W&B Logging ─────────────────────────────────
    # Use our custom MyAdam (AdamW with decoupled weight decay)
    optimizer = MyAdam(
        model.parameters(),
        lr=MIN_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # PyTorch GradScaler for mixed precision (avoids underflow in FP16)
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    # Initialize Weights & Biases if requested and installed
    if USE_WANDB and HAS_WANDB:
        wandb.init(
            project="mini-llama-training",
            config={
                "vocab_size": VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH,
                "d_model": D_MODEL,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "d_ff": D_FF,
                "batch_size": BATCH_SIZE,
                "total_steps": TOTAL_STEPS,
                "max_lr": MAX_LR,
                "min_lr": MIN_LR,
                "max_norm": MAX_NORM,
            },
        )
        print("Logged in to Weights & Biases (W&B).")

    # ── 6. Optimization Loop ─────────────────────────────────────────────────
    print("\nStarting Training Loop...")
    print(f"{'Step':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Perplexity':<12} | {'Grad Norm':<10} | {'LR':<10}")
    print("-" * 75)

    model.train()
    start_time = time.perf_counter()

    for step in range(1, TOTAL_STEPS + 1):
        # 1. Update Learning Rate according to Warmup + Cosine schedule
        current_lr = get_learning_rate(
            step=step,
            total_steps=TOTAL_STEPS,
            max_lr=MAX_LR,
            min_lr=MIN_LR,
            warmup_frac=WARMUP_FRAC,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 2. Sample training batch
        x_batch, y_batch = get_batch(train_tokens, BATCH_SIZE, CONTEXT_LENGTH, device)

        # 3. Reset gradients
        optimizer.zero_grad()

        # 4. Forward pass with Mixed-Precision (autocast)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            logits = model(x_batch)  # Shape: (batch_size, context_length, vocab_size)

            # Flatten logits and targets to compute cross-entropy loss
            flat_logits = logits.reshape(-1, VOCAB_SIZE)
            flat_targets = y_batch.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)

        # 5. Backward pass with GradScaler
        scaler.scale(loss).backward()

        # 6. Unscale gradients before clipping
        scaler.unscale_(optimizer)

        # 7. Gradient Clipping (max_norm = 1.0)
        # torch.nn.utils.clip_grad_norm_ clips gradients in-place and returns the pre-clip norm
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
        grad_norm = float(grad_norm_tensor.item())

        # 8. Optimizer step & update scaler
        scaler.step(optimizer)
        scaler.update()

        # ── Periodic Validation Evaluation ───────────────────────────────────
        if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for _ in range(EVAL_STEPS):
                    x_val, y_val = get_batch(val_tokens, BATCH_SIZE, CONTEXT_LENGTH, device)
                    with torch.autocast(device_type=device, dtype=amp_dtype):
                        v_logits = model(x_val)
                        v_loss = F.cross_entropy(v_logits.reshape(-1, VOCAB_SIZE), y_val.reshape(-1))
                    val_losses.append(v_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            perplexity = math.exp(avg_val_loss)

            print(
                f"{step:<7} | {loss.item():<12.4f} | {avg_val_loss:<10.4f} | "
                f"{perplexity:<12.2f} | {grad_norm:<10.4f} | {current_lr:<10.2e}"
            )

            # Log to W&B
            if USE_WANDB and HAS_WANDB:
                wandb.log({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": avg_val_loss,
                    "perplexity": perplexity,
                    "grad_norm": grad_norm,
                    "learning_rate": current_lr,
                })

            model.train()

        # ── Periodic Text Generation Sample ──────────────────────────────────
        if step % SAMPLE_INTERVAL == 0 or step == TOTAL_STEPS:
            sample_prompt = "ROMEO:"
            generated_text = generate_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=sample_prompt,
                max_new_tokens=40,
                temperature=0.8,
                context_length=CONTEXT_LENGTH,
                device=device,
            )
            print(f"\n[Step {step} Generated Sample]:\n{generated_text}\n")

            if USE_WANDB and HAS_WANDB:
                wandb.log({"step": step, "sample_text": generated_text})

    elapsed_total = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Training completed in {elapsed_total:.2f} seconds ({TOTAL_STEPS / elapsed_total:.1f} steps/sec).")

    # Save trained checkpoint
    save_path = PROJECT_ROOT / "mini_llama_MHAGELU_trained.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to: {save_path}")

    if USE_WANDB and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()
