import torch
import torch.nn.functional as F

# Import your model and tokenizer classes from your project files
from therapml.miniLLaMa import MiniLlama
from therapml.tokenizer import BasicTokenizer


def load_chatbot(
    weights_path: str = "mini_llama_trained.pt",
    tokenizer_path: str = "bpe_10_000.model",
    device: str | None = None,
) -> tuple[MiniLlama, BasicTokenizer, str]:
    """Loads the trained MiniLlama weights and tokenizer onto the target device."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # 1. Load tokenizer (adjust to match how you saved your trained BPE tokenizer)
    tokenizer = BasicTokenizer()
    tokenizer.load(tokenizer_path)

    # 2. Load weights.pt (handles both a full checkpoint dict and a raw state_dict)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback config matching your ~3.03M-parameter MiniLlama run
        config = {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 8,
            "num_kv_heads": 2,
            "use_MHA_GELU": True,
            "d_ff": 688,
            "rope_theta": 10_000.0,
            "tie_weights": True,
        }
        state_dict = checkpoint

    # 3. Instantiate model, load weights, and switch to inference mode
    model = MiniLlama(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, device


@torch.inference_mode()
def stream_reply(
    model: MiniLlama,
    tokenizer: BasicTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
    stop_strings: tuple[str, ...] = ("\nUser:", "\n\n"),
):
    """Yields newly generated text chunks one token at a time."""
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [ord(" ")]

    generated_ids = list(input_ids)
    prev_text = tokenizer.decode(generated_ids)
    generated_reply = ""

    for _ in range(max_new_tokens):
        # Keep only the most recent tokens that fit inside context_length
        context_window = generated_ids[-model.context_length :]
        x = torch.tensor([context_window], dtype=torch.long, device=device)

        # Forward pass -> extract logits for the final position
        logits = model(x)[0, -1, :]

        if temperature == 0.0:
            next_token_id = int(torch.argmax(logits).item())
        else:
            scaled_logits = logits / temperature

            # Top-K filtering: mask out low-probability tail tokens
            if top_k > 0:
                k = min(top_k, scaled_logits.size(-1))
                top_values, _ = torch.topk(scaled_logits, k)
                cutoff = top_values[-1]
                scaled_logits = torch.where(
                    scaled_logits < cutoff,
                    torch.tensor(float("-inf"), device=device),
                    scaled_logits,
                )

            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        generated_ids.append(next_token_id)

        # Decode full sequence and slice diff so multi-byte UTF-8 characters never split
        full_text = tokenizer.decode(generated_ids)
        new_chunk = full_text[len(prev_text) :]
        prev_text = full_text
        generated_reply += new_chunk

        # Stop early if the model starts hallucinating the next speaker turn
        if any(stop in generated_reply for stop in stop_strings):
            break

        yield new_chunk


def run_chat():
    model, tokenizer, device = load_chatbot("mini_llama_MHAGELU_trained.pt", "bpe_10_000.model")
    print(f"Loaded MiniLlama on {device}. Type '/reset' to clear history or '/quit' to exit.\n")

    conversation_history = ""

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit"}:
            break
        if user_input.lower() == "/reset":
            conversation_history = ""
            print("[Conversation history cleared]\n")
            continue

        # Append user turn and prime the model to respond
        conversation_history += f"User: {user_input}\nBot:"
        print("Bot:", end="", flush=True)

        bot_reply = ""
        for chunk in stream_reply(model, tokenizer, conversation_history, device):
            print(chunk, end="", flush=True)
            bot_reply += chunk

        print("\n")
        conversation_history += f"{bot_reply}\n"


if __name__ == "__main__":
    run_chat()