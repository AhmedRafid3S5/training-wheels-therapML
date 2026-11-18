import re
import unicodedata

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts  # create a dict or use the one provided
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    function behavior:
    provided a list of ids and a token id pair and a new index value
    return the new id list

    example:
    [1,2,3,1,2], pair = (1,2), idx = 4
    returns [4,3,4]
    """
    new_ids = []
    i = 0

    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids

# Base Class for Tokenizer
class Tokenizer:
    def __init__(self):
        self.merges = {}  # (1, 2) -> 4 (see above example)
        self.pattern = ""
        self.special_tokens = {}  # e.g. {'<PAD>': 596, '<BOS>': 597, ...}
        self.special_pattern = None
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False, special_tokens=None):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def get_vocab(self):
        """Return the learned vocabulary"""
        return self.vocab


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False, special_tokens=None):
        assert vocab_size >= 256, "vocab_size must be at least 256 for base byte tokens"
        if special_tokens is None:
            special_tokens = []

        num_specials = len(special_tokens)
        num_merges = vocab_size - 256 - num_specials
        assert num_merges >= 0, f"vocab_size {vocab_size} is too small for 256 bytes + {num_specials} special tokens"

        # 1. Protect special tokens from merge training:
        # If special tokens appear in the corpus, split the corpus around them
        # so merges are never learned across or within special tokens.
        if special_tokens:
            pattern = "|".join(re.escape(s) for s in special_tokens)
            chunks = [c for c in re.split(pattern, text) if c]
        else:
            chunks = [text]

        # Convert each chunk into initial UTF-8 byte token IDs
        chunk_ids = [list(c.encode("utf-8")) for c in chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  # Raw bytes 0..255

        # 2. Learn merges greedily up to num_merges
        for i in range(num_merges):
            stats = {}
            for c_ids in chunk_ids:
                get_stats(c_ids, stats)

            # If text is fully merged or no pairs remain, exit early gracefully
            if not stats:
                break

            # Pick pair with highest frequency (deterministic tie-breaking)
            pair = max(stats.keys(), key=lambda p: (stats[p], -p[0], -p[1]))
            idx = 256 + i
            chunk_ids = [merge(c_ids, pair, idx) for c_ids in chunk_ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r}) had {stats[pair]} occurrences")

        # 3. Add special tokens with IDs placed strictly AFTER all merges
        # IDs: [vocab_size - num_specials .. vocab_size - 1]
        self.special_tokens = {}
        for i, s in enumerate(special_tokens):
            special_id = vocab_size - num_specials + i
            self.special_tokens[s] = special_id
            vocab[special_id] = s.encode("utf-8")

        # Pre-compile regex for matching special tokens atomically during encode
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens.keys()) + ")"
            self.special_pattern = re.compile(pattern)
        else:
            self.special_pattern = None

        self.merges = merges
        self.vocab = vocab

    def _encode_chunk(self, text_chunk: str) -> list[int]:
        """Encodes a plain text segment (containing no special tokens) using learned BPE merges."""
        ids = list(text_chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        - Special tokens are matched atomically via regex and assigned their single special ID.
        - Regular text is encoded via byte-level BPE merges.
        - Because byte-level BPE covers 100% of UTF-8 byte space (0..255), <UNK> is NEVER emitted.
        - Whitespace and Unicode characters are strictly preserved without normalization.
        """
        if not text:
            return []

        if not self.special_pattern:
            return self._encode_chunk(text)

        # Split on special tokens, preserving the matched delimiters
        parts = self.special_pattern.split(text)
        token_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # Special token is atomic — emit single special ID directly
                token_ids.append(self.special_tokens[part])
            else:
                # Regular text passes through byte-level BPE
                token_ids.extend(self._encode_chunk(part))

        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back into string.
        - Maps each ID to its corresponding bytes in the vocabulary.
        - Decodes with errors='replace' so partial or invalid UTF-8 sequences (e.g. half of an emoji)
          safely become U+FFFD without crashing.
        """
        text_bytes = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")