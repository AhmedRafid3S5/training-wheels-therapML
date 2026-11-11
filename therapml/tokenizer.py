import unicodedata

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts #create a dict or use the one provided
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair,0) + 1
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

        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids

# Base Class for Tokenizer
class Tokenizer:

    def __init__(self):
        self.merges = {} # (1,2) -> 4 (see above example)
        self.pattern = ""
        self.special_tokens = {} #e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1] #p1,p2 itself act as index and the value it holds in the byte equivalent of the int(po) and int(p1)
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def get_vocab(self):
        """Return the learned vocabulary"""
        return self.vocab
        

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256 #initially must contain all 256 chars of ASCII
        # calc number of runs to meet rquired vocab size
        num_merges = vocab_size - 256

        text_in_bytes = text.encode("utf-8")
        ids = list(text_in_bytes) #white space is also considered as a token

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # a mapping of idx to bytes
        for i in range(num_merges):
            stats = get_stats(ids) # stats hold frequency of each pair of tokens
            # fetch the pair with highest frequency
            pair = max(stats.keys(), key=lambda p:(stats[p], -p[0], -p[1]))
            idx = 256 + i
            ids = merge(ids,pair,idx)
            merges[pair] = idx # add the new most-frequent pair back to the merges dictionary
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        """
        Provided a sequence of token ids, return the decoded
        text according to the vocabulary built up by training on a corpus
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        Provided a string, convert the string into token ids
        based on the learned vocabulary
        """

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            #fetch the lowest merge index pair
            #if a pair is not in the vocab, return inf as its index
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx) # update the ids list 
        return ids
        
    