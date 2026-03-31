import torch
import regex as re


# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Usage
## re.findall(PAT, "some text that i'll pre-tokenize")
## ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']

# read text -> split by speical tokens -> pre-tokenize -> count pre-tokens -> initialize vocab -> repeatedly compute best merge and apply it -> repeat vocab merges
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class BPETokenizer():
    def __init__(self, input_path, vocal_size, special_tokens):
        super().__init__()
        self.input_path = input_path
        self.vocab_size = vocal_size
        self.special_tokens = special_tokens
    
    def train(self):
        # 1. read original text
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 2. split by special tokens
        splited_text = text.split("<|endoftext|>")
        
        # 3. pre-tokenize noraml text segments with GPT-2 regex
        all_data = [re.findall(PAT, part) for part in splited_text]

        # 4. convert counted pre-tokens into training state
        pre_token_counts = {}
        for part_tokens in all_data:
            for tok in part_tokens:
                if tok not in pre_token_counts:
                    pre_token_counts[tok] = 0
                pre_token_counts[tok] += 1

        state = {}
        for tok, count in pre_token_counts.items():
            token_bytes = tok.encode("utf-8")
            symbols = tuple(bytes([b]) for b in token_bytes)
            state[symbols] = count

        # 5. initialize vocab()
        vocab = {}
        idx = 0
        for tok in self.special_tokens:
            vocab[idx] = tok.encode("utf-8")
            idx += 1
        for i in range(256):
            vocab[idx] = bytes([i])
            idx += 1
        
        # 6. initialize merges as empty list
        merges = []

        # 7. while current vocab size < target vocab size:
            # - compute pair counts from current state
            # - select the most frequent pair
            # - apply merge to every pre-token state
            # - append merge rule
            # - add merged btyes token to vocab
        while len(vocab) < self.vocab_size:
            pair_counts = {}

            for symbols, count in state.items():
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    if pair not in pair_counts:
                        pair_counts[pair] = 0
                    pair_counts[pair] += count
            
            if not pair_counts:
                break
            
            best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            merged_token = best_pair[0] + best_pair[1]

            updated_state = {}
            for symbols, count in state.items():
                updated_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        updated_symbols.append(merged_token)
                        i += 2
                    else:
                        updated_symbols.append(symbols[i])
                        i += 1
                updated_symbols = tuple(updated_symbols)
                
                if updated_symbols not in updated_state:
                    updated_state[updated_symbols] = 0

                updated_state[updated_symbols] = count
            
            merges.append(best_pair)
            vocab[idx] = merged_token
            idx += 1
            state = updated_state

        # import pdb; pdb.set_trace()
        return vocab, merges
        # 8. return vocab, merges


if __name__ == "__main__":
    input_path_debug = "data/TinyStoriesV2-GPT4-valid.txt"
    vocal_size_debug = 512
    special_tokens_debug = ["<|endoftext|>"]
    bpe = BPETokenizer(input_path=input_path_debug,
                       vocal_size=vocal_size_debug,
                       special_tokens=special_tokens_debug)
    vocab, merges = bpe.train()