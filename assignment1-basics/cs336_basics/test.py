from tokenizer import Tokenizer
import json
import regex as re

if __name__ == "__main__":

    with open(r"bpe_cpu.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # dict[int, bytes]
    vocab = {
        int(token_id): bytes.fromhex(token["hex"])
        for token_id, token in data["vocab"].items()
    }

    # list[tuple[bytes, bytes]]
    merges = [
        (
            bytes.fromhex(pair["left"]["hex"]),
            bytes.fromhex(pair["right"]["hex"]),
        )
        for pair in data["merges"]
    ]

    special_tokens = ["<|endoftext|>"]
    BPE_Tokenizer = Tokenizer(vocab, merges, special_tokens)


    with open(r"../data/TinyStoriesV2-GPT4-valid.txt") as f:
        text = f.read()
    if special_tokens:
        alternatives = sorted(special_tokens, key=len, reverse=True)
        special_pat = re.compile("|".join(re.escape(token) for token in alternatives))

    for match in special_pat.finditer(text): 
        print(match)
        break
    
