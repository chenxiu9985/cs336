from train_bpe import train_bpe

if __name__ == "__main__":
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(vocab)
    print(merges)