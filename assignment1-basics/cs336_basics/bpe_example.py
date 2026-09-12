import regex as re

# 文本
text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

# 初始化词汇表，token（字节串） -> token ID
vocab = { bytes([i]): i for i in range(256)}

# 预分词
text_list = text.split()
text_freq = {}
for a in text_list:
    if a not in text_freq:
        text_freq[a] = 1
    else:
        text_freq[a] += 1

print(text_freq)

# 全局字节情况
word_freq = {}
for word, freq in text_freq.items():
    # UTF-8 编码
    word_bytes = word.encode("utf-8")
    # 每一个 byte 单独作为一个初始 token
    tokens = tuple(bytes([b]) for b in word_bytes)
    word_freq[tokens] = freq

for word, freq in word_freq.items():
    print(word, ":", freq)


# 统计相邻 token pair 的频率
def get_pair_freq(word_freq):
    pair_freq = {}
    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in pair_freq:
                pair_freq[pair] = freq
            else:
                pair_freq[pair] += freq

    return pair_freq

# 合并指定的 pair
def merge_pair(word_freq, best_pair):
    new_word_freq = {}
    for word, freq in word_freq.items():
        new_word = []
        i = 0
        while i < len(word):
            # 当前 token 和下一个 token
            # 是否刚好组成 best_pair
            if (i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]):
                # 合并两个 byte token
                merged_token = (word[i] + word[i + 1])
                new_word.append(merged_token)
                # 一次吃掉两个 token
                i += 2

            else:
                new_word.append(word[i])
                i += 1

        new_word_freq[tuple(new_word)] = freq

    return new_word_freq


num_merges = 6
merges = []
for merge_index in range(num_merges):
    pair_freq = get_pair_freq(word_freq)
    if not pair_freq:
        break
    # 获取频率最高的字节对
    best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
    best_freq = pair_freq[best_pair]
    # 记录 merge
    merges.append(best_pair)

    # 创建新的token，放进词汇表
    new_token = (best_pair[0] + best_pair[1])
    new_token_id = len(vocab)
    vocab[new_token] = new_token_id

    # 合并操作
    word_freq = merge_pair(word_freq, best_pair)
    print(f"\n================ 第 {merge_index + 1} 轮合并 ================")
    for word, freq in word_freq.items():
        print(word, ":", freq)
    print(vocab)
