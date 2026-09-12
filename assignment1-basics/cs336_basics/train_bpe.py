import regex as re
from collections import Counter,defaultdict

# 模式匹配来分割文本
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
compiled_pat = re.compile(GPT2_PAT)

# 初始化词汇表
BYTE_TOKENS = tuple(bytes([i]) for i in range(256)) 

def load_file(file_name: str):
    with open(file_name,"r",encoding="utf-8") as f:
        text = f.read()
    return text

def word_to_byte_tuple(token: str):
    """ 单个token转成单个字节表示 """
    encoded = token.encode("utf-8")
    return tuple(BYTE_TOKENS[b] for b in encoded)

def merge_word(word, best_pair):
    """在一个词内部，从左向右执行不重叠合并。"""
    merged_token = best_pair[0] + best_pair[1]
    result = []
    i = 0

    while i < len(word):
        if (
            i + 1 < len(word)
            and word[i] == best_pair[0]
            and word[i + 1] == best_pair[1]
        ):
            result.append(merged_token)
            i += 2
        else:
            result.append(word[i])
            i += 1

    return tuple(result)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size 小于基础字节词表和特殊 token 的总数")

    # 1. 预分词：特殊 token 两侧分别处理
    text = load_file(input_path)

    if special_tokens:
        pattern = "|".join(
            re.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )
        segments = re.split(pattern, text)
    else:
        segments = [text]

    word_counts = Counter()

    for segment in segments:
        for match in compiled_pat.finditer(segment):
            word = word_to_byte_tuple(match.group(0))
            word_counts[word] += 1

    # 给每个初始词分配固定 ID。
    # 合并时只改变 words[word_id]，ID 和词频保持不变。
    words = list(word_counts)
    frequencies = [word_counts[word] for word in words]

    # 2. 只初始化一次全局 pair 计数和倒排索引
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    for word_id, word in enumerate(words):
        freq = frequencies[word_id]

        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)

    merges = []

    # 3. 每轮只更新受影响的词
    for _ in range(vocab_size - len(vocab)):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair),)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # 后面会修改索引，因此先复制受影响的 ID
        affected_ids = tuple(pair_to_words[best_pair])

        for word_id in affected_ids:
            old_word = words[word_id]
            freq = frequencies[word_id]

            new_word = merge_word(old_word, best_pair)

            # 单个词中，各 pair 的出现次数
            old_pairs = Counter(zip(old_word, old_word[1:]))
            new_pairs = Counter(zip(new_word, new_word[1:]))

            # 只将频次差额更新到全局统计
            for pair in old_pairs.keys() | new_pairs.keys():
                delta = new_pairs[pair] - old_pairs[pair]

                if delta:
                    updated_count = pair_counts[pair] + delta * freq

                    if updated_count:
                        pair_counts[pair] = updated_count
                    else:
                        del pair_counts[pair]

            # 这个词不再包含的 pair：移除索引
            for pair in old_pairs.keys() - new_pairs.keys():
                word_ids = pair_to_words[pair]
                word_ids.remove(word_id)

                if not word_ids:
                    del pair_to_words[pair]

            # 这个词新增的 pair：加入索引
            for pair in new_pairs.keys() - old_pairs.keys():
                pair_to_words[pair].add(word_id)

            words[word_id] = new_word

    return vocab, merges