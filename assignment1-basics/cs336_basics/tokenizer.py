import regex as re 
import json
from collections.abc import Iterable, Iterator

class Tokenizer:
    # 分割文本方法
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_pat = re.compile(GPT2_PAT)

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # 复制输入
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        if any(token == "" for token in self.special_tokens):
            raise ValueError("特殊 token 不能为空字符串")

        # 反向vocab，bytes: id
        self.byte_to_id = {token: token_id for token_id, token in self.vocab.items()}

        # 特殊 token 不在词表时，为它分配新 ID。
        next_id = max(self.vocab, default=-1) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.byte_to_id:
                self.vocab[next_id] = token_bytes
                self.byte_to_id[token_bytes] = next_id
                next_id += 1

        # 排名越小，合并优先级越高。
        self.merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, rank)

        # 校验基础字节和合并结果是否存在。
        missing_bytes = [
            b for b in range(256)
            if bytes([b]) not in self.byte_to_id
        ]
        if missing_bytes:
            raise ValueError("vocab 必须包含全部 256 个单字节 token")

        for left, right in self.merges:
            if any(
                token not in self.byte_to_id
                for token in (left, right, left + right)
            ):
                raise ValueError(
                    f"词表缺少合并规则涉及的 token: {(left, right)!r}"
                )

        self.special_token_ids = {
            token: self.byte_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

        # 较长的特殊 token 优先匹配，避免前缀重叠时被提前截断。
        self.special_pat = None
        if self.special_tokens:
            alternatives = sorted(
                self.special_tokens, key=len, reverse=True
            )
            self.special_pat = re.compile(
                "|".join(re.escape(token) for token in alternatives)
            )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """加载 save() 写出的 JSON 文件。"""
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_data = json.load(f)

        with open(merges_filepath, encoding="utf-8") as f:
            merges_data = json.load(f)

        vocab = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in vocab_data.items()
        }
        merges = [
            (bytes.fromhex(left), bytes.fromhex(right))
            for left, right in merges_data
        ]

        return cls(vocab, merges, special_tokens)

    @staticmethod
    def word_to_byte_tuple(token: str) -> tuple[bytes, ...]:
        return tuple(bytes([b]) for b in token.encode("utf-8"))

    @staticmethod
    def merge_word(
        word: tuple[bytes, ...],
        best_pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """从左到右，合并所有不重叠的 best_pair。"""
        result = []
        i = 0

        while i < len(word):
            if (
                i + 1 < len(word)
                and (word[i], word[i + 1]) == best_pair
            ):
                result.append(best_pair[0] + best_pair[1])
                i += 2
            else:
                result.append(word[i])
                i += 1

        return tuple(result)
    
    def _encode_ordinary(self, text: str) -> list[int]:
        """编码不含特殊 token 的普通文本。"""
        encoded_ids = []

        for match in self.compiled_pat.finditer(text):
            word = self.word_to_byte_tuple(match.group(0))

            while len(word) > 1:
                best_pair = min(
                    (
                        pair
                        for pair in zip(word, word[1:])
                        if pair in self.merge_ranks
                    ),
                    key=self.merge_ranks.__getitem__,
                    default=None,
                )

                if best_pair is None:
                    break

                word = self.merge_word(word, best_pair)

            encoded_ids.extend(self.byte_to_id[token] for token in word)

        return encoded_ids

    def encode(self, text: str) -> list[int]:
        if self.special_pat is None:
            return self._encode_ordinary(text)

        encoded_ids = []
        start = 0

        for match in self.special_pat.finditer(text): 
            # 特殊 token 前面的普通文本。
            encoded_ids.extend(
                self._encode_ordinary(text[start:match.start()])
            )

            # 特殊 token 整体对应一个 ID。
            encoded_ids.append(
                self.special_token_ids[match.group(0)]
            )
            start = match.end()

        encoded_ids.extend(self._encode_ordinary(text[start:]))
        return encoded_ids
     
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """逐条编码并惰性输出 ID，不一次性读取整个 iterable。"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")