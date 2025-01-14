from typing import List


class CharacterTokenizer:
    def __init__(self):
        self.encoder = dict()
        self.decoder = dict()

    def fit(self, text: str):
        chars = set(text)
        for i, c in enumerate(chars):
            self.encoder[c] = i
            self.decoder[i] = c

    def encode(self, text: str):
        return [self.encoder[c] for c in text]

    def decode(self, vector: List[int]):
        return "".join([self.decoder[i] for i in vector])


if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    tokenizer.fit("Hello world!")
    x = tokenizer.encode("world!")
    print(x)
    print(tokenizer.decode(x))