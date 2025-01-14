from tokenizers import CharacterTokenizer
import pickle
import torch
from sklearn.model_selection import train_test_split
from models import BigramModel

BLOCK_SIZE = 8
BATCH_SIZE = 4

torch.manual_seed(1337)


def load_data():
    with open("data/input.txt", 'r') as file:
        return file.read()


def fit_save_tokenizer(text: str) -> CharacterTokenizer:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)

    with open("binaries/character_tokenizer.pkl", 'wb') as file:
        pickle.dump(tokenizer, file)

    return tokenizer


def get_batch(data):
    x = list()
    y = list()
    for i in torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)):
        x.append(data[i:i + BLOCK_SIZE])
        y.append(data[i + 1:i + BLOCK_SIZE + 1])
    return torch.stack(x), torch.stack(y)


def train():
    text = load_data()
    tokenizer = fit_save_tokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, test_data = train_test_split(data, train_size=0.9)

    xb, yb = get_batch(train_data)
    print("inputs:")
    print(xb.shape)
    print(xb)
    print("targets:")
    print(yb.shape)
    print(yb)
    bigram_model = BigramModel(len(tokenizer.encoder))
    out = bigram_model(xb, yb)
    print(out.shape)


if __name__ == "__main__":
    train()
