from tokenizers import CharacterTokenizer
import pickle
import torch
from sklearn.model_selection import train_test_split
from models import BigramModel

BLOCK_SIZE = 8
BATCH_SIZE = 32
EVAL_ITERS = 200
MAX_ITERS = 3000
EVAL_INTERVAL = 100

torch.manual_seed(1337)

if torch.cuda.is_available():
    print("GPU is available!")
    DEVICE = torch.device("cuda")
else:
    print("No GPU available. Training will run on CPU.")
    DEVICE = torch.device("cpu")


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


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = dict()
    model.eval()
    for split, data in {'train': train_data, 'val': val_data}.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            logits, loss = model(*get_batch(data))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    text = load_data()
    tokenizer = fit_save_tokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(DEVICE)
    train_data, val_data = train_test_split(data, train_size=0.9)

    model = BigramModel(len(tokenizer.encoder))
    model.to(DEVICE)
    print(tokenizer.decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long).to(DEVICE), max_new_tokens=100)[0].tolist()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for i in range(MAX_ITERS):
        if not i % EVAL_INTERVAL:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {i} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
    print(tokenizer.decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long).to(DEVICE), max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    train()
