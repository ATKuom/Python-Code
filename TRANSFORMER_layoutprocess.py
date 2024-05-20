import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import config, copy
import matplotlib.pyplot as plt
from split_functions import string_to_equipment, token_to_string

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 22  # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 200
learning_rate = 5e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = max_iters // eval_interval
n_embd = 256  # 32
n_head = 4  # 4
n_layer = 2  # 2
dropout = 0.1  # 0.1
chars = classes
vocab_size = len(chars)
# breakpoint()


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    # breakpoint()
    return x, y


def get_batch2(split, batch_size, batch_start):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    x = data[batch_start : batch_start + batch_size]
    x = x[:, :-1]
    y = data[batch_start : batch_start + batch_size]
    y = y[:, 1:]
    x, y = x.to(device), y.to(device)
    # breakpoint()
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    accuracies = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                X, Y = get_batch2(
                    split,
                    batch_size,
                    np.random.randint(0, len(train_data) - batch_size),
                )
            else:
                X, Y = get_batch2(
                    split, batch_size, np.random.randint(0, len(val_data) - batch_size)
                )
            logits, loss = model(X, Y)
            correct = (logits.argmax(axis=1).reshape(Y.shape) == Y).sum().item()
            total = Y.numel()
            acc = correct / total
            accs[k] = acc
            losses[k] = loss.item()
        accuracies[split] = accs.mean()
        out[split] = losses.mean()
    model.train()
    return out, accuracies


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.reshape(-1)
            # targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            ##greedy search
            # idx_next = probs.topk(1)[1]
            ##sampling
            # idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            ##topk 5
            # topkk = probs.topk(5)
            # idx_next = topkk[1][0][torch.multinomial(topkk[0], num_samples=1)]
            ##topp 0.9
            k = 1
            topp = probs.topk(k)
            total_prob = topp[0].sum()
            while total_prob < 0.9:
                k += 1
                topp = probs.topk(k)
                total_prob = topp[0].sum()
            idx_next = topp[1][0][torch.multinomial(topp[0] / total_prob, 1)]
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            if idx_next.item() == len(classes) - 1:
                break
        return idx


model = GPTLanguageModel()


if __name__ == "__main__":
    text = np.load(config.DATA_DIRECTORY / "v21D10_m1.npy", allow_pickle=True)
    equipment_datalist = string_to_equipment(text, classes)

    for layout in equipment_datalist:
        layout.extend([11] * (block_size - len(layout)))

    data2 = torch.tensor(equipment_datalist, dtype=torch.long)
    # Train and test splits
    n = int(0.85 * len(data2))
    train_data = data2[:n]
    val_data = data2[n:]
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e3, "k parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_model = None
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    batch_start = 0
    epoch = 0
    early_stopping = 0
    indices = np.arange(len(train_data))
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses, accuracies = estimate_loss()
            train_accuracies.append(accuracies["train"])
            val_accuracies.append(accuracies["val"])
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            # breakpoint()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                f", train accuracy {accuracies['train']:.2f}, val accuracy {accuracies['val']:.2f}"
            )
            if losses["val"] < best_loss:
                best_loss = losses["val"].item()
                best_acc = accuracies["val"].item()
                best_model = copy.deepcopy(model.state_dict())
                best_iter = iter
                print("New best model found", best_iter)
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping > 10:
                    print("Early stopping")
                    break

        # sample a batch of data
        xb2, yb2 = get_batch2("train", batch_size, batch_start)

        batch_start += batch_size
        if batch_start > (len(train_data) - batch_size):
            batch_start = 0
            epoch += 1
            np.random.shuffle(indices)
            train_data = train_data[indices]

        # evaluate the loss
        logits, loss = model(xb2, yb2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(epoch)
    print("Best model found in", best_iter)
    iteration = np.arange(0, iter + 1, eval_interval)
    plt.plot(iteration, train_losses, label="Training")
    plt.plot(iteration, val_losses, label="Validation")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(iteration, train_accuracies, label="Trainining")
    plt.plot(iteration, val_accuracies, label="Validation")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    torch.save(best_model, config.MODEL_DIRECTORY / "transformer_trial_bestmodel.pt")
    torch.save(
        model.state_dict(), config.MODEL_DIRECTORY / "transformer_trial_lastmodel.pt"
    )
