from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
dropout = 0.2

with open('input.txt', 'r') as f:
    input_data = f.read()

characters = sorted(list(set(input_data)))
stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i: ch for i, ch in enumerate(characters)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: ''.join([itos[i] for i in s])
data = torch.tensor(encode(input_data), dtype=torch.long)

train_data = data[: int(0.9 * len(data))]
val_data = data[int(0.9) * len(data):]

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
max_iter=5000
eval_iter=200

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384


def get_batch(split):
    config = GPTConfig()
    block_size = config.block_size

    data = train_data if split=='train' else val_data
    indices = torch.randint(len(data) -block_size-1, (batch_size,))
    X = torch.stack([data[index: index+block_size] for index in indices])
    Y = torch.stack([data[index+1: index+block_size+1] for index in indices])
    x, y = X.to(device), Y.to(device)
    return x, y








class CausalAttention(nn.Module):

    def __init__(self, config, head_size):
        super().__init__()
        self.q = nn.Linear(config.n_embed, head_size, bias=False)
        self.k = nn.Linear(config.n_embed, head_size, bias=False)
        self.v = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x) # (B, T, h)
        k = self.k(x) # (B, T, h)
        v = self.v(x) # (B, T, h)
        qkT = q@k.transpose(k.dim()-2, k.dim()-1) / (T ** 0.5) # (B, T, T)
        qkT_masked = qkT.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        qkT_causal = self.dropout(F.softmax(qkT_masked, dim = qkT_masked.dim()-1)) # (B, T, T)
        return qkT_causal@v # (B, T, h)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.heads = nn.ModuleList([CausalAttention(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim = x.dim()-1)
        return self.dropout(self.proj(x))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embed, config.n_embed * 4)
        self.l2 = nn.Linear(config.n_embed*4, config.n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.dropout(self.l2(x))
        return x
        


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.mha = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
            ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.nomal_(module.weight, mean=0.0, std=0.2)

    def forward(self, x, targets=None):
        # x (B, T)
        B, T = x.shape
        device = x.device
        token_embedding = self.transformer.wte(x) # (B, T, C)
        position_embedding = self.transformer.wpe(torch.arange(T, device=device)) # (B, T, C)
        x = token_embedding + position_embedding # (B, T, C)
        for h in self.transformer.h:
            x = h(x)
        x = self.transformer.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
             B, T = idx.shape
             idx_cut = idx[:, T-self.config.block_size:] 
             logits, loss = self(idx_cut)
             B, T, V = logits.shape 
             logits = logits[:, T-1, :] # (B, V)
             probs = F.softmax(logits, dim=logits.dim()-1)
             idx_next = torch.multinomial(probs, num_samples=1)
             idx = torch.cat([idx, idx_next], dim=idx.dim()-1)
        return idx

@torch.no_grad()
def get_eval_loss(model):
    model.eval()
    losses = torch.zeros(eval_iter)
    for i in range(eval_iter):
        X, Y = get_batch('val')
        logits, loss = model(X, Y)
        losses[i] = loss

    losses.mean()
    model.train()


gpt_config = GPTConfig()
gpt_config.vocab_size = len(characters)
model = GPT(gpt_config).to(device) 
print(sum([parameter.numel() for parameter in model.parameters()]) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for i in tqdm(range(max_iter)):
    if i % eval_iter == 0 or i == max_iter-1:
        loss = get_eval_loss(model)
        print(f'############################')
        print(f'eval loss is {loss}')
        print('#############################')
        print(f'generating')
        idx = torch.zeros((1,1), dtype = torch.long, device = device)
        new_tokens = model.generate(idx, max_new_tokens=500)
        decoded_tokens = decode(new_tokens[0].tolist())
        print(decoded_tokens)
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'training loss is {loss}')






