from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


from transformers import GPT2LMHeadModel

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


FLASH = 0
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768




class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-al']

        config_args = {
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
            }
        config = GPTConfig(**config_args[model_type])
        model = GPT(config)
        model_sd_dict = model.state_dict()
        model_sd_keys = [key for key in model_sd_dict.keys() if not key.endswith('.attn.bias')]

        # hugging face model (openai replica)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        model_hf_dict = model_hf.state_dict()
        model_hf_keys = [key for key in model_hf_dict.keys() if not key.endswith(('.attn.masked_bias', '.attn.bias'))]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(model_sd_keys) == len(model_hf_keys), 'keys mismatch in state dict, length not  matching'

        for key in model_hf_keys:
            if any(key.endswith(w) for w in transposed):
                assert model_hf_dict[key].shape[::-1] == model_sd_dict[key].shape, f'key{key} not matching in dimesions'
                with torch.no_grad():
                    model_sd_dict[key].copy_(model_hf_dict[key].t())
            else:
                assert model_hf_dict[key].shape == model_sd_dict[key].shape, f'key {key} not matching in dimensions'
                with torch.no_grad():
                    model_sd_dict[key].copy_(model_hf_dict[key])
        return model

class DataLoaderManual:
    def __init__(self, B, T, filename='input.txt'):
        with open(filename) as file:
            data = file.read()
        tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokenizer.encode(data))
        self.B = B
        self.T = T
        self.pointer = 0

    def next_batch(self):
        next_tokens = self.tokens[self.pointer:self.pointer+self.B * self.T +1]
        next_x = next_tokens[:-1]
        next_y = next_tokens[1:]

        if self.pointer + self.B * self.T +1 < len(self.tokens):
            self.pointer = self.pointer + self.B * self.T
        else:
            self.pointer = 0
        return x, y


            

        

if __name__=='__main__':
    #model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to('cuda')
    batch_size = 4
    max_length = 55
    B = batch_size
    T = 32
    enc = tiktoken.get_encoding('gpt2')
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
   

    with open('input.txt') as file:
        data = file.read()
    sliced_data = data[:1000]
    tokens = enc.encode(sliced_data)
    sliced_tokens = torch.tensor(tokens[:B*T+1])
    x = sliced_tokens[:-1].view(B, T).to(device)
    y = sliced_tokens[1:].view(B,T).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dataloader = DataLoaderManual(B = B, T = T)

    for i in range(50):
        x, y = dataloader.next_batch()
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}')
        

    tokens = enc.encode('Hello! I am a language model')
    tokens = torch.tensor(tokens, dtype=torch.long) # T
    x = tokens.unsqueeze(0).repeat(batch_size, 1).to(device) # (B, T)


    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    with torch.no_grad():
        while x.size(1) < max_length:
            logits, _ = model(x) # (B, T, V)
            B, T, V = logits.shape
            logits = logits[:, T-1, :] # (B, V) 
            probs = F.softmax(logits, dim=1) # (B, V)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=1) # (B, 50)
            sampled_index = torch.multinomial(topk_probs, 1) # (B, 1)
            pred_x = torch.gather(topk_indices, 1, sampled_index)
            x = torch.cat((x, pred_x), dim=1)


        for i in range(batch_size):
            decoded = enc.decode(x[i,:].tolist())
            print('#############################')
            print(decoded)




