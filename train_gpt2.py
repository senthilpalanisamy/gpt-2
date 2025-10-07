from tqdm import tqdm
import tiktoken
import time
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path


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

@dataclass
class GPTConfigWordEmbedding:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


def get_batch(split):
    config = GPTConfig()
    block_size = config.block_size

    data = train_data if split=='train' else val_data
    indices = torch.randint(len(data) -block_size-1, (batch_size,))
    X = torch.stack([data[index: index+block_size] for index in indices])
    Y = torch.stack([data[index+1: index+block_size+1] for index in indices])
    x, y = X.to(device), Y.to(device)
    return x, y

FLASH_ATTENTION=1
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.ns = config.n_head
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.project = nn.Linear(config.n_embed, config.n_embed)
        self.project.IS_RESIDUAL_INPUT = True

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.attention(x) # (B, T, 3C)
        q, k, v = qkv.split(C, dim = qkv.dim()-1)
        q = q.view(B, T, self.ns, C // self.ns).transpose(1, 2)
        k = k.view(B, T, self.ns, C // self.ns).transpose(1, 2)
        v = v.view(B, T, self.ns, C // self.ns).transpose(1, 2)
        if FLASH_ATTENTION:
            x = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        else:
            qkT = q@k.transpose(k.dim()-1, k.dim()-2) / (k.size(k.dim()-1) ** 0.5)  # (B, nh, T, Ch) @ (B, nh, Ch, T) -> (B, nh, T, T)
            qkT_masked = qkT.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            weights = F.softmax(qkT_masked, dim=qkT_masked.dim()-1) 
            x = weights@v # (B, nh, T, T) @ (B, nh, T, Ch) -> (B, nh, T, Ch)
        x = x.transpose(1,2).contiguous().view(B, T, C)
        x = self.project(x)
        return x







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
    def __init__(self, config, efficient=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        if not efficient:
            self.mha = MultiHeadAttention(config)
        else:
            self.mha = CausalMultiHeadAttention(config)
        self.mlp = MLP(config)
        self.mlp.IS_RESIDUAL_INPUT = True
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
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'IS_RESIDUAL_INPUT'): 
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * (self.config.n_layer * 2)**(-0.5))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def configure_optimizer(self, weight_decay, learning_rate, device):
        decay_group = [parameter for name, parameter in self.named_parameters() if parameter.dim() >= 2 and parameter.requires_grad]
        non_decay_group = [parameter for name, parameter in self.named_parameters() if parameter.dim() < 2 and parameter.requires_grad]
        optim_groups = [
                {'params': decay_group, 'decay': weight_decay},
                {'params': non_decay_group, 'decay': 0.0}
        ]
        num_decay_params = sum([p.numel() for p in decay_group])
        num_non_decay_params = sum([p.numel() for p in non_decay_group])
        print(f'decay parameters: {num_decay_params}, non_decay_params: {num_non_decay_params}')
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)

@torch.no_grad()
def get_eval_loss(model):
    model.eval()
    losses = torch.zeros(eval_iter)
    for i in range(eval_iter):
        X, Y = get_batch('val')
        logits, loss = model(X, Y)
        losses[i] = loss

    loss = losses.mean()
    model.train()
    return loss


# gpt_config = GPTConfig()
# gpt_config.vocab_size = len(characters)
# model = GPT(gpt_config).to(device) 
# print(sum([parameter.numel() for parameter in model.parameters()]) / 1e6, 'M parameters')
# 
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# for i in tqdm(range(max_iter)):
#     if i % eval_iter == 0 or i == max_iter-1:
#         loss = get_eval_loss(model)
#         print(f'############################')
#         print(f'eval loss is {loss}')
#         print('#############################')
#         print(f'generating')
#         idx = torch.zeros((1,1), dtype = torch.long, device = device)
#         new_tokens = model.generate(idx, max_new_tokens=500)
#         decoded_tokens = decode(new_tokens[0].tolist())
#         print(decoded_tokens)
#     x, y = get_batch('train')
#     logits, loss = model(x, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f'training loss is {loss}')

class DataLoaderManual:
    def __init__(self, B, T, local_rank, world_size, filename='input.txt'):
        with open(filename) as file:
            data = file.read()
        tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokenizer.encode(data))
        self.B = B
        self.T = T
        self.local_rank = local_rank
        self.world_size = world_size
        self.pointer = self.local_rank * self.B * self.T

    def next_batch(self):
        next_tokens = self.tokens[self.pointer:self.pointer+self.B * self.T +1]
        next_x = next_tokens[:-1]
        next_y = next_tokens[1:]
        self.pointer = self.pointer + self.B * self.T * self.world_size

        if self.pointer + self.B * self.T +1 > len(self.tokens):
            self.pointer = self.local_rank * self.B * self.T
        # print(f'{self.pointer}, {len(self.tokens)}')
        return next_x.view(self.B, self.T), next_y.view(self.B, self.T)

def get_token_count_in_shard(shard_path):
    with open(shard_path, 'rb') as file:
        buffer = file.read(256 * 4)
        header_info = np.frombuffer(buffer,  dtype=np.int32)
        assert header_info[0] == 20240520, 'header info magic number mismatch'
        assert header_info[1] == 1, 'version mismatch'
        ntok = header_info[2]
        tokens = np.frombuffer(file.read(), dtype=np.int16)
        assert ntok == len(tokens), 'header token count info and number of tokens mismatch'
    return ntok

def load_shard(shard_path):
    with open(shard_path, 'rb') as file:
        buffer = file.read(256 * 4)
        header_info = np.frombuffer(buffer,  dtype=np.int32)
        assert header_info[0] == 20240520, 'header info magic number mismatch'
        assert header_info[1] == 1, 'version mismatch'
        ntok = header_info[2]
        tokens = np.frombuffer(file.read(), dtype=np.int16)
        assert ntok == len(tokens), 'header token count info and number of tokens mismatch'
    return tokens

class DataLoaderFineWeb:
    def __init__(self, B, T, local_rank, world_size, split):
        self.B = B
        self.T = T
        self.local_rank = local_rank
        self.world_size = world_size
        dataset_basepath = 'fineweb10B'
        shards = os.listdir(dataset_basepath)
        filtered_sorted_shards = [s for s in sorted(shards) if split in s]
        self.shard_paths = [os.path.join(dataset_basepath, shard_name) for shard_name in filtered_sorted_shards]
        self.shard_pointer = None
        self.tokens = None
        self.current_position = None

    def reset(self):
        if self.shard_pointer != 0:
            self.shard_pointer = 0
            self.tokens = load_shard(self.shard_paths[self.shard_pointer])
        self.current_position = self.local_rank * B * T

    def advance(self):
        self.shard_pointer = (self.shard_pointer + 1) % len(self.shard_paths)
        self.tokens = load_shard(self.shard_paths[self.shard_pointer])
        self.current_position = self.local_rank * B * T

    def next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.current_position: self.current_position + B * T + 1]
        buff = torch.tensor(buff.astype(np.int32), dtype = torch.long)
        x = buff[:-1]
        y = buff[1:]
        self.current_position += B * T * self.world_size
        if self.current_position + B * T + 1 > len(self.tokens):
            self.advance()
        return x.view(B, T), y.view(B, T)
        


def get_lr(step,  max_lr=6e-4, warmup_steps=10, max_steps=1000):
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return (step+1) / warmup_steps * max_lr

    if step > max_steps:
        return min_lr

    decay_weight = (step+1) / (max_steps-warmup_steps)
    cosine_weight = 0.5 * (1+math.cos(decay_weight * math.pi))
    return min_lr + (max_lr - min_lr) * cosine_weight


            

        

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--dataset_type', type=str, default='fineweb')
    parser.add_argument('--val_frequency', type=int, default=100)
    parser.add_argument('--max_val_steps', type=int, default=20)
    parser.add_argument('--sample_frequency', type=int, default=200)
    parser.add_argument('--save_frequency', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=10000)
    args = parser.parse_args()

    best_val_loss = float('inf')

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), 'cuda is needed for ddp'
        rank = os.environ.get('RANK')
        init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        world_size = int(os.environ.get('WORLD_SIZE'))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master = local_rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master = True

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    #model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfigWordEmbedding())
    model.eval()
    model.to('cuda')

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    batch_size = 16
    max_length = 55
    B = batch_size
    T = 32
    enc = tiktoken.get_encoding('gpt2')
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
   

    # with open('input.txt') as file:
    #     data = file.read()
    # sliced_data = data[:1000]
    # tokens = enc.encode(sliced_data)
    # sliced_tokens = torch.tensor(tokens[:B*T+1])
    # x = sliced_tokens[:-1].view(B, T).to(device)
    # y = sliced_tokens[1:].view(B,T).to(device)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device = device)
    if args.dataset_type == 'fineweb':
        train_dataloader = DataLoaderFineWeb(B = B, T = T, local_rank = local_rank, world_size = world_size, split='train')
        val_dataloader = DataLoaderFineWeb(B = B, T = T, local_rank = local_rank, world_size = world_size, split='val')
        train_dataloader.reset()
    else:
        # shakespeare dataset
        # val_dataloader doesn't have a reset, might not work fully well
        train_dataloader = DataLoaderManual(B = B, T = T, local_rank = local_rank, world_size = world_size)
        val_dataloader = DataLoaderManual(B = B, T = T, local_rank = local_rank, world_size = world_size)
        
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
    max_steps = 10000
    total_batch_size = 524288
    grad_accum_steps = total_batch_size // (B * T * world_size)
    base_dir = f'runs/{args.experiment_name}'
    model_dir = f'{base_dir}/models'
    
    if master:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(base_dir)

    for i in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0

        if args.val_frequency > 0 and ((i % args.val_frequency == 0) or (i == max_steps-1)):
            model.eval()
            val_dataloader.reset()
            total_val_loss = 0.0
            with torch.no_grad():
                for _ in range(args.max_val_steps):
                    x, y = val_dataloader.next_batch()
                    x = x.to(device)
                    y = y.to(device)
                    logits, loss = model(x, y)
                    total_val_loss += loss.item()
                total_val_loss = total_val_loss / args.max_val_steps
                if master:
                    writer.add_scalar('loss/val', total_val_loss, i)
                    if total_val_loss < best_val_loss:
                        best_val_loss  = total_val_loss
                        torch.save(raw_model, f'{model_dir}/best_model.pth')
                    print(f'#############val loss:{total_val_loss}')


        if args.sample_frequency > 0 and ((i % args.sample_frequency == 0) or (i==max_steps-1)) and master:
            model.eval()
            tokens = enc.encode('Hello! I am a language model')
            tokens = torch.tensor(tokens, dtype=torch.long) # T
            x = tokens.unsqueeze(0).repeat(batch_size, 1).to(device) # (B, T)

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


        model.train()
        for j in range(grad_accum_steps):
            x, y = train_dataloader.next_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type = device, dtype = torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            if ddp:
                model.require_backward_grad_sync = (j == grad_accum_steps-1)
            total_loss += loss.detach()

        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)

        lr = get_lr(i, warmup_steps= args.warmup_steps, max_steps=args.max_steps)
        norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = B * T  * grad_accum_steps * world_size / (t1-t0)
        if master:
            print(f'loss: {total_loss.item()}')
            print(f'time: {(t1 - t0)*1000}')
            print(f'tokens per second: {tokens_per_second}')
            print(f'graident norm: {norm}')
            writer.add_scalar('loss/train', total_loss, i)
            writer.add_scalar('learning_rate', lr, i)
            writer.add_scalar('grad_norm', norm, i)

        if args.save_frequency > 0 and ((i % args.save_frequency == 0) or (i == max_steps-1)) and master:
            torch.save(raw_model, f'{model_dir}/model_{i}.pth')
    if ddp:
        destroy_process_group()
        










