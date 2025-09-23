from torch.nn import nn
from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384

class GPT(nn.module): 
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.Layernorm(config.n_embed)
            ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)





