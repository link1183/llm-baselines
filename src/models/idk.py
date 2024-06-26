import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ Layer normalization with an optional bias. """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ Causal self-attention layer. """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    """ MLP as used in the Transformer. """

    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act_fn(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """ A single block of the Transformer. """

    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """ The full GPT language model, with a config. """

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.wte(idx)
        position_embeddings = self.wpe(torch.arange(0, t, device=idx.device))
        x = self.blocks(token_embeddings + position_embeddings)
        x = self.ln_f(x)
        logits = self.head(x)

        # return logits if no target, else compute the loss
        return logits if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Example config class
class GPTConfig:
    vocab_size = 50257 # Size of the vocabulary
    block_size = 1024 # Length of the model context
    n_layer = 12 # Number of transformer blocks
    n_head = 12 # Number of attention heads
    n_embd = 768 # Embedding dimension
    dropout = 0.1 # Dropout rate

# # Example usage
# model_config = GPTConfig()
# model = GPT(model_config)
