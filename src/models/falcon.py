import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import GPTBase

class LayerNorm(nn.Module):
    """ Layer normalization with an optional bias. """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ Causal self-attention layer optimized with FlashAttention for Falcon LLM. """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.register_buffer("mask", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                     .view(1, 1, config.sequence_length, config.sequence_length))
        self.flash_attention = True  # Assuming PyTorch >= 2.0 for FlashAttention

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        if self.flash_attention:
            # Use FlashAttention if available for efficient computation
            y = torch.nn.functional.flash_attention(q, k, v, attn_mask=self.mask)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    """ MLP as used in the Transformer, optimized for Falcon LLM. """

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
    """ A single block of the Transformer optimized for Falcon LLM. """

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

class Falcon(GPTBase):
    """ The full Falcon LLM language model, with a config. """

    def get_parameter_group_specs(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    no_decay.add(fpn)

        # Add 'transformer.wte.weight' to no_decay set if not already present
        if 'transformer.wte.weight' not in no_decay:
            no_decay.add('transformer.wte.weight')

        # Remove 'lm_head.weight' from decay set if present
        decay.discard("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters {} made it into both decay/no_decay sets!".format(inter_params)
        assert len(param_dict.keys() - union_params) == 0, "parameters {} were not separated into either decay/no_decay set!".format(param_dict.keys() - union_params)

        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]


    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.sequence_length = config.sequence_length
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.sequence_length, "Cannot forward, model block size is exhausted."

        token_embeddings = self.wte(idx)
        position_embeddings = self.wpe(torch.arange(0, t, device=idx.device))
        x = self.blocks(token_embeddings + position_embeddings)
        x = self.ln_f(x)
        logits = self.head(x)

        # return logits if no target, else compute the loss
        return logits if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# # Example config class
# class FalconLLMConfig:
#     vocab_size = 50257 # Size of the vocabulary
#     sequence_length = 1024 # Length of the model context
#     n_layer = 12 # Number of transformer blocks
#     n_head = 12 # Number of attention heads
#     n_embd = 768 # Embedding dimension
#     dropout = 0.1 # Dropout rate

# # Example usage
# model_config = FalconLLMConfig()
# model = Falcon(model_config)
