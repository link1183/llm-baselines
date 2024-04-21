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
        #self.flash_attention = False  # Assuming PyTorch >= 2.0 for FlashAttention
        self.flash_attention = False

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
    
    # def _get_parameter_group_specs(self):
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear,)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    #     # Iterate over all modules and their parameters
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = f"{mn}.{pn}" if mn else pn
    #             if isinstance(m, blacklist_weight_modules) or pn.endswith("bias"):
    #                 # Add parameters of blacklist modules and biases to no_decay set
    #                 no_decay.add(fpn)
    #             elif isinstance(m, whitelist_weight_modules) and pn.endswith("weight"):
    #                 # Add weights of whitelist modules to decay set
    #                 decay.add(fpn)

    #     # Manually add layer normalization weights and transformer.wte.weight to no_decay set
    #     no_decay.update({
    #         'transformer.wte.weight',
    #         'ln_f.weight',
    #         *{f'blocks.{i}.ln1.weight' for i in range(12)},
    #         *{f'blocks.{i}.ln2.weight' for i in range(12)}
    #     })
    #         # Manually add the problematic layer normalization weights to no_decay set
    #     no_decay.update({
    #         'transformer.h.5.ln_1.weight', 'transformer.h.9.ln_2.weight', 'transformer.h.3.ln_2.weight',
    #         'transformer.h.4.ln_1.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.1.ln_1.weight',
    #         'transformer.h.2.ln_2.weight', 'transformer.h.7.ln_2.weight', 'transformer.h.10.ln_1.weight',
    #         'transformer.h.3.ln_1.weight', 'transformer.h.11.ln_1.weight', 'transformer.h.0.ln_2.weight',
    #         'transformer.ln_f.weight', 'transformer.h.0.ln_1.weight', 'transformer.h.7.ln_1.weight',
    #         'transformer.h.10.ln_2.weight', 'transformer.h.6.ln_2.weight', 'transformer.h.11.ln_2.weight',
    #         'transformer.h.8.ln_2.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.5.ln_2.weight',
    #         'transformer.h.2.ln_1.weight', 'transformer.h.6.ln_1.weight', 'transformer.h.9.ln_1.weight',
    #         'transformer.h.8.ln_1.weight'
    #     })

    #     # Ensure 'lm_head.weight' is handled correctly if it's tied to 'transformer.wte.weight'
    #     decay.discard('lm_head.weight')

    #     # Validate that all parameters are in one of the sets
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert len(inter_params) == 0, "parameters {} made it into both decay/no_decay sets!".format(inter_params)
    #     assert len(param_dict.keys() - union_params) == 0, "parameters {} were not separated into either decay/no_decay set!".format(param_dict.keys() - union_params)

    #     # Create the optimizer parameter groups
    #     return [
    #         {"params": sorted(list(decay)), "weight_decay": self.config.weight_decay},
    #         {"params": sorted(list(no_decay)), "weight_decay": 0.0},
    #     ]



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
        #return logits if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Calculate loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Reduce the loss to a scalar
            loss = loss.mean()  # Ensure the loss is a scalar
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}

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
