import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
import math

class MQA(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, max_seq: int):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim)
        
        # shared k,v across all heads
        self.k_proj = nn.Linear(dim, head_dim)
        self.v_proj = nn.Linear(dim, head_dim)

        # causal mask
        self.mask: torch.Tensor 
        base_mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer(
            "mask",
            base_mask.unsqueeze(0).unsqueeze(0).masked_fill(base_mask == 0, float('-inf'))
        )
 
        self.out_proj = nn.Linear(n_heads * head_dim, dim)
    
    def forward(self, x: Tensor, start_pos: int, k_cache: Optional[Tensor] = None, v_cache: Optional[Tensor] = None):
        batch, T, dim  = x.shape
        
        # q: (batch, n_heads, T, head_dim)
        q = self.q_proj(x).view(batch, T, self.n_heads, self.head_dim).transpose(1,2)        

        # k,v: (batch, 1, T, head_dim)
        k_new = self.k_proj(x).unsqueeze(1)
        v_new = self.v_proj(x).unsqueeze(1)
        
        # kv cache
        if k_cache is not None:
            k_cat = torch.cat([k_cache, k_new], dim=2)
        else:
            k_cat = k_new

        if v_cache is not None:
            v_cat = torch.cat([v_cache, v_new], dim=2)
        else:
            v_cat = v_new

        # attention (batch, n_heads, T, T) w/ causal mask
        prod = torch.matmul(q, k_cat.transpose(-2,-1)) / math.sqrt(self.head_dim)
        causal_mask = self.mask[:, :, :T, :T]
        prod = prod + causal_mask
        attention = torch.softmax(prod, dim = -1)
        
        # context (batch, n_heads, T, head_dim)
        context = torch.matmul(attention, v_cat)
        
        # combine heads
        output = context.transpose(1,2).reshape(batch, T, self.n_heads * self.head_dim)
        output = self.out_proj(output)

        return output, k_cat, v_cat
    
class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, max_seq: int):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim)
        self.k_proj = nn.Linear(dim, n_heads * head_dim)
        self.v_proj = nn.Linear(dim, n_heads * head_dim)
        
        # causal mask
        self.mask: torch.Tensor 
        base_mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer(
            "mask",
            base_mask.unsqueeze(0).unsqueeze(0).masked_fill(base_mask == 0, float('-inf'))
        )
        
        self.out_proj = nn.Linear(n_heads * head_dim, dim)
    
    def forward(self, x: Tensor, start_pos: int, k_cache: Optional[Tensor] = None, v_cache: Optional[Tensor] = None):
        batch, T, dim  = x.shape
        
        # (batch, n_heads, T, head_dim)
        q = self.q_proj(x).view(batch, T, self.n_heads, self.head_dim).transpose(1,2)
        k_new = self.k_proj(x).view(batch, T, self.n_heads, self.head_dim).transpose(1,2)
        v_new = self.v_proj(x).view(batch, T, self.n_heads, self.head_dim).transpose(1,2)
        
        # kv cache
        if k_cache is not None:
            k_cat = torch.cat([k_cache, k_new], dim=2)
        else:
            k_cat = k_new

        if v_cache is not None:
            v_cat = torch.cat([v_cache, v_new], dim=2)
        else:
            v_cat = v_new

        # attention (batch, n_heads, T, T) w/ causal mask
        prod = torch.matmul(q, k_cat.transpose(-2,-1)) / math.sqrt(self.head_dim)
        causal_mask = self.mask[:, :, :T, :T]
        prod = prod + causal_mask
        attention = torch.softmax(prod, dim=-1)
        
        # context (batch, n_heads, T, head_dim)
        context = torch.matmul(attention, v_cat)
        
        # combine heads
        output = context.transpose(1,2).reshape(batch, T, self.n_heads * self.head_dim)
        output = self.out_proj(output)
        return output, k_cat, v_cat

class Transformer(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, p: float, max_seq: int, use_mqa: bool):
        super().__init__()
        if use_mqa:
            self.attention = MQA(dim, n_heads, head_dim, max_seq)
        else:
            self.attention = MHA(dim, n_heads, head_dim, max_seq)
        
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        
        # multilayer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), 
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(4*dim, dim),
            nn.Dropout(p)
        )

    def forward(self, x: Tensor, start_pos: int, k_cache: Optional[Tensor] = None, v_cache: Optional[Tensor] = None):
        # Pre LayerNorm
        out, k_out, v_out = self.attention(self.layernorm1(x), start_pos, k_cache, v_cache)
        x = x + out
        x = x + self.mlp(self.layernorm2(x))
        return x, k_out, v_out

class Model(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_heads: int, n_layers: int, max_seq: int, p: float, use_mqa: bool):
        super().__init__()
        
        assert dim % n_heads == 0, "must be divisible"
        head_dim = dim // n_heads
        
        # token embedding (batch, seq_length or T, dim)
        self.embed = nn.Embedding(vocab_size, dim)

        # positional embedding (1, <=max_seq, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq, dim))
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            Transformer(dim, n_heads, head_dim, p, max_seq, use_mqa)
            for _ in range(n_layers)
        ])
        
        # final layerNorm
        self.layernorm_f = nn.LayerNorm(dim)
        
        # project to vocab
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, start_pos: int, k_cache: Optional[List[Optional[Tensor]]] = None, v_cache: Optional[List[Optional[Tensor]]] = None):
        batch, T = input_ids.shape
    
        # adding positional vector to every sequence in batch (batch, T, dim) + (1, T, dim)
        x = self.embed(input_ids) + self.pos_embed[:, start_pos:start_pos+T, :]

        k_new_list = []
        v_new_list = []

        for i, block in enumerate(self.blocks):
            # kv cache logic
            k_in = k_cache[i] if k_cache is not None else None
            v_in = v_cache[i] if v_cache is not None else None
            
            # run through transformer block
            x, k_out, v_out = block(x, start_pos, k_in, v_in)
            
            k_new_list.append(k_out)
            v_new_list.append(v_out)

        x = self.layernorm_f(x)
        output = self.head(x) # (batch, T, vocab_size)

        return  output, k_new_list, v_new_list
