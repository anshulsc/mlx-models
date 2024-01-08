import math
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

@dataclass
class ModelArgs:
    dim: int 
    n_layers: int
    head_dim: int 
    hidden_dim: int 
    n_heads: int 
    n_kv_heads: int 
    norm_eps: float
    vocab_size: int 


class LoRALinear(nn.Module):

    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8): 

        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin
    
    def __init__(
            self, input_dims: int, output_dims: int, lora_rank: int = 8, bias: bool = False
    ):
        super().__init__()

        scale =  1 / math.sqrt(input_dims)
        self.linear = nn.Linear(input_dims,output_dims, bias = bias)
        self.lora_a = mx.random.uniform(
            low = -scale,
            high = scale,
            shape = (input_dims, lora_rank)
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))


    def __call__(self, x: mx.array):
        dtype = self.linear.weight.dtype

        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + 2.0 * z 
    






class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output

class Attention(nn.Module):

    def __init__(self, args: ModelArgs):

        super().__init__()

        # Indicates the number of heads for the keys and values
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        # Indicates the number of heads for queries
        self.n_heads_q =  args.n_heads
        # Indicates how many times the heads of keys and Values should be repeated to match the number of heads for queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indiactes the dimension of the embedding
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(args.head_dim, traditional=True)
    
    def __call__(self, x: mx.array,
            mask : Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,):
        
        batch_size, seq_len, dim = x.shape # (B,1,dim)

        # (B,1,dim) -> (B, 1, H_Q * Head_dim)
        xq = self.wq(x)
        # (B,1,dim) -> (B, 1, H_KV * Head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, 1, H_Q * Head_dim) -> (B, 1, H_Q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_dim) -> (B, 1, H_KV, Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Replace the entry in the cache with the new keys and values
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(xq,offset=key_cache.shape[2])
            keys = self.rope(xk,offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, xv], axis=2)
        else:
            # (B, 1, H_Q, Head_dim) -> (B, 1, H_Q, Head_dim)
            queries = self.rope(xq)
            # (B, 1, H_KV, Head_dim) -> (B, 1, H_KV, Head_dim)
            keys = self.rope(xk)
            values = xv

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.n_rep, axis=2)
            return a
        
        # Reorder the keys and values to match the number of heads for queries
        # (B, 1, H_KV, Head_dim) -> (B, 1, H_Q, Head_dim)
        keys = repeat(keys)
        values = repeat(values)

        # (B, H_Q, Seq_len_Q, Head_dim) -> (B, Seq_len_Q, H_Q, Head_dim)
        queries = queries.transpose(0,2,1,3)
        keys = keys.transpose(0,2,1,3)
        values = values.transpose(0,2,1,3)

        # (B, H_Q, 1, Head_dim) * (B, H_KV, Head_dim, Seq_len_KV) -> (B, H_Q, 1, Seq_len_KV)
        scores = (queries  @ keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
         # (B, H_Q, 1, Seq_len_KV) * (B, H_Q , Seq_len_KV, Head_dim) -> (B, H_Q, 1, Head_dim)
        output = (scores @ values).transpose(0,2,1,3).reshape(batch_size, seq_len, -1)
        return self.wo(output), (keys, values)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__() 

        self.w1 = nn.Linear(args.dim, args.hidden_dim)
        self.w2 = nn.Linear(args.hidden_dim, args.dim)
        self.w3 = nn.Linear(args.dim, args.hidden_dim)   

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.args = args

    def __call__(
            self,
            x: mx.array,
            mask : Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r 
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache
    


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    
    def __call__(
            self,
            inputs: mx.array,
            cache = None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
           h, cache[e] = layer(h,mask, cache[e])

        return self.output(self.norm(h)), cache
                 
