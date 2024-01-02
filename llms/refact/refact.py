import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer


def upcast_masked_softmax(
        x: mx.array, mask: mx.array, mask_value: mx.array, softmax_dtype
):
    input_dtype = x.dtype
    x = x.astype(softmax_dtype)
    x = mx.where(mask, x, mask_value)
    x = mx.softmax(x, axis=-1).astype(input_dtype)
    return x


def upcast_softmax(x: mx.array, softmax_dtype):
    input_dtype = x.dtype
    x = x.astype(softmax_dtype)
    x = mx.softmax(x, dim=-1).astype(input_dtype)
    return x


def _get_slopes(attn_heads: int) -> mx.array:
    """
    ## Get head-specific slope $m$ for each head
    * `n_heads` is the number of heads in the attention layer $n$
    The slope for first head is
    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$
    The slopes for the rest of the heads are in a geometric series with a ratio same as above.
    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log(attn_heads, 2))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    arange_sequence = mlx.core.arange(1, 1 + n)  # Similar to torch.arange
    m = mlx.core.power(m_0, arange_sequence)  # Similar to torch.pow

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < attn_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        # 假设 m_hat_0 是已定义的数组，attn_heads 和 n 是整数
        arange_sequence = mlx.core.arange(1, 1 + 2 * (attn_heads - n), 2)  # 类似于 torch.arange
        m_hat = mlx.core.power(m_hat_0, arange_sequence)  # 类似于 torch.pow
        # Concatenate the slopes with the remaining slopes.
        m = mlx.core.concatenate([m, m_hat])
    return m


def get_alibi_biases(
        B: int,
        T: int,
        attn_heads: int,
        dtype: mx.Dtype) -> mx.array:
    """
    ## Calculate the attention biases matrix
    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`
    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """

    # Get slopes $m$ for each head
    mask = mlx.core.ones((T, T), dtype=mlx.core.bool_)

    m = _get_slopes(attn_heads).astype(dtype)

    # Calculate distances $[0, 1, \dots, N]$
    # Here we calculate the distances using the mask.
    #
    # Since it's causal mask we can just use $[0, 1, \dots, N]$ too.
    # `distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[None, :]`
    distance = mask.cumsum(axis=-1).astype(dtype)

    # Multiply them pair-wise to get the AliBi bias matrix
    biases = distance[:, :, None] * m[None, None, :]
    biases_transposed = biases.transpose(2, 0, 1)  # Rearrange dimensions
    biases = biases_transposed[None, :, :T, :T]  # Increase new dimensions and perform indexing.
    return biases


@dataclass
class ModelArgs:
    vocab_size: int = 49216
    n_positions: int = 4096
    n_embd: int = 2048
    hidden_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    attention_softmax_in_fp32: bool = True
    attention_bias_in_fp32: bool = True
    scale_attention_softmax_in_fp32: bool = True
    multi_query: bool = True
    n_inner = None
    layer_norm_epsilon: float = 1e-5
    num_attention_heads: int = 32
    use_cache: bool = True


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        self.mask_value = None
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_attn_heads = 1

        self.scale_factor = self.head_dim ** -0.5

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
                config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )
        self.attention_bias_in_fp32 = config.attention_bias_in_fp32

        self.q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.kv = nn.Linear(self.embed_dim, self.head_dim * 2, bias=False)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def _get_mask_value(self, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype:
            # Translate from: self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
            # It seems no dtype.min in mlx, so hard code here, -3.4028235e+38 for float32
            self.mask_value = mlx.core.full([], -3.4028235e+38, dtype=dtype)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, alibi=None):
        dtype = query.dtype
        softmax_dtype = mlx.core.float32 if self.attention_softmax_in_fp32 else dtype
        mask_value = self._get_mask_value(softmax_dtype)
        upcast = dtype != softmax_dtype

        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.shape[-1]

        # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
        # -> (batch_size, query_length, num_heads, key_length)
        query_length = query_shape[1]
        attn_shape = (batch_size, query_length, self.num_heads, key_length)
        attn_view = (batch_size, query_length * self.num_heads, key_length)
        # No copy needed for MQA 2, or when layer_past is provided.
        query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)

        alibi = alibi.transpose(0, 2, 1, 3).reshape(alibi.shape[0], -1, alibi.shape[-1])
        initial_dtype = query.dtype
        new_dtype = mlx.core.float32 if self.attention_bias_in_fp32 else initial_dtype
        # It seems no baddbmm operator in mlx, so use translated method of the following code:
        # attn_weights = alibi.baddbmm(
        #     batch1=query.astype(new_dtype),
        #     batch2=key.astype(new_dtype),
        #     beta=1,
        #     alpha=self.scale_factor
        # ).view(attn_shape).astype(initial_dtype)
        query_new = query.astype(new_dtype)
        key_new = key.astype(new_dtype)
        result = mlx.core.matmul(query_new, key_new) * self.scale_factor + alibi
        result_reshaped = result.reshape(attn_shape)
        attn_weights = result_reshaped.astype(initial_dtype)

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, softmax_dtype)
            else:
                attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, softmax_dtype)
        else:
            if attention_mask is not None:
                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = mx.where(attention_mask, attn_weights, mask_value)
            attn_weights = mx.softmax(attn_weights, axis=-1)

        attn_output = mx.matmul(attn_weights.reshape(attn_view), value).reshape(query_shape)

        return attn_output, attn_weights

    def __call__(self, hidden_states, use_cache=True, attention_mask=None, alibi=None, layer_past=None):
        query = self.q(hidden_states)
        kv = self.kv(hidden_states)
        key, value = mx.split(kv, 2, axis=-1)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = mx.concatenate([past_key, key], axis=-2)
            value = mx.concatenate([past_value, value], axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key.transpose(0, 2, 1), value, attention_mask, alibi)
        attn_output = self.c_proj(attn_output)

        return attn_output, present


class MLP(nn.Module):
    def __init__(self, intermediate_size, config: ModelArgs, multiple_of: int = 256):
        super().__init__()

        embed_dim = config.hidden_size
        hidden_dim = intermediate_size
        hidden_dim = int(2 * hidden_dim / 3)
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_up_proj = nn.Linear(embed_dim, self.hidden_dim * 2, bias=False)
        self.c_proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)

    def __call__(self, x):
        up_proj = self.gate_up_proj(x)
        # shape [1, 9, 11264] split into two [1, 9, 5632]
        split_arrays = mlx.core.split(up_proj, 2, axis=-1)
        x1 = split_arrays[0]
        x2 = split_arrays[1]
        x = self.c_proj(nn.silu(x1) * x2)
        return x


class LayerNormNoBias(nn.Module):
    def __init__(self, shape: int, eps: float = 1e-5):
        super().__init__()
        self.shape = (shape,)
        self.eps = eps
        self.weight = mx.full(shape, 0)

    def __call__(self, x, debug=False):
        means = mx.mean(x, axis=-1, keepdims=True)
        # seems mlx has bug with var function, some case here get inf,
        # simply change data type to float32 can resolve this issue
        var = mx.var(x.astype(mx.float32), axis=-1, keepdims=True, ddof=0).astype(mx.float16)
        x = (x - means) / mx.sqrt(var + self.eps)
        return self.weight * x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNormNoBias(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNormNoBias(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(self.inner_dim, config)

    def __call__(self, hidden_states, use_cache=True, attention_mask=None, alibi=None, layer_past=None):
        hidden_states_norm = self.ln_1(hidden_states)
        attn_output, cache = self.attn(hidden_states_norm,
                                       use_cache=use_cache,
                                       attention_mask=attention_mask,
                                       alibi=alibi,
                                       layer_past=layer_past)
        # residual connection
        mix = attn_output + hidden_states
        norm_mix = self.ln_2(mix)
        feed_forward_hidden_states = self.mlp(norm_mix)
        # residual connection
        hidden_states = mix + feed_forward_hidden_states

        return hidden_states, cache


# Refact class is adopted from https://huggingface.co/smallcloudai/Refact-1_6B-fim/blob/main/modeling_gpt_refact.py
class Refact(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.multi_query = config.multi_query
        self.max_positions = config.n_positions  # n_positions equals to max_position_embeddings
        self.attention_bias_in_fp32 = config.scale_attention_softmax_in_fp32

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = [TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)]

        # Translated from this code:
        # self.register_buffer(
        #     "bias", torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool)),
        #     persistent=False
        # )
        ones_array = mlx.core.full([self.max_positions, self.max_positions], True, dtype=mx.bool_)
        self.bias = mlx.core.tril(ones_array)

        # Merged from GPTRefactForCausalLM class
        self.ln_f = LayerNormNoBias(self.embed_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, inputs, use_cache=True, past_key_values=None):
        input_shape = inputs.shape
        inputs = inputs.reshape(-1, input_shape[-1])
        batch_size = inputs.shape[0]

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].shape[-2]

        query_length = inputs.shape[-1]
        seq_length_with_past = past_length + query_length

        key_length = past_length + query_length
        self_attention_mask = self.bias[None, key_length - query_length: key_length, :key_length]
        original_shape = self_attention_mask.shape
        new_shape = original_shape[:2] + [1, ] + original_shape[2:]
        attention_mask = self_attention_mask.reshape(new_shape)

        hidden_states = self.wte(inputs)

        alibi_dtype = mx.float32 if self.attention_bias_in_fp32 else self.wte.weight.dtype
        alibi = get_alibi_biases(hidden_states.shape[0], seq_length_with_past,
                                 self.num_heads, alibi_dtype)[:, :, -query_length:, :]

        hidden_states_last_dim_size = hidden_states.shape[-1]
        output_shape = input_shape + [hidden_states_last_dim_size, ]

        presents = [] if use_cache else None

        for e, layer in enumerate(self.h):
            hidden_states, past_key_value = layer(hidden_states,
                                                  use_cache=use_cache,
                                                  attention_mask=attention_mask,
                                                  alibi=alibi,
                                                  layer_past=past_key_values[e])
            if use_cache:
                presents.append(past_key_value)

        hidden_states = hidden_states.reshape(output_shape)

        # Merged from GPTRefactForCausalLM class
        # add a linear layer to the output of the last layer
        x = self.ln_f(hidden_states, debug=True)
        lm_logits = self.lm_head(x)

        return lm_logits, hidden_states, presents


def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temp))


def generate(prompt: mx.array, model: Refact, temp: 0.0):
    logits, hidden_states, cache = model(prompt, use_cache=True)
    y = sample(logits[:, -1, :])
    yield y

    cache, y = None, None
    while True:
        y = prompt if y is None else y[:, None]
        logits, hidden_states, cache = model(y, use_cache=True, past_key_values=cache)
        y = sample(logits[:, -1, :])
        yield y


def load_model(model_path: str, tokenizer_path: str = "smallcloudai/Refact-1_6B-fim"):
    model_args = ModelArgs()

    model_path = Path(model_path)
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
        model_args.n_embd = config["n_embd"]
        model_args.hidden_size = model_args.n_embd
        model_args.n_head = config["n_head"]
        model_args.num_attention_heads = model_args.n_head
        model_args.n_layer = config["n_layer"]
        model_args.n_positions = config["n_positions"]
        model_args.attention_bias_in_fp32 = config["attention_bias_in_fp32"]
        model_args.attention_softmax_in_fp32 = config["attention_softmax_in_fp32"]
        model_args.scale_attention_softmax_in_fp32 = config["scale_attention_softmax_in_fp32"]
        model_args.vocab_size = config["vocab_size"]
        model_args.multi_query = config["multi_query"]
        model_args.use_cache = config["use_cache"]

    model = Refact(model_args)
    weights = mx.load(str(model_path / "weights.npz"))
    # TODO: Support quantization
    # if quantization := config.get("quantization", False):
    #     nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, eos_token="<|endoftext|>"
    )
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refact inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and config",
    )
    parser.add_argument(
        "--tokenizer",
        help="The tokenizer to be used, defaults to smallcloudai/Refact-1_6B-fim",
        default="smallcloudai/Refact-1_6B-fim",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="How to load json file in python code.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path, args.tokenizer)

    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"]

    prompt = mx.array(prompt)
    print('prompt is:', prompt)

    print(args.prompt, end="", flush=True)

    tokens = []
    token_count = 0
    start_time = time.time()
    for token, _ in zip(generate(prompt, model, args.temp), range(args.max_tokens)):
        tokens.append(token)
        token_count += 1

        # stop if eos_token
        if token.item() == tokenizer.eos_token_id:
            break

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            eos_index = next(
                (i for i, t in enumerate(tokens) if t.item() == tokenizer.eos_token_id),
                None,
            )

            if eos_index is not None:
                tokens = tokens[:eos_index]

            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
            if eos_index is not None:
                break

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
    print(f'Token count is: {token_count},',
          f'time cost: {time.time() - start_time:.2f}s',
          f'token speed: {token_count / (time.time() - start_time):0.2f} tokens/sec')
