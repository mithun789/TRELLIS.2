import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import sparse as sp


def _pack_4bit_tensor(q: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.int8)
    q = torch.clamp(q, -8, 7)
    q_u = ((q + 8) & 0xF).to(torch.uint8).reshape(-1)
    if q_u.numel() % 2 == 1:
        q_u = torch.cat([q_u, torch.zeros(1, dtype=torch.uint8, device=q_u.device)])
    q_u = q_u.view(-1, 2)
    return q_u[:, 0] | (q_u[:, 1] << 4)


def _unpack_4bit_tensor(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    packed = packed.reshape(-1)
    low = packed & 0xF
    high = packed >> 4
    q = torch.empty((packed.numel() * 2,), dtype=torch.int8, device=packed.device)
    q[0::2] = low.to(torch.int8)
    q[1::2] = high.to(torch.int8)
    q = q[: torch.prod(torch.tensor(shape, dtype=torch.int64)).item()]
    q = q.view(shape)
    return q - 8


def _quantize_weight(weight: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    if bits != 4:
        raise ValueError('Only 4-bit quantization is supported.')

    flat_weight = weight.view(weight.size(0), -1)
    max_vals = flat_weight.abs().amax(dim=1)
    scales = torch.where(max_vals > 0, max_vals / 7.0, torch.ones_like(max_vals))
    scales = scales.to(weight.dtype)

    q_weight = torch.clamp((flat_weight / scales[:, None]).round(), -8, 7).to(torch.int8)
    q_weight = q_weight.view(weight.shape)
    packed = _pack_4bit_tensor(q_weight)
    return packed, scales


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        bits: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.dtype = dtype

        self.register_buffer('weight_packed', torch.empty(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.empty((out_features,), dtype=self.dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=self.dtype))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, module: nn.Linear, bits: int = 4, dtype: torch.dtype = torch.float16):
        quantized = cls(module.in_features, module.out_features, module.bias is not None, dtype=dtype, bits=bits)
        packed, scales = _quantize_weight(module.weight.data, bits=bits)
        quantized.weight_packed = packed.to(dtype=torch.uint8)
        quantized.weight_scales = scales.to(dtype=dtype)
        if module.bias is not None:
            quantized.bias.data = module.bias.data.to(dtype=dtype)
        return quantized

    def dequantize_weight(self) -> torch.Tensor:
        q = _unpack_4bit_tensor(self.weight_packed, torch.Size((self.out_features, self.in_features)))
        scales = self.weight_scales.view(self.out_features, *([1] * (q.ndim - 1)))
        return q.to(self.dtype) * scales

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.dtype)
        weight = self.dequantize_weight()
        return F.linear(input, weight, self.bias)


class QuantizedSparseLinear(QuantizedLinear):
    def forward(self, input):
        feats = input.feats.to(self.dtype)
        weight = self.dequantize_weight()
        return input.replace(F.linear(feats, weight, self.bias))


def quantize_model(model: nn.Module, bits: int = 4, dtype: torch.dtype = torch.float16) -> None:
    """Replace linear modules in a model with 4-bit quantized equivalents."""
    for name, child in list(model.named_children()):
        if isinstance(child, sp.SparseLinear):
            model._modules[name] = QuantizedSparseLinear.from_float(child, bits=bits, dtype=dtype)
        elif isinstance(child, nn.Linear):
            model._modules[name] = QuantizedLinear.from_float(child, bits=bits, dtype=dtype)
        else:
            quantize_model(child, bits=bits, dtype=dtype)
