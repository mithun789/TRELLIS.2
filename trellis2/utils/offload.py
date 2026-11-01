"""
Fast CPU <-> GPU model offloading for low-VRAM inference.

The default low-VRAM path calls `model.to('cuda')` before each stage and
`model.cpu()` after it. That allocates new pageable host memory and copies
every weight back to the host on each unload, even though inference never
changes the weights.

`ModelOffloader` keeps one pinned host copy of each weight. Loading copies
it to the GPU with `non_blocking=True`. Unloading only rebinds the tensors to
the pinned host copy, so no device-to-host copy happens at all.
"""
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'no', 'off', '')


class ModelOffloader:
    """
    Moves modules between pinned host memory and the GPU.

    Args:
        pin_memory: Keep host copies in page-locked memory. Pinned memory makes
            transfers about 2x faster but locks host RAM. Defaults to the
            `TRELLIS2_PIN_MEMORY` environment variable (on by default).
    """
    def __init__(self, pin_memory: bool = None):
        if pin_memory is None:
            pin_memory = _env_flag('TRELLIS2_PIN_MEMORY', True)
        self.pin_memory = pin_memory and torch.cuda.is_available()
        # id(module) -> list of (owner submodule, slot dict name, key, host tensor)
        self._host: Dict[int, List[Tuple[nn.Module, str, str, torch.Tensor]]] = {}

    def _collect(self, module: nn.Module) -> List[Tuple[nn.Module, str, str, torch.Tensor]]:
        entries = []
        host_of: Dict[int, torch.Tensor] = {}  # shared (tied) tensors get one host copy
        for sub in module.modules():
            for slot in ('_parameters', '_buffers'):
                for key, tensor in getattr(sub, slot).items():
                    if tensor is None:
                        continue
                    host = host_of.get(id(tensor))
                    if host is None:
                        host = tensor.data.cpu()
                        if self.pin_memory:
                            host = host.pin_memory()
                        host_of[id(tensor)] = host
                    entries.append((sub, slot, key, host))
        return entries

    @staticmethod
    def _unwrap(module: Any) -> Any:
        # Feature extractor and rembg wrappers keep the network in `.model`.
        inner = getattr(module, 'model', None)
        if not isinstance(module, nn.Module) and isinstance(inner, nn.Module):
            return inner
        return module

    def onload(self, module: Any, device: torch.device) -> None:
        device = torch.device(device)
        module = self._unwrap(module)
        if not isinstance(module, nn.Module) or device.type != 'cuda':
            module.to(device)
            return
        key = id(module)
        if key not in self._host:
            self._host[key] = self._collect(module)
        moved: Dict[int, torch.Tensor] = {}
        for sub, slot, name, host in self._host[key]:
            gpu = moved.get(id(host))
            if gpu is None:
                gpu = host.to(device, non_blocking=True)
                moved[id(host)] = gpu
            tensor = getattr(sub, slot)[name]
            if slot == '_parameters':
                tensor.data = gpu
            else:
                sub._buffers[name] = gpu
        # Catch tensors registered after the first load.
        module.to(device)

    def offload(self, module: Any) -> None:
        module = self._unwrap(module)
        if not isinstance(module, nn.Module) or id(module) not in self._host:
            module.cpu()
            return
        for sub, slot, name, host in self._host[id(module)]:
            tensor = getattr(sub, slot)[name]
            if slot == '_parameters':
                tensor.data = host
            else:
                sub._buffers[name] = host
        # Moves anything created on the GPU during forward. No-op for host tensors.
        module.cpu()

    def forget(self, module: Any) -> None:
        self._host.pop(id(self._unwrap(module)), None)
