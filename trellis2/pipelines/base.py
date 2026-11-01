from typing import *
import torch
import torch.nn as nn
from .. import models
from ..utils.offload import ModelOffloader


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", quantize_bits: int = 0, quantize_dtype: str = 'float16') -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}", quantize_bits=quantize_bits, quantize_dtype=quantize_dtype)
            except Exception as e:
                _models[k] = models.from_pretrained(v, quantize_bits=quantize_bits, quantize_dtype=quantize_dtype)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    @property
    def offloader(self) -> ModelOffloader:
        if getattr(self, '_offloader', None) is None:
            self._offloader = ModelOffloader()
        return self._offloader

    def _onload(self, model) -> None:
        """Move a model to the pipeline device (low-VRAM mode)."""
        self.offloader.onload(model, self.device)

    def _offload(self, model) -> None:
        """Release a model's GPU memory (low-VRAM mode)."""
        self.offloader.offload(model)

    def offload_all(self) -> None:
        """Move every model off the GPU, e.g. after an out-of-memory error."""
        for model in self.models.values():
            self._offload(model)
        for name in ('image_cond_model', 'rembg_model'):
            extra = getattr(self, name, None)
            if extra is not None:
                self._offload(extra)

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))