"""
Ready-made settings for running TRELLIS.2 on GPUs with limited memory.

Usage:
    from trellis2.utils import vram_profiles
    vram_profiles.apply_runtime_settings()          # call before any CUDA work
    profile = vram_profiles.get_profile('8gb')
    pipeline = vram_profiles.load_pipeline('microsoft/TRELLIS.2-4B', profile)
    mesh = vram_profiles.run_with_fallback(pipeline, image, profile)[0]

Select a profile with the `TRELLIS2_VRAM_PROFILE` environment variable
(`8gb`, `12gb`, `24gb`). The default is `24gb`, the original behavior.
"""
import copy
import os
from typing import Any, Dict, List, Optional

PROFILES: Dict[str, Dict[str, Any]] = {
    # 8 GB cards (RTX 3060 Ti / 3070 / 4060 / 4060 Ti 8GB, laptops).
    # One model on the GPU at a time, 1024 cascade, small export mesh with a
    # baked normal map to keep surface detail.
    '8gb': {
        'low_vram': True,
        'pipeline_type': '1024_cascade',
        'fallback_pipeline_types': ['1024_cascade', '512'],
        'max_num_tokens': 32768,
        'preview_max_faces': 1_000_000,
        'glb': {
            'decimation_target': 100_000,
            'texture_size': 2048,
            'remesh': True,
            'remesh_band': 1,
            'remesh_project': 0,
            'bake_normal_map': True,
        },
    },
    # 12-16 GB cards (RTX 3060 12GB, 4070, 4080).
    '12gb': {
        'low_vram': True,
        'pipeline_type': '1024_cascade',
        'fallback_pipeline_types': ['1024_cascade', '512'],
        'max_num_tokens': 49152,
        'preview_max_faces': 4_000_000,
        'glb': {
            'decimation_target': 200_000,
            'texture_size': 4096,
            'remesh': True,
            'remesh_band': 1,
            'remesh_project': 0,
            'bake_normal_map': True,
        },
    },
    # 24 GB and above: original repository defaults.
    '24gb': {
        'low_vram': None,  # keep the value from pipeline.json
        'pipeline_type': None,  # keep the pipeline default
        'fallback_pipeline_types': [],
        'max_num_tokens': 49152,
        'preview_max_faces': 16_777_216,  # nvdiffrast limit
        'glb': {
            'decimation_target': 1_000_000,
            'texture_size': 4096,
            'remesh': True,
            'remesh_band': 1,
            'remesh_project': 0,
            'bake_normal_map': False,
        },
    },
}


def get_profile(name: Optional[str] = None) -> Dict[str, Any]:
    """Return a copy of a profile. `None` reads `TRELLIS2_VRAM_PROFILE`."""
    name = (name or os.environ.get('TRELLIS2_VRAM_PROFILE', '24gb')).lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown VRAM profile '{name}'. Choose from: {', '.join(PROFILES)}")
    profile = copy.deepcopy(PROFILES[name])
    profile['name'] = name
    return profile


def apply_runtime_settings() -> None:
    """
    Memory and speed settings for PyTorch. Call before the first CUDA allocation.
    """
    # Lets the caching allocator grow segments in place instead of fragmenting.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    # TF32 matmuls/convs for the fp32 parts (decoders, renderers). No effect on bf16/fp16 math.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_pipeline(path: str, profile: Dict[str, Any], pipeline_cls=None, **kwargs):
    """Load a pipeline, apply the profile, and move it to the GPU."""
    if pipeline_cls is None:
        from ..pipelines import Trellis2ImageTo3DPipeline as pipeline_cls
    pipeline = pipeline_cls.from_pretrained(path, **kwargs)
    if profile.get('low_vram') is not None:
        pipeline.low_vram = profile['low_vram']
    if profile.get('pipeline_type') and hasattr(pipeline, 'default_pipeline_type'):
        pipeline.default_pipeline_type = profile['pipeline_type']
    pipeline.cuda()
    return pipeline


def run_with_fallback(pipeline, image, profile: Dict[str, Any], pipeline_type: Optional[str] = None, **run_kwargs) -> List[Any]:
    """
    Run the image-to-3D pipeline. On CUDA out-of-memory, free memory and retry
    with the profile's fallback pipeline types (lower resolution).
    """
    import torch
    first = pipeline_type or profile.get('pipeline_type') or pipeline.default_pipeline_type
    rank = {'512': 0, '1024': 1, '1024_cascade': 1, '1536_cascade': 2}
    candidates = [first] + [
        t for t in profile.get('fallback_pipeline_types', [])
        if rank.get(t, 0) < rank.get(first, 0)
    ]
    run_kwargs.setdefault('max_num_tokens', profile.get('max_num_tokens', 49152))
    for i, ptype in enumerate(candidates):
        try:
            return pipeline.run(image, pipeline_type=ptype, **run_kwargs)
        except torch.cuda.OutOfMemoryError:
            if i + 1 == len(candidates):
                raise
            print(f"[vram_profiles] Out of memory at '{ptype}'. Retrying with '{candidates[i + 1]}'.")
        # Outside the except block, so the traceback and its tensors are released.
        if getattr(pipeline, 'low_vram', False):
            pipeline.offload_all()
        torch.cuda.empty_cache()


def peak_memory_report() -> str:
    import torch
    if not torch.cuda.is_available():
        return 'CUDA not available'
    gb = 1024 ** 3
    return (f"peak allocated {torch.cuda.max_memory_allocated() / gb:.2f} GB, "
            f"peak reserved {torch.cuda.max_memory_reserved() / gb:.2f} GB")
