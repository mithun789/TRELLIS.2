import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from trellis2.pipelines import from_pretrained


def main():
    model_repo = os.environ.get('TRELLIS2_MODEL', 'microsoft/TRELLIS.2-4B')
    quantize_bits = int(os.environ.get('TRELLIS2_QUANTIZE_BITS', '4'))
    quantize_dtype = os.environ.get('TRELLIS2_QUANTIZE_DTYPE', 'float16')

    print('Repo root:', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    print('Model:', model_repo)
    print('Quantize bits:', quantize_bits)
    print('Quantize dtype:', quantize_dtype)
    print('CUDA available:', torch.cuda.is_available())
    print('Torch version:', torch.__version__)

    if not torch.cuda.is_available():
        print('ERROR: CUDA is not available in this environment.')
        print('Run this script on a GPU-backed machine or Hugging Face Space with CUDA-enabled PyTorch.')
        return

    device = torch.device('cuda')
    print('Loading pipeline... this may take a while if it downloads model weights.')
    pipeline = from_pretrained(model_repo, quantize_bits=quantize_bits, quantize_dtype=quantize_dtype)
    pipeline.to(device)

    print('Pipeline loaded successfully.')
    print('CUDA device:', torch.cuda.get_device_name(device))
    print('Memory allocated (GB):', torch.cuda.memory_allocated(device) / 1024**3)
    print('Memory reserved  (GB):', torch.cuda.memory_reserved(device) / 1024**3)

    try:
        import subprocess
        print('nvidia-smi output:')
        subprocess.run(['nvidia-smi'], check=False)
    except Exception:
        pass


if __name__ == '__main__':
    main()
