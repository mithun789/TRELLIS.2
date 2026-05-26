import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trellis2.quantization import quantize_model


def main():
    m = nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4)
    )

    print('model loaded')
    quantize_model(m, bits=4, dtype=torch.float16)
    print('model quantized')

    x = torch.randn(2, 16)
    y = m(x)

    print('forward ok')
    print('output shape:', y.shape)
    print('output dtype:', y.dtype)
    print('output sample:', y[0].tolist())


if __name__ == '__main__':
    main()

import torch
print("allocated GB:", torch.cuda.memory_allocated() / 1024**3)
print("reserved GB:", torch.cuda.memory_reserved() / 1024**3)
