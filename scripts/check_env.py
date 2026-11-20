"""Report the active Python and PyTorch runtime."""
import json
import platform

import torch

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "devices": [
        {"name": device.name, "memory_bytes": device.total_memory}
        for device in map(torch.cuda.get_device_properties, range(torch.cuda.device_count()))
    ],
}, indent=2))
