# Transforms

## 🌀 [`TransformsFactory.py`](TransformsFactory.py)
Creates [TorchIO transform](https://docs.torchio.org/transforms/transforms.html) pipelines from JSON config.

**Supported:**
- TorchIO preprocessing transforms (resampling, cropping, etc.)
- TorchIO augmentation transforms (flip, affine, noise, etc.)

**Custom Transforms:**
Alternatively, implement preprocessing/augmentation directly in your dataset class.
