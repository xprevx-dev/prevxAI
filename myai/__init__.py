"""myai — a decoder-only transformer language model written from scratch.

No model APIs, no pretrained weights, no HF transformers. The only third-party
dependency is PyTorch (autograd + CUDA). Everything else in this package —
tokenizer, model, data pipeline, training loop, sampler, chat template, server —
is implemented here.
"""

__version__ = "0.1.0"

from myai.config import ModelConfig, TrainConfig, PRESETS, pick_preset_for_vram

__all__ = ["ModelConfig", "TrainConfig", "PRESETS", "pick_preset_for_vram", "__version__"]
