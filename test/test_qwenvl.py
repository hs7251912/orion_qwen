#!/usr/bin/env python3
import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except Exception:
    from transformers import AutoModelForVision2Seq as QwenVLModel


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def main():
    model_path = '/root/autodl-tmp/Orion_modify/ckpts/Qwen2-VL-7B-Instruct'

    print(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    dtype = pick_dtype()
    print(f"Loading model: {model_path} (dtype={dtype}, device_map='auto' if CUDA else CPU)")
    model = QwenVLModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    device = next(model.parameters()).device
    print(f"Model loaded to device: {device}")
    _ = model.eval()


if __name__ == "__main__":
    main()