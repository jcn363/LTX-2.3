# LTX-2.3

**LTX-2.3** is the core library for Lightricks' open-source generative video architecture based on the Diffusion Transformer (DiT). The model delivers commercial-grade generation quality with support for text-to-video, image-to-video, and audio-to-video generation pipelines.

---

## Key Features

* **Text-to-video** generation
* **Image-to-video** generation
* **Audio-to-video** generation with synchronized audio-video cross-attention
* **Spatiotemporal Attention & v-prediction:** Optimized mathematical noise prediction model that minimizes visual artifacts in highly dynamic scenes
* **FP8 Quantization:** Efficient 8-bit floating-point quantization for reduced memory usage
* **LoRA Fusion:** Apply and fuse LoRA adapters for model customization
* **Tiled Processing:** Spatial and temporal tiling for large-scale video generation
* **Multi-Scheduler Support:** LTX2, Linear-Quadratic, and Beta distribution schedulers
* **Multiple Attention Backends:** PyTorch SDPA, xFormers, and FlashAttention3 (when installed)

---

## Architecture

LTX-2.3 uses a DiT (Diffusion Transformer) architecture with the following core components:

### Models
- **Diffusion Transformer:** Audio-Video cross-attention transformer with RoPE positional embeddings, AdaLN conditioning, and per-head gating
- **Video VAE:** Encoder-decoder with causal convolutions, tiled processing, and multi-scale spatial/temporal compression (8x temporal, 32x spatial)
- **Audio VAE:** Spectrogram encoder-decoder with causal convolutions for audio latent representations
- **Latent Upsampler:** Spatial/temporal upsampling of VAE latents with pixel shuffle and rational resampling

### Text Encoding
- **Gemma3 Text Encoder:** Multi-modal text encoder with vision tower support
- **Feature Extractors:** V1 (per-segment norm) and V2 (per-token RMS norm with dual aggregate embeds)

### Diffusion Components
- **Schedulers:** LTX2 (token-count-dependent shifting), Linear-Quadratic, and Beta distribution
- **Guiders:** CFG, CFG-Star Rescaling, STG, APG, and multi-modal guider factory
- **Noisers:** Gaussian noise with denoise mask scaling
- **Diffusion Steps:** Euler and Res2s (second-order SDE) methods

### Loading & Quantization
- **Model Loader:** Safetensors loading with LoRA fusion and state dict registry
- **FP8 Quantization:** Per-tensor and cast-only FP8 weight quantization
- **Module Operations:** Model-level weight transformations

---

## System Requirements

| Component | Minimum Requirements | Recommended |
| --- | --- | --- |
| **Python** | 3.10+ | 3.11+ |
| **PyTorch** | 2.7+ (CUDA 12.9) | 2.7+ (CUDA 12.9) |
| **GPU** | NVIDIA GPU with CUDA support | NVIDIA RTX 4080/4090 |
| **RAM** | 16 GB | 32 GB |

### Optional Dependencies
- **xformers:** Memory-efficient attention backend
- **tensorrt-llm:** FP8 quantization via TensorRT-LLM

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd LTX-2.3

# Install the core package
cd ltx-core
uv pip install -e .

# With xformers support
uv pip install -e ".[xformers]"

# With FP8 TensorRT-LLM support
uv pip install -e ".[fp8-trtllm]"
```

---

## Usage

### Loading a Model

```python
from ltx_core.loader import SafetensorsModelStateDictLoader, SingleGPUModelBuilder

# Load model configuration and weights
loader = SafetensorsModelStateDictLoader()
config = loader.metadata("model.safetensors")
builder = SingleGPUModelBuilder(loader=loader, path="model.safetensors")
model = builder.build()
```

### Video Encoding/Decoding

```python
from ltx_core.model.video_vae import VideoEncoder, VideoDecoder, decode_video

# Encode video to latent
encoder = VideoEncoder(...)
latent = encoder(video_tensor)

# Decode latent to video
decoder = VideoDecoder(...)
for chunk in decode_video(latent, decoder):
    # Process video chunks
    pass
```

### Diffusion Pipeline Components

```python
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.guiders import CFGGuider, MultiModalGuiderFactory
from ltx_core.components.diffusion_steps import EulerDiffusionStep

# Create scheduler
scheduler = LTX2Scheduler()
sigmas = scheduler.execute(steps=20, latent=latent)

# Create guider
guider = CFGGuider(scale=3.0)

# Create diffusion step
step = EulerDiffusionStep()
```

### Attention Backends

```python
from ltx_core.model.transformer.attention import AttentionFunction

# Use default (xformers if available, else PyTorch)
attn_fn = AttentionFunction.DEFAULT

# Force specific backend
attn_fn = AttentionFunction.PYTORCH
attn_fn = AttentionFunction.XFORMERS
attn_fn = AttentionFunction.FLASH_ATTENTION_3
```

### LoRA Fusion

```python
from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.primitives import LoraStateDictWithStrength

# Apply LoRA adapters
lora_sds = [LoraStateDictWithStrength(state_dict=lora_sd, strength=0.8)]
fused_sd = apply_loras(model_sd, lora_sds)
```

---

## Prompting for LTX-2.3

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

- Start with main action in a single sentence
- Add specific details about movements and gestures
- Describe character/object appearances precisely
- Include background and environment details
- Specify camera angles and movements
- Describe lighting and colors
- Note any changes or sudden events

---

## FAQ

**Q: How fast is text-to-video generation?**
**A:** Speed depends on your hardware and configuration. The core library provides the building blocks; actual generation speed depends on your pipeline implementation, model size, and hardware.

**Q: How large are the model weights?**
**A:** The base model weights vary by configuration. FP8 quantized weights reduce size by approximately 50% compared to BF16 precision.

**Q: What text encoder is used?**
**A:** LTX-2.3 uses the Gemma3 text encoder (not T5). It includes a vision tower for multi-modal understanding and feature extraction with both V1 and V2 configurations.

**Q: Can this run without a GPU?**
**A:** The library requires PyTorch with CUDA support. While some components may work on CPU, GPU is recommended for practical inference speeds.

---

## License

See LICENSE file for details.
