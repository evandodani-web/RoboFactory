"""Frozen SigLIP feature extractor for the CLS-DP contextualizer.

The paper encodes the agent's current local RGB frame and the shared task instruction with
frozen pretrained SigLIP encoders (ViT-B/16, feature dim 768). Because both towers are
frozen and the prior consumes only the *current* frame, features are precomputed offline
and cached into the multi-agent zarr; see script/precompute_siglip_features.py.

This module is deliberately never a submodule of a trained policy. Keeping SigLIP outside
the saved model keeps checkpoints at ~20MB instead of ~800MB, and means dataloader workers
never need a GPU. The trainable PriorNet consumes cached token features instead.

Patch tokens are average-pooled from the native 14x14 grid down to a coarse grid (4x4 by
default) before caching. Caching all 196 patch tokens would cost ~300KB/frame in fp16,
which is larger than the source image; 4x4 plus the pooled embedding is ~26KB/frame while
still giving the fusion cross-attention real spatial structure to attend over.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"

# SigLIP rescales images to [-1, 1] rather than using ImageNet statistics.
SIGLIP_MEAN = 0.5
SIGLIP_STD = 0.5


class SigLIPFeatureExtractor(nn.Module):
    """Wraps a frozen HuggingFace SigLIP model and exposes token-level features."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        pool_grid: int = 4,
        text_max_length: int = 64,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "CLS-DP needs `transformers` for the frozen SigLIP encoders. "
                "Install it with: pip install 'transformers>=4.37'"
            ) from exc

        device = torch.device(device)
        if device.type != "cuda" and dtype == torch.float16:
            # fp16 matmuls are not well supported on CPU
            dtype = torch.float32

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._device = device
        self._dtype = dtype
        self.pool_grid = pool_grid
        self.text_max_length = text_max_length
        self.image_size = self.model.config.vision_config.image_size

    @property
    def device(self):
        return self._device

    @property
    def feature_dim(self) -> int:
        return self.model.config.vision_config.hidden_size

    @property
    def n_image_tokens(self) -> int:
        """Pooled embedding plus the coarse patch grid."""
        return 1 + self.pool_grid * self.pool_grid

    def train(self, mode: bool = True):
        # The backbone stays in eval mode permanently.
        super().train(False)
        return self

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB frames.

        Args:
            images: (B, 3, H, W) float tensor with values in [0, 1].

        Returns:
            (B, 1 + pool_grid^2, feature_dim) float32 tensor. Index 0 is the attention
            pooled embedding; the remainder is the pooled patch grid in row-major order.
        """
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = images.to(device=self._device, dtype=torch.float32)

        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - SIGLIP_MEAN) / SIGLIP_STD

        outputs = self.model.vision_model(pixel_values=images.to(self._dtype))
        patch_tokens = outputs.last_hidden_state  # (B, P, D)
        pooled = outputs.pooler_output  # (B, D)

        b, n_patches, dim = patch_tokens.shape
        side = int(round(n_patches**0.5))
        if side * side != n_patches:
            raise RuntimeError(
                f"expected a square patch grid, got {n_patches} tokens"
            )

        grid = patch_tokens.transpose(1, 2).reshape(b, dim, side, side)
        grid = F.adaptive_avg_pool2d(grid.float(), self.pool_grid)
        grid = grid.reshape(b, dim, self.pool_grid * self.pool_grid).transpose(1, 2)

        return torch.cat([pooled.float().unsqueeze(1), grid], dim=1)

    @torch.no_grad()
    def encode_text(self, texts):
        """Encode a list of instruction strings.

        Returns:
            tokens: (B, text_max_length, feature_dim) float32
            mask:   (B, text_max_length) float32, 1 for real tokens
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(input_ids)
        mask = mask.to(self._device).float()

        # SigLIP's text tower is trained with fixed-length padding and full attention, so
        # the mask is used only for our own downstream pooling, not passed to the model.
        outputs = self.model.text_model(input_ids=input_ids)
        return outputs.last_hidden_state.float(), mask
