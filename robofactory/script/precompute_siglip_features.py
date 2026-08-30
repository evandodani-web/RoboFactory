"""Precompute frozen SigLIP features for a multi-agent CLS-DP zarr.

Both SigLIP towers are frozen and the CLS-DP prior consumes only the *current* frame, so
features can be computed once offline. This is a required pipeline step, not an
optimisation: the dataset reads cached features and never touches SigLIP, which keeps
dataloader workers CPU-only and keeps checkpoints small.

Image features are written back into the same zarr as `siglip_img_agent{i}` with shape
(T, 1 + pool_grid^2, feature_dim) in float16. Text features go to a sidecar .npz because
their leading dimension is the instruction count, not the timestep count, and the
ReplayBuffer expects every array under data/ to share a time axis.

Usage:
    python script/precompute_siglip_features.py \
        --zarr_path data/zarr_data/LiftBarrier_multi_150.zarr
"""

import argparse
import json
import os

import numpy as np
import torch
import zarr

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTION_DIR = os.path.join(_PACKAGE_DIR, "configs", "instructions")

_POLICY_DIR = os.path.join(_PACKAGE_DIR, "policy", "Diffusion-Policy")
import sys  # noqa: E402

if _POLICY_DIR not in sys.path:
    sys.path.insert(0, _POLICY_DIR)

from diffusion_policy.model.cls.siglip_encoder import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    SigLIPFeatureExtractor,
)


def text_cache_path(zarr_path):
    return zarr_path.rstrip("/") + "_text.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pool_grid", type=int, default=14)
    parser.add_argument("--text_max_length", type=int, default=64)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute image features even if they already exist",
    )
    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode="a")
    data_group = root["data"]

    task_name = root.attrs["task_name"]
    instruction_task = root.attrs.get("instruction_task", task_name)
    n_agents = int(root.attrs["n_agents"])

    extractor = SigLIPFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        pool_grid=args.pool_grid,
        text_max_length=args.text_max_length,
    )
    feature_dim = extractor.feature_dim
    n_tokens = extractor.n_image_tokens
    print(
        f"{task_name}: {n_agents} agents, "
        f"{n_tokens} image tokens x {feature_dim} dims per frame"
    )

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    for agent_id in range(n_agents):
        cam_key = f"head_camera_agent{agent_id}"
        out_key = f"siglip_img_agent{agent_id}"

        if out_key in data_group and not args.overwrite:
            print(f"{out_key} already exists, skipping (use --overwrite to redo)")
            continue

        camera = data_group[cam_key]
        n_steps = camera.shape[0]

        out = data_group.zeros(
            out_key,
            shape=(n_steps, n_tokens, feature_dim),
            chunks=(args.batch_size, n_tokens, feature_dim),
            dtype="float16",
            overwrite=True,
            compressor=compressor,
        )

        for start in range(0, n_steps, args.batch_size):
            end = min(start + args.batch_size, n_steps)
            batch = torch.from_numpy(np.asarray(camera[start:end]))
            features = extractor.encode_image(batch)
            out[start:end] = features.cpu().numpy().astype(np.float16)
            print(
                f"agent {agent_id}: {end}/{n_steps} frames",
                end="\r",
            )
        print()

    # Text features for both instruction splits.
    bank_path = os.path.join(INSTRUCTION_DIR, f"{instruction_task}.json")
    with open(bank_path) as f:
        bank = json.load(f)

    text_cache = {}
    for split in ("train", "eval"):
        tokens, mask = extractor.encode_text(bank[split])
        text_cache[f"{split}_tokens"] = tokens.cpu().numpy().astype(np.float16)
        text_cache[f"{split}_mask"] = mask.cpu().numpy().astype(np.float16)
        print(f"{split} instructions: {text_cache[f'{split}_tokens'].shape}")

    out_path = text_cache_path(args.zarr_path)
    np.savez_compressed(out_path, **text_cache)

    root.attrs["siglip_model_name"] = args.model_name
    root.attrs["siglip_feature_dim"] = feature_dim
    root.attrs["siglip_n_image_tokens"] = n_tokens
    root.attrs["siglip_text_max_length"] = args.text_max_length

    print(f"image features -> {args.zarr_path}")
    print(f"text features  -> {out_path}")


if __name__ == "__main__":
    main()
