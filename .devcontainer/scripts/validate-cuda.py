#!/usr/bin/env python3
"""
Validate that JAX can see CUDA-capable GPU devices.
Runs automatically on every container start (postStartCommand).
"""
import sys


def main() -> None:
    try:
        import jax
    except ImportError:
        print("ERROR: jax is not installed. Run `uv sync` inside the container.")
        sys.exit(1)

    print(f"JAX version : {jax.__version__}")

    devices = jax.devices()
    print(f"All devices : {devices}")

    gpu_devices = [d for d in devices if d.platform == "gpu"]

    if gpu_devices:
        print(f"CUDA status : OK — {len(gpu_devices)} GPU(s) detected")
        for d in gpu_devices:
            print(f"             {d}")
    else:
        print("CUDA status : WARNING — no GPU devices found; JAX is running on CPU only.")
        print("              Check that --gpus=all is set and the NVIDIA driver is accessible.")
        sys.exit(1)


if __name__ == "__main__":
    main()
