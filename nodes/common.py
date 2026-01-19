"""Common utilities and constants for UltraShape nodes."""

import os
import time

# Directory paths
NODES_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(NODES_DIR)

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
    COMFY_INPUT_DIR = folder_paths.get_input_directory()
except ImportError:
    COMFY_MODELS_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), "models")
    COMFY_OUTPUT_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), "output")
    COMFY_INPUT_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), "input")

ULTRASHAPE_MODELS_DIR = os.path.join(COMFY_MODELS_DIR, "UltraShape")

# HuggingFace model info
ULTRASHAPE_HF_REPO = "infinith/UltraShape"
DEFAULT_CHECKPOINT = "ultrashape_v1.pt"

# Config directory (inside nodes/)
CONFIG_DIR = os.path.join(NODES_DIR, "configs")


def ensure_ultrashape_checkpoint():
    """Download default checkpoint from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download

    os.makedirs(ULTRASHAPE_MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(ULTRASHAPE_MODELS_DIR, DEFAULT_CHECKPOINT)

    if not os.path.exists(ckpt_path):
        print(f"[UltraShape] Checkpoint not found. Downloading {DEFAULT_CHECKPOINT} from HuggingFace...")
        try:
            hf_hub_download(
                repo_id=ULTRASHAPE_HF_REPO,
                filename=DEFAULT_CHECKPOINT,
                local_dir=ULTRASHAPE_MODELS_DIR,
                local_dir_use_symlinks=False
            )
            print(f"[UltraShape] Download complete: {ckpt_path}")
        except Exception as e:
            print(f"[UltraShape] Failed to download checkpoint: {e}")


def get_timestamp():
    """Generate timestamp string for filenames."""
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"
