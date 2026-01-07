"""
ComfyUI-UltraShape1

Image-guided 3D mesh refinement with coarse-to-fine diffusion.
"""

import sys
import os

# Track initialization status
INIT_SUCCESS = False

if not os.environ.get('PYTEST_CURRENT_TEST'):
    print("[ComfyUI-UltraShape1] Initializing custom node...")

    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[ComfyUI-UltraShape1] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        import traceback
        print(f"[ComfyUI-UltraShape1] [WARNING] Failed to import node classes: {e}")
        print(f"[ComfyUI-UltraShape1] Traceback:\n{traceback.format_exc()}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    if INIT_SUCCESS:
        print("[ComfyUI-UltraShape1] [OK] Loaded successfully!")
    else:
        print("[ComfyUI-UltraShape1] [ERROR] Failed to load - check errors above")

else:
    print("[ComfyUI-UltraShape1] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.2.0"
