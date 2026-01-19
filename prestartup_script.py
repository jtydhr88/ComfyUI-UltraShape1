import shutil
from pathlib import Path

def copy_assets():
    """Copy assets to ComfyUI/input directories"""
    try:
        custom_node_dir = Path(__file__).parent
        comfyui_dir = custom_node_dir.parent.parent
        input_dir = comfyui_dir / "input"
        input_dir_3d = input_dir / "3d"

        # Ensure destinations exist
        input_dir.mkdir(parents=True, exist_ok=True)
        input_dir_3d.mkdir(parents=True, exist_ok=True)

        # Copy 1.glb to input/3d
        glb_src = custom_node_dir / "assets" / "1.glb"
        shutil.copy2(glb_src, input_dir_3d)
        print(f"[UltraShape] Copied 1.glb to {input_dir_3d}")

        # Copy 1.png to input
        png_src = custom_node_dir / "assets" / "1.png"
        shutil.copy2(png_src, input_dir)
        print(f"[UltraShape] Copied 1.png to {input_dir}")
    except Exception as e:
        print(f"[UltraShape] Failed to copy assets: {e}")

# Run on import (not just __main__)
copy_assets()
