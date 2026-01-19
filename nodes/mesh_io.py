"""UltraShape mesh IO nodes (file selection and saving)."""

import os
import uuid

from common import COMFY_OUTPUT_DIR, get_timestamp

try:
    import folder_paths
except ImportError:
    folder_paths = None


class UltraShapeMeshSelector:
    """Upload and select mesh file.

    Provides file upload functionality for mesh files.
    Upload files to ComfyUI/input/ directory and select them.
    Outputs a relative path string that can be connected to UltraShape Load Coarse Mesh.

    Supported formats: .glb, .gltf, .obj, .ply, .stl
    """

    MESH_EXTENSIONS = [".glb", ".gltf", ".obj", ".ply", ".stl"]

    @classmethod
    def INPUT_TYPES(s):
        # Scan for mesh files in input directory
        files = []

        if folder_paths:
            input_dir = folder_paths.get_input_directory()
            if os.path.exists(input_dir):
                for root, dirs, filenames in os.walk(input_dir):
                    for f in filenames:
                        if any(f.lower().endswith(ext) for ext in s.MESH_EXTENSIONS):
                            full_path = os.path.join(root, f)
                            rel_path = os.path.relpath(full_path, input_dir)
                            files.append(rel_path.replace("\\", "/"))

        files.sort()

        if not files:
            files = ["(upload a mesh file)"]

        return {
            "required": {
                # Use "image" as parameter name with image_upload to enable upload button
                "image": (files, {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    FUNCTION = "select"
    CATEGORY = "UltraShape/Loaders"

    @classmethod
    def IS_CHANGED(s, image):
        if not folder_paths or image == "(upload a mesh file)":
            return float("nan")
        image_path = folder_paths.get_annotated_filepath(image)
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if image == "(upload a mesh file)":
            return "Please upload a mesh file (.glb, .obj, .ply, .stl)"
        if folder_paths and not folder_paths.exists_annotated_filepath(image):
            return f"Mesh file not found: {image}"
        return True

    def select(self, image):
        if image == "(upload a mesh file)":
            raise ValueError("Please upload a mesh file first.")

        # Return path with input/ prefix so UltraShapeLoadCoarseMesh can resolve it
        output_path = f"input/{image}"
        print(f"[UltraShape] Selected mesh: {output_path}")
        return (output_path,)


class UltraShapeSaveGLB:
    """Save refined mesh as GLB file"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "ultrashape_output"}),
                "filename_prefix": ("STRING", {"default": "refined"}),
                "file_format": (["glb", "obj", "ply", "stl"], {"default": "glb"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "UltraShape"
    OUTPUT_NODE = True

    def save(self, mesh, output_dir="ultrashape_output",
             filename_prefix="refined", file_format="glb"):
        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)

        ts = get_timestamp()
        uid = str(uuid.uuid4())[:8]
        filename = f"{filename_prefix}_{ts}_{uid}.{file_format}"
        save_path = os.path.join(out_dir, filename)

        # Export mesh (trimesh object)
        mesh.export(save_path)

        print(f"[UltraShape] Saved: {save_path}")

        # Return relative path
        rel_path = os.path.relpath(save_path, COMFY_OUTPUT_DIR)
        return (rel_path,)
