# sampling 
# python scripts/sampling.py \
#     --mesh_json data/mesh_paths.json \
#     --output_dir data/sample

# inference refine_dit
python scripts/infer_dit_refine.py \
    --ckpt checkpoints/ultrashape_v1.pt \
    --image inputs/image/1.png \
    --mesh inputs/coarse_mesh/1.glb \
    --config configs/infer_dit_refine.yaml
