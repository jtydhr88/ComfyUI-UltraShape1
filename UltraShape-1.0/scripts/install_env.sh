conda create -n ultrashape python=3.10
conda activate ultrashape 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation

pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
