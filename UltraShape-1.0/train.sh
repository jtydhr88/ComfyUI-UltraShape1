export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8

export node_num=1
export node_rank=$1
export master_ip= # [your master ip here]


############## vae ##############
# export config=configs/train_vae_refine.yaml
# export output_dir=outputs/vae_ultrashape/exp1_token8192
# bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir

############## dit ##############
export config=configs/train_dit_refine.yaml
export output_dir=outputs/dit_ultrashape/exp1_token8192
bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir

