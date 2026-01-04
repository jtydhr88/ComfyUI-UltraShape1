
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
NET_TYPE="high"
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi
# export NCCL_DEBUG=INFO

node_num=$1
node_rank=$2
num_gpu_per_node=$3
master_ip=$4
config=$5
output_dir=$6

echo node_num $node_num
echo node_rank $node_rank
echo master_ip $master_ip
echo config $config
echo output_dir $output_dir

if test -d "$output_dir"; then
    cp $config $output_dir
else
    mkdir -p "$output_dir"
    cp $config $output_dir
fi

NODE_RANK=$node_rank \
HF_HUB_OFFLINE=0 \
MASTER_PORT=12348 \
MASTER_ADDR=$master_ip \
NCCL_SOCKET_IFNAME=bond1 \
NCCL_IB_GID_INDEX=3 \
NCCL_NVLS_ENABLE=0 \
python3 main.py \
    --num_nodes $node_num \
    --num_gpus $num_gpu_per_node \
    --config $config \
    --output_dir $output_dir \
    --deepspeed
