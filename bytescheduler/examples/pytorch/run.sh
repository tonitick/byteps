export USE_BYTESCHEDULER=1
export BYTESCHEDULER_CREDIT_TUNING=1
export BYTESCHEDULER_PARTITION_TUNING=0
export BYTESCHEDULER_ROOT_IP=localhost
export BYTESCHEDULER_ROOT_PORT=38888
export CUDA_VISIBLE_DEVICES=2,3

iters="1000"
ngpu="1"
nproc=$(($ngpu + $ngpu))
netif="eno1"
# for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
for model in resnet50
do
	# mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H localhost:${nproc} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x BYTESCHEDULER_CREDIT_TUNING -x BYTESCHEDULER_PARTITION_TUNING -x CUDA_VISIBLE_DEVICES -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32
	mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x BYTESCHEDULER_CREDIT_TUNING -x BYTESCHEDULER_PARTITION_TUNING -x CUDA_VISIBLE_DEVICES -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32
done
