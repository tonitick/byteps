export USE_BYTESCHEDULER=1
# export BYTESCHEDULER_CREDIT_TUNING=0
# export BYTESCHEDULER_PARTITION_TUNING=1
# export BYTESCHEDULER_CREDIT=4000000
# export BYTESCHEDULER_PARTITION=1000000
export CUDA_VISIBLE_DEVICES=0,1,
iters="1000"
ngpu="1"
nproc=$(($ngpu + $ngpu))
netif="eno1"
# for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
for model in resnet50
do
	mpirun --allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model
	# mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x BYTESCHEDULER_CREDIT_TUNING -x BYTESCHEDULER_PARTITION_TUNING -x BYTESCHEDULER_CREDIT -x BYTESCHEDULER_PARTITION -x CUDA_VISIBLE_DEVICES -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32
done
