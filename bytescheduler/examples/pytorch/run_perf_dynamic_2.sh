export USE_BYTESCHEDULER=1
export BYTESCHEDULER_TUNE_THRES=0.0
profdir=/data2/home/zzhong/perfexp_adt_dynamic
iters="200"
ngpu="1"
nproc=$(($ngpu + $ngpu))
netif="eno1"

# general setting
export BYTESCHEDULER_CREDIT=5000000
export BYTESCHEDULER_PARTITION=500000

export BYTESCHEDULER_ROOT_IP=proj54

export BYTESCHEDULER_ROOT_PORT=58889
export CUDA_VISIBLE_DEVICES=1

export SLOW_START_THRES=1.2
# export COLLECT_FREQ=2

cround=75
# for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
for model in resnet50
do
	mkdir -p ${profdir}/throughput_cs_${BYTESCHEDULER_CREDIT}_${cround}
	mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x CUDA_VISIBLE_DEVICES -x BYTESCHEDULER_ROOT_IP -x BYTESCHEDULER_ROOT_PORT -x BYTESCHEDULER_PARTITION -x BYTESCHEDULER_CREDIT -x BYTESCHEDULER_TUNE_THRES -x SLOW_START_THRES -x COLLECT_FREQ -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32
done
