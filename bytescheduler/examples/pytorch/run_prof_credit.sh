export USE_BYTESCHEDULER=1
export BYTESCHEDULER_TUNE_THRES=0.0
profdir=/data2/home/zzhong/profexp_credit
iters="100"
ngpu="1"
nproc=$(($ngpu + $ngpu))
netif="eno1"
export BYTESCHEDULER_CREDIT=4000000
export BYTESCHEDULER_PARTITION=500000
export BYTESCHEDULER_ROOT_IP=proj54

export BYTESCHEDULER_ROOT_PORT=58888
export CUDA_VISIBLE_DEVICES=0

collocate_count="2"
# for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
for model in resnet50
do
	mkdir -p ${profdir}/throughput_${collocate_count}
	mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x CUDA_VISIBLE_DEVICES -x BYTESCHEDULER_ROOT_IP -x BYTESCHEDULER_ROOT_PORT -x BYTESCHEDULER_PARTITION -x BYTESCHEDULER_CREDIT -x BYTESCHEDULER_TUNE_THRES -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 | tee ${profdir}/throughput_${collocate_count}/${model}.txt
done
