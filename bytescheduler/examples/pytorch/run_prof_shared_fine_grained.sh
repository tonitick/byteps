export USE_BYTESCHEDULER=1
export BYTESCHEDULER_CREDIT_TUNING=0
export BYTESCHEDULER_ROOT_IP=localhost
export BYTESCHEDULER_ROOT_PORT=38888
profdir=/data2/home/zzhong/profexp_shared_2
iters="5"
ngpu="1"
nproc=$(($ngpu + $ngpu))
netif="eno1"
sleeptime=200
# for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
for model in resnet18 resnet50 resnet152 densenet121 densenet201
# for model in vgg16
do
	# for credit in 4000000 2000000 1000000 8000000 400000
	# for credit in 3500000 4500000 800000 1500000 200000

	# for credit in 1000000
	# for credit in 4000000
	# for credit in 8000000
	for credit in 16000000
	do
		# for partition in 1000000 500000 100000 10000 2000000 4000000
		# for partition in 200000 300000 400000 500000 1100000 1300000 1400000 1600000 1700000 1900000
		# for partition in 70000 60000 50000 40000 30000 20000 10000 110000 120000 130000 140000 150000 160000 170000 180000 190000

		for partition in 1000000 900000 800000 700000 600000 500000 400000 300000 200000 100000
		do
			mkdir -p ${profdir}/result_pytorch_horovod_single_${ngpu}gpupm_${nproc}proc_cs${credit}_ps${partition}/throughput

    		export CUDA_VISIBLE_DEVICES=0
			export BYTESCHEDULER_CREDIT=${credit}
			export BYTESCHEDULER_PARTITION=${partition}
			mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x BYTESCHEDULER_CREDIT_TUNING -x BYTESCHEDULER_CREDIT -x BYTESCHEDULER_PARTITION -x CUDA_VISIBLE_DEVICES -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 > ${profdir}/result_pytorch_horovod_single_${ngpu}gpupm_${nproc}proc_cs${credit}_ps${partition}/throughput/${model}.txt &
			pid1=$!
			echo "pid1=${pid1}"

    		export CUDA_VISIBLE_DEVICES=1
			mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x USE_BYTESCHEDULER -x BYTESCHEDULER_CREDIT_TUNING -x BYTESCHEDULER_CREDIT -x BYTESCHEDULER_PARTITION -x CUDA_VISIBLE_DEVICES -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 &
			pid2=$!
			echo "pid2=${pid2}"
			
			sleep $sleeptime
			kill  $pid1
			kill  $pid2
		done
	done
done

# profdir=/data2/home/zzhong/profdatashare
# interval="0.1"
# iters="5"
# ngpu="2"
# nproc=$(($ngpu + $ngpu))
# netif="eno1"
# # for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
# # for model in vgg11 vgg13 vgg16 vgg19 resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
# for model in resnet50 resnet34
# do
# 	mkdir -p ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc/throughput
# 
#     export CUDA_VISIBLE_DEVICES=0,1
# 	mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 > ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc/throughput/${model}.txt &
# 	pid1=$!
# 	echo "pid1=${pid1}"
# 	
#     export CUDA_VISIBLE_DEVICES=2,3
#     mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 > ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc_${interval}/throughput/${model}.txt &
# 	pid2=$!
# 	echo "pid2=${pid2}"
# 
# 	sleep $sleeptime
# 	wait $pid1
# 	wait $pid2
# 
# done
