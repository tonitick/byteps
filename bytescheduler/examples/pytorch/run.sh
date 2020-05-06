export USE_BYTESCHEDULER=1
export BYTESCHEDULER_WITH_PYTORCH=1
export BYTESCHEDULER_WITHOUT_MXNET=1

# for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
for model in resnet50
do
	export CUDA_VISIBLE_DEVICES=2,3
	mpirun --allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters 1000 --model=$model
done
