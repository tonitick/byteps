from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision

from datetime import datetime

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)(num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Horovod: (optional) compression algorithm.
# compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
compression = hvd.Compression.none

# bytescheduler wrapper
use_bytescheduler = int(os.environ.get('USE_BYTESCHEDULER', '0'))
if use_bytescheduler > 0:
    if args.partition:
        os.environ["BYTESCHEDULER_PARTITION"] = str(1000 * args.partition)
    import bytescheduler.pytorch.horovod as bsc
    bsc.init()

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

if use_bytescheduler > 0:
    optimizer = bsc.ScheduledOptimizer(model, optimizer, args.num_warmup_batches + args.num_iters * args.num_batches_per_iter)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fake data
dataset = []
targetset = []
# for _ in range(100):
#     data = torch.rand(args.batch_size, 3, 224, 224)
#     target = torch.LongTensor(args.batch_size).random_() % 1000
#     # if args.cuda:
#     #     data, target = data.cuda(), target.cuda()
#     if args.cuda:
#         target = target.cuda()
#     dataset.append(data)
data_index = 0


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

for batch_idx, (inputs, targets) in enumerate(trainloader):
    dataset.append(inputs)
    targetset.append(targets)

def benchmark_step():
    global data_index
    global dataset
    global targetset

    data = dataset[data_index%len(dataset)]
    if args.cuda:
        data = data.cuda()
    target = targetset[data_index%len(targetset)]
    if args.cuda:
        target = target.cuda()
    data_index += 1
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    # print(loss) # convergence check
    optimizer.step()


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
time = timeit.timeit(benchmark_step, number=args.num_warmup_batches)
log('Warmup time: %.2fs' % (time))
log('Warmup end timestamp: %s' % (datetime.now().strftime("%s")))

# Benchmark
log('Running benchmark...')
img_secs = []

for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.2f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.2f +-%.2f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.2f +-%.2f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
