from __future__ import print_function

import time
import os
import argparse
import numpy as np
import cv2
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from augmentations import Augmentation_traininig

from loss import FocalLoss, OHEM_loss
from retinanet import RetinaNet
from datagen import ListDataset
from encoder import DataEncoder

from torch.autograd import Variable

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def adjust_learning_rate(cur_lr, optimizer, gamma, step):
    lr = cur_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--input_size', default=768, type=int, help='Input size for training')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--resume', default=None, type=str,  help='resume from checkpoint')
parser.add_argument('--dataset', type=str, help='select training dataset')
parser.add_argument('--multi_scale', default=False, type=str2bool, help='Use multi-scale training')
parser.add_argument('--focal_loss', default=True, type=str2bool, help='Use Focal loss or OHEM loss')
parser.add_argument('--logdir', default='logs/', type=str, help='Tensorboard log dir')
parser.add_argument('--max_iter', default=1200000, type=int, help='Number of training iterations')
parser.add_argument('--gamma', default=0.5, type=float, help='Gamma update for SGD')
parser.add_argument('--save_interval', default=500, type=int, help='Location to save checkpoint models')
parser.add_argument('--save_folder', default='eval/', help='Location to save checkpoint models')
parser.add_argument('--evaluation', default=False, type=str2bool, help='Evaulation during training')
parser.add_argument('--eval_step', default=1000, type=int, help='Evauation step')
parser.add_argument('--eval_device', default=6, type=int, help='GPU device for evaluation')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
assert args.focal_loss, "OHEM + ce_loss is not working... :("

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)

# Data
print('==> Preparing data..')
trainset = ListDataset(root='/root/DB/', dataset=args.dataset, train=True, 
                       transform=Augmentation_traininig, input_size=args.input_size, multi_scale=args.multi_scale)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                          shuffle=True, num_workers=args.num_workers, collate_fn=trainset.collate_fn)

# set model (focal_loss vs OHEM_CE loss)
if args.focal_loss:
    imagenet_pretrain = 'weights/retinanet_se50.pth'
    criterion = FocalLoss()
    num_classes = 1
else:
    imagenet_pretrain = 'weights/retinanet_se50_OHEM.pth'
    criterion = OHEM_loss()
    num_classes = 2

    
# Training Detail option\
stepvalues = (10000, 20000, 30000, 40000, 50000) if args.dataset in ["SynthText"] else (2000, 4000, 6000, 8000, 10000)
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
iteration = 0
cur_lr = args.lr
mean=(0.485,0.456,0.406)
var=(0.229,0.224,0.225)
step_index = 0
pEval = None

# Model
net = RetinaNet(num_classes)
net.load_state_dict(torch.load(imagenet_pretrain))

if args.resume:
    print('==> Resuming from checkpoint..', args.resume)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    #start_epoch = checkpoint['epoch']
    #iteration = checkpoint['iteration']
    #cur_lr = checkpoint['lr']
    #step_index = checkpoint['step_index']
    #optimizer.load_state_dict(state["optimizer"])

    
print("multi_scale : ", args.multi_scale)
print("input_size : ", args.input_size)
print("stepvalues : ", stepvalues)
print("start_epoch : ", start_epoch)
print("iteration : ", iteration)
print("cur_lr : ", cur_lr)
print("step_index : ", step_index)
print("num_gpus : ", torch.cuda.device_count())

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

# Training
net.train()
net.module.freeze_bn() # you must freeze batchnorm

optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=0.9, weight_decay=1e-4)
#optimizer = optim.Adam(net.parameters(), lr=cur_lr)

encoder = DataEncoder()

# tensorboard visualize
writer = SummaryWriter(log_dir=args.logdir)

t0 = time.time()

for epoch in range(start_epoch, 10000):
    if iteration > args.max_iter:
        break

    for inputs, loc_targets, cls_targets in trainloader:
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)

        loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss = loc_loss + cls_loss
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            t1 = time.time()
            print('iter ' + repr(iteration) + ' (epoch ' + repr(epoch) + ') || loss: %.4f || l loc_loss: %.4f || l cls_loss: %.4f (Time : %.1f)'\
                 % (loss.sum().item(), loc_loss.sum().item(), cls_loss.sum().item(), (t1 - t0)))
            t0 = time.time()
            writer.add_scalar('loc_loss', loc_loss.sum().item(), iteration)
            writer.add_scalar('cls_loss', cls_loss.sum().item(), iteration)
            writer.add_scalar('loss', loss.sum().item(), iteration)

            # show inference image in tensorboard
            infer_img = np.transpose(inputs[0].cpu().numpy(), (1,2,0))
            infer_img *= var
            infer_img += mean
            infer_img *= 255.
            infer_img = np.clip(infer_img, 0, 255)
            infer_img = infer_img.astype(np.uint8)
            h, w, _ = infer_img.shape

            boxes, labels, scores = encoder.decode(loc_preds[0], cls_preds[0], (w,h))
            boxes = boxes.reshape(-1, 4, 2).astype(np.int32)
            
            if boxes.shape[0] is not 0:
                infer_img = cv2.polylines(infer_img, boxes, True, (0,255,0), 4)

            writer.add_image('image', infer_img, iteration)
            writer.add_scalar('input_size', h, iteration)
            writer.add_scalar('learning_rate', cur_lr, iteration)

            t0 = time.time()

        if iteration % args.save_interval == 0 and iteration > 0:
            print('Saving state, iter : ', iteration)
            state = {
                'net': net.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration' : iteration,
                'epoch': epoch,
                'lr' : cur_lr,
                'step_index' : step_index
            }
            model_file = args.save_folder + 'ckpt_' + repr(iteration) + '.pth'
            torch.save(state, model_file)

        if iteration in stepvalues:
            step_index += 1
            cur_lr = adjust_learning_rate(cur_lr, optimizer, args.gamma, step_index)

        if iteration > args.max_iter:
            break

        if args.evaluation and iteration % args.eval_step == 0:
            try:
                if pEval is None:
                    print("Evaluation started at iteration {} on IC15...".format(iteration))
                    eval_cmd = "CUDA_VISIBLE_DEVICES=" + str(args.eval_device) + \
                                    " python eval.py" + \
                                    " --tune_from=" + args.save_folder + 'ckpt_' + repr(iteration) + '.pth' + \
                                    " --input_size=1024" + \
                                    " --output_zip=result_temp1"

                    pEval = Popen(eval_cmd, shell=True, stdout=PIPE, stderr=PIPE)

                elif pEval.poll() is not None:
                    (scorestring, stderrdata) = pEval.communicate()

                    hmean = float(str(scorestring).strip().split(":")[3].split(",")[0].split("}")[0].strip())

                    writer.add_scalar('test_hmean', hmean, iteration)
                    
                    print("test_hmean for {}-th iter : {:.4f}".format(iteration, hmean))

                    if pEval is not None:
                        pEval.kill()
                    pEval = None

            except Exception as e:
                print("exception happened in evaluation ", e)
                if pEval is not None:
                    pEval.kill()
                pEval = None

        iteration += 1


