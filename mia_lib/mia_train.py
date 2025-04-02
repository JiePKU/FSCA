import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from itertools import cycle
# from log import  print_and_log
import time
from mia_lib.mia_util import  obtain_membership_feature

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self,pred,label):
        loss = self.ce(pred,label)
        return loss,torch.Tensor([0])

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mia_train(args, adversary, device, train_private_enum, optimizer_mia, \
                size , num_batchs=1000):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    adversary.train()

    end = time.time()
    first_id = -1
    # when short dataloader is over, thus end
    for batch_idx, ((member_image_inputs,  member_text_inputs), (non_member_image_inputs, non_member_text_inputs)) in train_private_enum:

        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)

        member_image_inputs = member_image_inputs.to(device) 
        non_member_image_inputs = non_member_image_inputs.to(device) 
    
        member_text_inputs = member_text_inputs.to(device) 
        non_member_text_inputs = non_member_text_inputs.to(device) 
    
        with torch.no_grad():

            member_features = obtain_membership_feature(member_image_inputs, member_text_inputs, args.feature_type)
            non_member_features = obtain_membership_feature(non_member_image_inputs, non_member_text_inputs, args.feature_type)
            
            v_is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.ones(member_features.size(0)), np.zeros(non_member_features.size(0)))), [-1, 1])).to(device).float()

            attack_model_input = torch.cat((member_features, non_member_features))
            
        ## train NN model
        r = np.arange(v_is_member_labels.size()[0]).tolist()
        random.shuffle(r)
        attack_model_input = attack_model_input[r]
        v_is_member_labels = v_is_member_labels[r]
        member_output = adversary(attack_model_input)

        loss = F.binary_cross_entropy(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.item(), member_output.size(0))
        top1.update(prec1, member_output.size(0))

        # compute gradient and do SGD step
        optimizer_mia.zero_grad()
        loss.backward()
        optimizer_mia.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx - first_id > num_batchs:
            break

        # plot progress
        if batch_idx % 10 == 0:
            print(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))

    return (losses.avg, top1.avg)




