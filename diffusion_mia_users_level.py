import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam

cudnn.benchmark = True
cudnn.deterministic = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from mia_lib.mia_util import get_dataloaders
from mia_lib.mia_model import Adversary
from mia_lib.mia_train import mia_train
# from mia_lib.mia_eval import mia_evaluate
from mia_lib.person_wise_inference import mia_evaluate

def save_checkpoints(args, attacker, epoch, save_name):
    obj = {
        'net':attacker.state_dict(),
        'epoch': epoch
    }
    torch.save(obj, os.path.join(args.output_dir, save_name))

def load_checkpoints(args, attacker, save_name):
    obj = torch.load(os.path.join(args.output_dir, save_name))
    attacker.load_state_dict(obj['net'])
    return attacker
    


def get_args_parser():

    parser = argparse.ArgumentParser('Diffusion model & Membership inference attack')
    parser.add_argument('--datadir',type=str, default='/root/paddlejob/workspace/env_run/output/move2t7/diffusion_inference_feature', help='especially')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100, help='attacker training epoch')
    parser.add_argument('--num_generated_images', type=int, default=64, help='the number of generated image')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--feature_type',type=str, default='both', help='feature type to use')
    parser.add_argument('--output_dir',type=str, default='./output_dir', help='place to save checkpoint')
    return parser


def main(args):
    
    
    print('mdkir output dir')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    train_member_loader, test_member_loader, train_non_member_loader, test_non_member_loader = get_dataloaders(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    
    # attacker = Adversary(args.num_generated_images*2 if args.feature_type=="both" else args.num_generated_images).to(device)
    attacker = Adversary(args.num_generated_images, args.feature_type).to(device)
    print('instantiate adversary', attacker)

    optimizer = Adam(attacker.parameters(), lr = args.lr, weight_decay=args.l2)


    size = len(train_member_loader)
    best_acc = 0

    # test
    acc_list = []
    pre_list = []
    rec_list = []
    attacker = load_checkpoints(args, attacker, 'last.pth')
    for i in range(3):
        private_testset = enumerate(zip(test_member_loader, test_non_member_loader))
        acc, pre, recal = mia_evaluate(args, attacker, device, private_testset, is_test_set=True)
        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(recal)

    acc = np.array(acc_list)
    pre = np.array(pre_list)
    rec = np.array(rec_list)
    if len(pre) > 0 and len(rec) > 0 and (pre.mean() + rec.mean()) != 0:
        f1_score = 2 * pre.mean() * rec.mean() / (pre.mean() + rec.mean())
    else:
        f1_score = 0

    print(f"Accuracy: {acc.mean():.4f} ± {acc.std():.4f}",  
        f"Precision: {pre.mean():.4f} ± {pre.std():.4f}",  
        f"Recall: {rec.mean():.4f} ± {rec.std():.4f}",  
        f"F1 Score: {f1_score:.4f}")

        
    attacker = load_checkpoints(args, attacker, 'best.pth')
    acc_list = []
    pre_list = []
    rec_list = []

    for i in range(3):
            private_testset = enumerate(zip(test_member_loader, test_non_member_loader))
            acc, pre, recal = mia_evaluate(args, attacker, device, private_testset, is_test_set=True)
            acc_list.append(acc)
            pre_list.append(pre)
            rec_list.append(recal)

    acc = np.array(acc_list)
    pre = np.array(pre_list)
    rec = np.array(rec_list)

    if len(pre) > 0 and len(rec) > 0 and (pre.mean() + rec.mean()) != 0:
        f1_score = 2 * pre.mean() * rec.mean() / (pre.mean() + rec.mean())
    else:
        f1_score = 0

    print(f"Accuracy: {acc.mean():.4f} ± {acc.std():.4f}",  
        f"Precision: {pre.mean():.4f} ± {pre.std():.4f}",  
        f"Recall: {rec.mean():.4f} ± {rec.std():.4f}",  
        f"F1 Score: {f1_score:.4f}")

    print('finish')


if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = get_args_parser()
    args = args.parse_args()
    main(args) 

