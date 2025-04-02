
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

class CustomImageDataset(Dataset):
    def __init__(self, feature_dir):
        
        """
        Args:
            image_dir (str):
            transform (callable, optional): 
        """

        self.feature_dir = feature_dir
        self.features = os.listdir(self.feature_dir)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 
        Returns:
            feature (Tensor)
        """
        feature_name = self.features[idx]
        feature_path = os.path.join(self.feature_dir, feature_name)
        
        # 加载图像
        # print(feature_path)

        feature_dict = torch.load(feature_path, map_location='cuda')
        image_feature = None
        text_feature = None
        keys = feature_dict.keys()

        if "image_feature" in keys:
            image_feature = feature_dict["image_feature"].float() # N C
            # N, C = image_feature.size()
            # image_feature = image_feature[:N//2,:]

        if "text_feature" in keys:
            text_feature = feature_dict["text_feature"].float() # 1 C
        
        return image_feature, text_feature


# class CustomImageDataset(Dataset):
#     def __init__(self, feature_dir):
#         """
#         Args:
#             feature_dir (str): 
#         """
#         self.feature_dir = feature_dir
#         self.features = os.listdir(self.feature_dir)

#         self.similarities = []
#         for idx, feature_name in enumerate(self.features):
#             feature_path = os.path.join(self.feature_dir, feature_name)
#             feature_dict = torch.load(feature_path, map_location='cpu')

#             image_feature = feature_dict.get("image_feature")
#             text_feature = feature_dict.get("text_feature")

#             if image_feature is not None and text_feature is not None:
#                 similarity = F.cosine_similarity(image_feature[:1, :].float().flatten(), text_feature.float().flatten(), dim=0)
#             else:
#                 similarity = torch.tensor(-1.0) 
            
#             self.similarities.append((idx, similarity.item()))

#         self.similarities.sort(key=lambda x: x[1], reverse=False)
#         self.sorted_indices = [idx for idx, _ in self.similarities]


#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
        
#         sorted_index = self.sorted_indices[idx]
#         feature_name = self.features[sorted_index]
#         similarity = self.similarities[idx][1]

#         # print(feature_name)
#         feature_path = os.path.join(self.feature_dir, feature_name)
        
#         feature_dict = torch.load(feature_path, map_location='cuda')
#         image_feature = feature_dict.get("image_feature", None)
#         text_feature = feature_dict.get("text_feature", None)

#         if image_feature is not None:
#             image_feature = image_feature.float()
        
#         if text_feature is not None:
#             text_feature = text_feature.float()
        
#         return image_feature, text_feature, torch.Tensor([similarity]).float().cuda()


def obtain_membership_feature(image_feature, text_feature, feature_type="both"):

    """
    input: feature B (L+1) C
    output: member feature B L
    """
    norm = True

    if norm:
        image_feature /= image_feature.norm(dim=2, keepdim=True)
        text_feature /= text_feature.norm(dim=2, keepdim=True) 
    
    target_image_feature = image_feature[:,:1,:] # B 1 C
    generated_image_feature = image_feature[:,1:,:] # B L C

    # B, N, C = generated_image_feature.size()
    # generated_image_feature = generated_image_feature[:, :N, :]
    
    feature_type="both"

    if feature_type=="both":
        # image_feature
        cosin_score = (generated_image_feature * target_image_feature).sum(dim=2) # B L
        sorted_consin_score = torch.sort(cosin_score, dim=1, descending=True)[0]

        # text_feature
        target_clip_score = (text_feature * target_image_feature).sum(dim=2) # B 1
        generated_clip_score = (text_feature * generated_image_feature).sum(dim=2) # B L
        difference_clip_score = generated_clip_score-target_clip_score
        sorted_difference_clip_score = torch.sort(difference_clip_score, dim=1, descending=True)[0]

        return torch.cat([sorted_consin_score, sorted_difference_clip_score], dim=1)


def get_dataloaders(args):
    member_dir = os.path.join(args.datadir, 'member')
    non_member_dir = os.path.join(args.datadir, 'non_member')

    member_set = CustomImageDataset(member_dir)
    non_member_set = CustomImageDataset(non_member_dir)

    member_set_length = len(member_set)
    
    num = 2
    random.seed(42)
    train_member_set_index = random.sample(range(member_set_length), int(member_set_length / num))
    # unseen part to test inference model
    test_member_set_index = list(set(range(member_set_length)).difference(set(train_member_set_index)))
    
    train_member_set = torch.utils.data.Subset(member_set, train_member_set_index)
    test_member_set = torch.utils.data.Subset(member_set, test_member_set_index)
    
    train_member_loader = torch.utils.data.DataLoader(train_member_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_member_loader = torch.utils.data.DataLoader(test_member_set, batch_size=10, shuffle=False, num_workers=4)
    
    non_member_set_length = len(non_member_set)

    random.seed(18)
    train_non_member_index = random.sample(range(non_member_set_length), int(non_member_set_length/num))
    test_non_member_index = list(set(range(non_member_set_length)).difference(set(train_non_member_index)))
    
    train_non_member_set = torch.utils.data.Subset(non_member_set, train_non_member_index)
    test_non_member_set = torch.utils.data.Subset(non_member_set, test_non_member_index)

    train_non_member_loader = torch.utils.data.DataLoader(train_non_member_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_non_member_loader = torch.utils.data.DataLoader(test_non_member_set, batch_size=10, shuffle=False, num_workers=4)
    
    return train_member_loader, test_member_loader, train_non_member_loader, test_non_member_loader