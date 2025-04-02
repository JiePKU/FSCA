"""
To extract feature of user's image and generated image with clip model
"""

from diffusers import StableDiffusionPipeline
import torch
import requests
from PIL import Image
from io import BytesIO
import random
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import clip  ## openclip as well
import torch
import clip
import os
from PIL import Image
from multiprocessing import Pool 

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "coco_m_real_sdv14"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")

model, preprocess = clip.load("ViT-L-14-336px.pt", device=device)

num = 64

file_path = "coco_dataset/coco_ori_test/metadata.jsonl"
des_path = "coco_m_real_diffusion_inference_feature/non_member"
image_path = "coco_dataset/coco_ori_test"


def extract_feature(pair):

    
    image_name = pair['file_name']
    print(image_name)
    caption = pair['text']

    target_image_path = os.path.join(image_path, image_name)
    target_image = Image.open(target_image_path) # online_process_image(url)
    target_image = preprocess(target_image).unsqueeze(0).to(device)


    feature_list = []
    feature_list.append(target_image)
    for ind in range(num):
        generated_image = pipe(caption).images[0]
        generated_image = preprocess(generated_image).unsqueeze(0).to(device)
        feature_list.append(generated_image)

    ## cal score per sample
    batched_images = torch.cat(feature_list, dim=0) ## B C
    with torch.no_grad():
        batched_features = model.encode_image(batched_images)
        text = clip.tokenize([caption]).to(device)
        text_feature = model.encode_text(text)

    path = os.path.join(des_path, image_name[:-3] + 'pth')
    
    obj = {"image_feature": batched_features, "text_feature":text_feature , "caption":caption}
    torch.save(obj, path)   ## the first is target feature, while the rest is feature of generated images
    

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser('Diffusion model & Membership inference attack')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()


    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(data[:2])

    print(len(data))

    # list_B = os.listdir("diffusion_inference_feature/non_member") + os.listdir("diffusion_inference_feature_sdv14/non_member")
    # print(len(list_B))
    # image_list = [x for x in image_list if (x.replace(".jpg", ".pth") not in list_B)]

    random.seed(2137)
    random.shuffle(data)
    length=2500
    i = args.index
    print(i, len(data))
    print(data[0])
    data = data[length*i:length*(i+1)]

    for pair in data:
        extract_feature(pair)
