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
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
model, preprocess = clip.load("ViT-L-14-336px.pt", device=device)



device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("/root/paddlejob/workspace/env_run/output/blip2-opt-6.7b-coco")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "/root/paddlejob/workspace/env_run/output/blip2-opt-6.7b-coco", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT


num = 64

## All captions are in a file 
txt_path = "liaon_dataset/member_caption"

image_path = "liaon_dataset/member"

des_path = "diffusion_inference_feature_sdv15_no_text/member"


def extract_feature(image_name):

    print(image_name)
    target_image_path = os.path.join(image_path, image_name)
    target_image = Image.open(target_image_path) # online_process_image(url)

    inputs = processor(images=target_image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    target_image = preprocess(target_image).unsqueeze(0).to(device)

    caption = generated_text 

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

    image_list = os.listdir(image_path)
    print(len(image_list))

    # list_B = os.listdir("diffusion_inference_feature_sdv15_no_text/member") # + os.listdir("diffusion_inference_feature_sdv14/non_member")
    # print(len(list_B))
    # image_list = [x for x in image_list if (x.replace(".jpg", ".pth") not in list_B)]

    random.seed(127)
    random.shuffle(image_list)
    length= 2500
    i = args.index
    print(i, len(image_list))
    print(image_list[0])
    image_list = image_list[length*i:length*(i+1)]

    for image_name in image_list:
        extract_feature(image_name)
