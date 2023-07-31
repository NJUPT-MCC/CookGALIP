import torch
import clip
import os
import argparse
from PIL import Image
import numpy as np

def get_imgs(img_path):
    img_list=[]
    imgs=os.listdir(img_path)
    for img in imgs:
        img_list.append(os.path.join(img_path,img))
    return img_list

def get_txts(txt_path):
    texts=[]
    with open(txt_path,'r') as f:
        line=f.readline().strip('\n')
        while line:
            texts.append(line)
            line=f.readline()
    return texts


def cal_CLIPsim(img_list,texts,device):
    model,preprocess=clip.load("ViT-B/32",device=device)
    text=clip.tokenize(texts,truncate=True).to(device)
    clip_sim=[]
    idx=0
    for file in img_list:
        image=preprocess(Image.open(file)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            clip_sim.append(probs[0][idx])
        idx+=1
    print(clip_sim)
    print(np.mean(clip_sim))
        


if __name__=="__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument('--img_path',type=str,default='')
    parse.add_argument('--txt_path',type=str,default='')
    args = parse.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    img_list=get_imgs(args.img_path)
    texts=get_txts(args.txt_path)
    cal_CLIPsim(img_list,texts,device)
    




